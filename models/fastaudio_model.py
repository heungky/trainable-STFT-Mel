import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
import sys
sys.path.insert(0, './package/Installation/')
from nnAudio.features.mel import MelSpectrogram, STFT
import sys
import matplotlib.pyplot as plt
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from models.custom_model import Filterbank #use for fastaudio model 
from .utils import SubSpectralNorm, BroadcastedBlock, TransitionBlock
from tasks.speechcommand import SpeechCommand
from speechbrain.processing.features import InputNormalization
from tasks.Timit import Timit

class BCResNet_Fastaudio(SpeechCommand):
    def __init__(self, cfg_model):
        super().__init__()        
        self.mel_layer = STFT(**cfg_model.spec_args)   
        #STFT from nnAudio.features.stft
        #stft output is complex number 
        #'Magnitude' = abosulute value of complex number 
              
        self.fastaudio_filter = Filterbank(**cfg_model.fastaudio)
        self.optimizer_cfg = cfg_model.optimizer
        self.cfg_model = cfg_model
                
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16, 8)
        self.block1_2 = BroadcastedBlock(8)

        self.block2_1 = TransitionBlock(8, 12, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12, 16, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16, 20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20, 20, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20, 32, 1, bias=False)
        self.conv4 = nn.Conv2d(32, self.cfg_model.args.output_dim, 1, bias=False)
        

        self.criterion = nn.CrossEntropyLoss()        
        #self.mel_layer use in validation step for [mel_filter_banks = self.mel_layer.mel_basis]
        #self.criterion use in traning & validation step
        
    def forward(self, x):        
        #x: 2D [Batch_size,16000]      
        stft_output =  self.mel_layer(x) 
        #stft_output: 3D [B, F(201), T]
      
        output = self.fastaudio_filter(stft_output.transpose(-1,-2))                                 
        #transpose to [B,T F] to use fastaudio process stft spectrogram
        #size of fbank_matrix is 201x40 [F, n_filters]

        spec = output.transpose(1,2)
        #spec: 3D [B,F(40),T] 
               
        spec = spec.unsqueeze(1)
        #spec: 4D bcoz conv1 need 4D [B,1,F,T]

        out = self.conv1(spec)

        out = self.block1_1(out)
        out = self.block1_2(out)

        out = self.block2_1(out)
        out = self.block2_2(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

        out = self.conv4(out)            #4D
        out = out.squeeze(2).squeeze(2)  
        #2D for CrossEntropy
        spec = spec.squeeze(1)
        #spec: 3D [B,F,T]

        return out, spec   

    
class BCResNet_Fastaudio_ASR(Timit):
    def __init__(self, cfg_model,text_transform,lr):
        super().__init__(text_transform,lr)        
        self.mel_layer = STFT(**cfg_model.spec_args)   
        #STFT from nnAudio.features.stft
        #stft output is complex number 
        #'Magnitude' = abosulute value of complex number 
              
        self.fastaudio_filter = Filterbank(**cfg_model.fastaudio)
        self.optimizer_cfg = cfg_model.optimizer
        self.cfg_model = cfg_model
                
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16, 8)
        self.block1_2 = BroadcastedBlock(8)

        self.block2_1 = TransitionBlock(8, 12, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12, 16, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16, 20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20, 20, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20, 32, 1, bias=False)
        self.conv4 = nn.Conv2d(32*2, self.cfg_model.args.output_dim, 1, bias=False)
        

        self.criterion = nn.CrossEntropyLoss()        
        #self.mel_layer use in validation step for [mel_filter_banks = self.mel_layer.mel_basis]
        #self.criterion use in traning & validation step
        
        self.lstmlayer = nn.LSTM(input_size=32,
                             hidden_size=32,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)
    def forward(self, x):        
        #x: 2D [Batch_size,16000]        
        stft_output =  self.mel_layer(x) 
        #stft_output: 3D [B,F,T]       
        output = self.fastaudio_filter(stft_output.transpose(-1,-2))                        
        #output: 3D [B,T,F]
        
        spec = output.transpose(1,2)
        #spec: 3D [B,F(40),T]
        
        spec = spec.unsqueeze(1)  #4D
        #spec: 4D bcoz conv1 need 4D [B,1,F,T]

        out = self.conv1(spec)

        out = self.block1_1(out)
        out = self.block1_2(out)

        out = self.block2_1(out)
        out = self.block2_2(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        out = self.conv2(out)

        out = self.conv3(out)

        out = out.squeeze(2)    
        #out: 3D [B,32,T] bcoz lstmlayer need 3D
        out = out.transpose(1,2) 
        #out: 3D [B,T,32]
        
        out, _ = self.lstmlayer(out) 
        out = out.transpose(1,2) 
        #out: 3D [B,F,T]
        out = out.unsqueeze(2)  
        #out: 4D for conv4 [B,F,1,T]

        out = self.conv4(out)    #out: 4D [B,F,1,T]
        out = out.squeeze(2)     #out: 3D [B,F,T]
        out = out.transpose(1,2) #out: 3D [B,T,F]
        spec = spec.squeeze(1)
        #spec: from 4D [B,1,F,T] to 3D [B,F,T]

        return out, spec   
    
    
    
class Linearmodel_Fastaudio(SpeechCommand):
    def __init__(self,cfg_model): 
        super().__init__()
        self.mel_layer = STFT(**cfg_model.spec_args)
        self.fastaudio_filter = Filterbank(**cfg_model.fastaudio)
        self.optimizer_cfg = cfg_model.optimizer
        
        self.criterion = nn.CrossEntropyLoss()
        self.cfg_model = cfg_model
        self.linearlayer = nn.Linear(self.cfg_model.args.input_dim, self.cfg_model.args.output_dim)
        
            
    def forward(self, x): 
        #x: 2D [B, 16000]
        stft_output =  self.mel_layer(x)  
        #stft_output: 3D [B, F(241), T(101)]
              
        output = self.fastaudio_filter(stft_output.transpose(-1,-2))     
        #from [B,F,T] to [B,T,F], use fastaudio process stft spectrogram
        #bcoz stft_output [F, T],size of fbank_matrix is [F, n_filters(40)], need transpose to multiply
        #output: 3D [B,T,F(40])

        output = output.transpose(1,2)
        #output: 3D [B,F,T] for plotting
    
        flatten_spec = torch.flatten( output, start_dim=1) 
        #flatten_spec: 2D [B, T*F] 
        #start_dim: flattening start from 1st dimention
        
        out = self.linearlayer(flatten_spec) 
        #out: 2D [B,number of class(12)] 
                                
        return out, output   
    
        #raw waveform 2D [B, 16000] -> mel-layer 3D [B, F241, T101] -> fastaudio filter [B, F40, T101] -> .transpose() [B, T101, F40] -> flatten 2D [B, T*F(101*40)] -> linear model instead of convolution [multiply by n_mels*101, output 12 class]


class Linearmodel_Fastaudio_ASR(Timit):
    def __init__(self,cfg_model,text_transform,lr): 
        super().__init__(text_transform,lr)
        self.mel_layer = STFT(**cfg_model.spec_args)
        self.fastaudio_filter = Filterbank(**cfg_model.fastaudio)
        self.optimizer_cfg = cfg_model.optimizer
        
        self.criterion = nn.CrossEntropyLoss()
        self.cfg_model = cfg_model
        self.linearlayer = nn.Linear(self.cfg_model.args.hidden_dim*2, self.cfg_model.args.output_dim)
        
        #cfg.model.args.input_dim will be calculated in main script
        #cfg_model.args.output_dim = 62 classes
        self.lstmlayer = nn.LSTM(input_size=self.cfg_model.args.input_dim,
                                 hidden_size=self.cfg_model.args.hidden_dim,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

            
    def forward(self, x): 
        #x: 2D [B, 16000]
        stft_output =  self.mel_layer(x) 
        #stft_output: 3D [B, F, T]
              
        output = self.fastaudio_filter(stft_output.transpose(-1,-2))   
        #from [B,F,T] to [B,T, F], use fastaudio process stft spectrogram
        #size of fbank_matrix is 201x40 [F, n_filters]
        #output: 3D [B(100), T, F40])  
        
        x,h = self.lstmlayer(output)
        out = self.linearlayer(x) 
        #out: 2D [B,number of class(62)] 
                                
        return out, output   
        #for ASR task, predict at each time stamp, so no need to flatten.

