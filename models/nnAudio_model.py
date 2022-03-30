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
from tasks.speechcommand_maskout import SpeechCommand_maskout
from tasks.Timit import Timit
from tasks.Timit_maskout import Timit_maskout
from speechbrain.processing.features import InputNormalization

class BCResNet_nnAudio(SpeechCommand):
    def __init__(self,cfg_model): 
        #in main script, will pass cfg_spec to model
        super().__init__()
        self.fastaudio_filter = None
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
        self.conv4 = nn.Conv2d(32,  self.cfg_model.args.output_dim, 1, bias=False)
                
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        self.criterion = nn.CrossEntropyLoss()        
        #self.mel_layer use in validation step for [mel_filter_banks = self.mel_layer.mel_basis]
        #self.criterion use in traning & validation step        
        
        if self.cfg_model.random_mel == True:
            nn.init.kaiming_uniform_(self.mel_layer.mel_basis, mode='fan_in')
            self.mel_layer.mel_basis.requires_grad = False
            torch.relu_(self.mel_layer.mel_basis)
            self.mel_layer.mel_basis.requires_grad = True
            #for randomly initialize mel bases        
        
    def forward(self, x):        
        #x: 2D [Batch_size,16000]
        spec = self.mel_layer(x) 
        #spec: 3D [B,F(40),T]
        
        spec = torch.log(spec+1e-10)         
        spec = spec.unsqueeze(1)
        #spec: bcoz conv1 need 4D [B,1,F,T]

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

        out = self.conv4(out)   
        #out: 4D [8, 35, 1, 1]
        out = out.squeeze(2).squeeze(2)  
        #out: 2D        
        #crossentropy expect[B, C], so need to squeeze to be 2D
        #ref:https://pytorch.org/docs/1.9.1/generated/torch.nn.CrossEntropyLoss.html

        spec = spec.squeeze(1) 
        #spec: from 4D [B,1,F,T] to 3D [B,F,T]
        #the return spec is for plot log_images, so need 3D

        return out, spec



class BCResNet_nnAudio_maskout(SpeechCommand_maskout):
    def __init__(self,cfg_model): 
        super().__init__()
        self.fastaudio_filter = None
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
        self.conv4 = nn.Conv2d(32,  self.cfg_model.args.output_dim, 1, bias=False)
        
        self.STFT_layer = STFT(**cfg_model.stft_args)
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        self.criterion = nn.CrossEntropyLoss()        
          
        
    def forward(self, x):        
        #x: 2D [Batch_size,16000]
        stft = self.STFT_layer(x)
       
        stft[:,self.cfg_model.maskout_start:self.cfg_model.maskout_end] = 0 #mask out       
        spec = torch.matmul(self.mel_layer.mel_basis, stft) 
        #spec: 3D [B,F(40),T]

        spec = torch.log(spec+1e-10)         
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

        out = self.conv4(out)   
        #out: 4D
        out = out.squeeze(2).squeeze(2)  
        #out: 2D bcoz crossentropy expect[B, C]

        spec = spec.squeeze(1)
        #spec: from 4D [B,1,F,T] to 3D [B,F,T]
        #the return spec is for plot log_images, so need 3D

        return out, spec, stft


class BCResNet_nnAudio_ASR(Timit):
    def __init__(self,cfg_model,text_transform,lr): 
        super().__init__(text_transform,lr)
        self.fastaudio_filter = None
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
        self.conv4 = nn.Conv2d(32*2,  self.cfg_model.args.output_dim, 1, bias=False)
                
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        self.criterion = nn.CrossEntropyLoss()        
        #self.mel_layer use in validation step for [mel_filter_banks = self.mel_layer.mel_basis]
        #self.criterion use in traning & validation step
        
        self.lstmlayer = nn.LSTM(input_size=32,
                                 hidden_size=32,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        
        if self.cfg_model.random_mel == True:
            nn.init.kaiming_uniform_(self.mel_layer.mel_basis, mode='fan_in')
            self.mel_layer.mel_basis.requires_grad = False
            torch.relu_(self.mel_layer.mel_basis)
            self.mel_layer.mel_basis.requires_grad = True
            #for randomly initialize mel bases        
        
    def forward(self, x):        
        #x: 2D [Batch_size,16000]
        spec = self.mel_layer(x) 
        #spec: 3D [B,F(40),T]

        spec = torch.log(spec+1e-10)               
        spec = spec.unsqueeze(1)
        #spec: 4D [B,1,F,T]

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

        out = self.conv3(out)    #out: 4D [B, 32, 1, T]
        out = out.squeeze(2)     #out: 3D [B, 32, T] bcoz lstmlayer need 3D
        out = out.transpose(1,2) #out: 3D [B, T, 32]
        out, _ = self.lstmlayer(out) 
        out = out.transpose(1,2) #out: 3D [B,32, T]
        out = out.unsqueeze(2)   #out: 4D for conv4
        #out = out.mean(-1, keepdim=True) 
        #diff setting from KWS, remove mean(). since after taking mean, time dimension is gone 

        out = self.conv4(out)    #out: 4D [B, C, 1, T]
        
        out = out.squeeze(2)     #out: 3D [B,num_classes,T]
        out = out.transpose(1,2) #out: 3D [B,T,num_classes]
        
        spec = spec.squeeze(1)
        #spec: from 4D [B,1,F,T] to 3D [B,F,T]

        return out, spec


    
class Linearmodel_nnAudio(SpeechCommand):
    def __init__(self,cfg_model): 
        super().__init__()
        self.fastaudio_filter = None
        self.optimizer_cfg = cfg_model.optimizer
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        
        self.criterion = nn.CrossEntropyLoss()
        self.cfg_model = cfg_model
        self.linearlayer = nn.Linear(self.cfg_model.args.input_dim, self.cfg_model.args.output_dim)
        #cfg.model.args.input_dim will be calculated in training script 
   
        if self.cfg_model.random_mel == True:
            nn.init.kaiming_uniform_(self.mel_layer.mel_basis, mode='fan_in')
            self.mel_layer.mel_basis.requires_grad = False
            torch.relu_(self.mel_layer.mel_basis)
            self.mel_layer.mel_basis.requires_grad = True
            #for randomly initialize mel bases
    
    def forward(self, x): 
        #x: 2D [B, 16000]
        spec = self.mel_layer(x)  
        #spec: 3D [B, F40, T101]
        
        spec = torch.log(spec+1e-10)
        flatten_spec = torch.flatten(spec, start_dim=1) 
        #flatten_spec: 2D [B, F*T(40*101)] 
        #start_dim: flattening start from 1st dimention
        
        out = self.linearlayer(flatten_spec) 
        #out: 2D [B,number of class(12)] 
                               
        return out, spec               
                                                     
    
class Linearmodel_nnAudio_ASR(Timit):
    def __init__(self,cfg_model,text_transform,lr): 
        super().__init__(text_transform,lr)

        self.fastaudio_filter = None
        self.optimizer_cfg = cfg_model.optimizer
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        
        self.criterion = nn.CrossEntropyLoss()
        self.cfg_model = cfg_model
        self.linearlayer = nn.Linear(self.cfg_model.args.hidden_dim*2, self.cfg_model.args.output_dim)
        self.lstmlayer = nn.LSTM(input_size=self.cfg_model.args.input_dim,
                                 hidden_size=self.cfg_model.args.hidden_dim,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)        
        #cfg.model.args.input_dim will be calculated in main script 
        #add LSTM layer for ASR task
        
        if self.cfg_model.random_mel == True:
            nn.init.kaiming_uniform_(self.mel_layer.mel_basis, mode='fan_in')
            self.mel_layer.mel_basis.requires_grad = False
            torch.relu_(self.mel_layer.mel_basis)
            self.mel_layer.mel_basis.requires_grad = True
            #for randomly initialize mel bases  
    
    def forward(self, x): 
        #x: 2D [B, 16000]
        spec = self.mel_layer(x)  
        #spec: 3D [B, F40, T101]
        
        spec = torch.log(spec+1e-10)
        spec = spec.transpose(1,2)
        #spec: 3D [B, T, F40] 

        x, h = self.lstmlayer(spec) 
        #x: [B, T, hidden*2] 
        #h: [B, hideen*2]
        
        out = self.linearlayer(x) 
        #out: 2D [B,number of class(62)]
       
        return out, spec                                       
        
        #for ASR task, predict at each time stamp, so no need to flatten
        #lstmlayer retuen x as final hidden state/short term memory for each element in the batch
        #lstmlayer return h as final cell state/long term memory for each element in the batch
    
    
class Linearmodel_nnAudio_maskout(SpeechCommand_maskout):
    def __init__(self,cfg_model): 
        super().__init__()
        self.fastaudio_filter = None
        self.STFT_layer = STFT(**cfg_model.stft_args)
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        self.optimizer_cfg = cfg_model.optimizer
                
        self.criterion = nn.CrossEntropyLoss()
        self.cfg_model = cfg_model
        self.linearlayer = nn.Linear(self.cfg_model.args.input_dim, self.cfg_model.args.output_dim)
        #cfg.model.args.input_dim will be calculated in main script            
        
    def forward(self, x): 
        #x: 2D [B, 16000]
        stft = self.STFT_layer(x) 
        #stft: 3D [B, 241, T(101)]
        
        stft[:,self.cfg_model.maskout_start:self.cfg_model.maskout_end] = 0 #mask out
        spec = torch.matmul(self.mel_layer.mel_basis, stft)
        spec = torch.log(spec+1e-10)

        flatten_spec = torch.flatten(spec, start_dim=1) 
        #flatten_spec: 2D [B, F*T(40*101)] 
        #start_dim: flattening start from 1st dimention
        out = self.linearlayer(flatten_spec) 
        #out: 2D [B,number of class(12)] 
                               
        return out, spec ,stft              

    
class Linearmodel_nnAudio_ASR_maskout(Timit_maskout):
    def __init__(self,cfg_model,text_transform,lr): 
        super().__init__(text_transform,lr)

        self.fastaudio_filter = None
        self.optimizer_cfg = cfg_model.optimizer
        self.STFT_layer = STFT(**cfg_model.stft_args)
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        
        self.criterion = nn.CrossEntropyLoss()
        self.cfg_model = cfg_model
        self.linearlayer = nn.Linear(self.cfg_model.args.hidden_dim*2, self.cfg_model.args.output_dim)
        self.lstmlayer = nn.LSTM(input_size=self.cfg_model.args.input_dim,
                                 hidden_size=self.cfg_model.args.hidden_dim,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        #cfg.model.args.input_dim will be calculated in main script 
        #add LSTM layer for ASR task
            
    def forward(self, x): 
        #x: 2D [B, 16000] 
        stft = self.STFT_layer(x)  
        #stft: 3D [B, 241, T]

        stft[:,self.cfg_model.maskout_start:self.cfg_model.maskout_end] = 0 #mask out
        spec = torch.matmul(self.mel_layer.mel_basis, stft) 
        #spec: 3D [B, F40, T] 
  
        spec = torch.log(spec+1e-10)
        spec = spec.transpose(1,2)
        #spec: 3D [B, T, F40] 

        x, h = self.lstmlayer(spec)
        #x: [B, T, hidden*2] 
        #h: [B, hideen*2]
        out = self.linearlayer(x)
        #out: 2D [B,number of class(62)]
        
        return out, spec, stft                             
    
    
    
    
    
    
    
    
    
    
    