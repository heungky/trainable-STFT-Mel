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
from .lightning_module import SpeechCommand

class BCResNet_Fastaudio(SpeechCommand):
    def __init__(self, no_output_chan, cfg_model):
        super().__init__()        
        self.mel_layer = STFT(**cfg_model.spec_args)   
        #STFT from nnAudio.features.stft
        #stft output is complex number 
        #'Magnitude' = abosulute value of complex number 
        
       
        self.fastaudio_filter = Filterbank(**cfg_model.fastaudio)
        self.optimizer_cfg = cfg_model.optimizer
                
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
        self.conv4 = nn.Conv2d(32, 35, 1, bias=False)
        

        self.criterion = nn.CrossEntropyLoss()
        
#self.mel_layer use in validation step for [mel_filter_banks = self.mel_layer.mel_basis]
#self.criterion use in traning & validation step
        
    def forward(self, x):        
        # x [Batch_size,16000] 2D
        #self.mel_layer.mel_basis = torch.clamp(self.mel_layer.mel_basis, 0, 1)
        
        
        stft_output =  self.mel_layer(x) #3D [B, F, T]
        torch.save(stft_output, './stft_output.pt') 
#         print(f'{stft_output.max()=}')
#         print(f'{stft_output.min()=}')
        output = self.fastaudio_filter(stft_output.transpose(-1,-2))                
        
        
        
        #[B,T F], use fastaudio process stft spectrogram
        #bcoz stft_output [201, 161], [F, T]
        #size of fbank_matrix is 201x40 [F, n_filters]
#         torch.save(stft_output, './stft_output.pt')
        torch.save(output, './output.pt')
#        print(f'{output.max()=}')
#        print(f'{output.min()=}')
        
#         sys.exit()
        spec = output.transpose(1,2)
        
        
        #old spec = self.mel_layer(x) 
        # [B,F,T] (B, 40, T)  3D
        
        #spec = torch.relu(spec) #this is for throw out negative mel_filter band value 
        
        #spec = torch.log(spec+1e-10)  #3D
        
        
        spec = spec.unsqueeze(1)  #4D
#x is training_step_batch['waveforms' [B,16000]
#after self.mel_layer(x) --> 3D [B,F,T]
#after spec.unsqueeze(1) --> 4D bcoz conv1 need 4D [B,1,F,T]

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

        out = self.conv4(out)   #4D
        out = out.squeeze(2).squeeze(2)  #2D
        spec = spec.squeeze(1)

        return out, spec   