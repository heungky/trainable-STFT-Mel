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

class BCResNet(SpeechCommand):
    def __init__(self, no_output_chan, cfg_model): 
        #in main script, will pass no_output_chan, cfg_spec to model
        super().__init__()
        self.fastaudio_filter = None
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
                
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        self.criterion = nn.CrossEntropyLoss()
        
        #self.mel_layer use in validation step for [mel_filter_banks = self.mel_layer.mel_basis]
        #self.criterion use in traning & validation step
        
    def forward(self, x):        
        # x [Batch_size,16000]
#         self.mel_layer.mel_basis = torch.clamp(self.mel_layer.mel_basis, 0, 1)
        spec = self.mel_layer(x) # [B,F,T]
        # (B, 40, T)
        #print(f'{spec.max()=}')
        #print(f'{spec.min()=}')
        
        #spec = torch.relu(spec)
        
        spec = torch.log(spec+1e-10)
        spec = spec.unsqueeze(1)
#x is training_step_batch['waveforms' [B,16000]
#after self.mel_layer(x) --> 3D [B,F,T]
#after spec.unsqueeze(1) --> 4D bcoz conv1 need 4D [B,1,F,T]

#        print('INPUT SHAPE:', x.shape)
        out = self.conv1(spec)

#        print('BLOCK1 INPUT SHAPE:', out.shape)
        out = self.block1_1(out)
        out = self.block1_2(out)

#        print('BLOCK2 INPUT SHAPE:', out.shape)
        out = self.block2_1(out)
        out = self.block2_2(out)

#        print('BLOCK3 INPUT SHAPE:', out.shape)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

#        print('BLOCK4 INPUT SHAPE:', out.shape)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

#        print('Conv2 INPUT SHAPE:', out.shape)
        out = self.conv2(out)

#        print('Conv3 INPUT SHAPE:', out.shape)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

#        print('Conv4 INPUT SHAPE:', out.shape)
        out = self.conv4(out)   #4D
        out = out.squeeze(2).squeeze(2)  #2D
        spec = spec.squeeze(1)

#        print('OUTPUT SHAPE:', out.shape)
#        OUTPUT SHAPE: torch.Size([8, 35, 1, 1])  out

#crossentropy expect[B, C], so need to squeeze to be 2 dimension
#ref:https://pytorch.org/docs/1.9.1/generated/torch.nn.CrossEntropyLoss.html
#old spec :4D [B,1,F,T] , the return spec is for plot log_images, so need 3D
        return out, spec



class BCResNet_exp(SpeechCommand):        
    def __init__(self, no_output_chan, cfg_model): 
        #in main script, will pass no_output_chan, cfg_spec to model
        super().__init__()
        self.fastaudio_filter = None
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
        
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
        self.criterion = nn.CrossEntropyLoss()
        
        #self.mel_layer use in validation step for [mel_filter_banks = self.mel_layer.mel_basis]
        #self.criterion use in traning & validation step

    def on_after_backward(self):
        # freeze bins from 20 onward
        # i.e. only updates bins 0-19
#         print(f"{self.mel_layer.mel_basis.shape=}")
        
#        self.mel_layer.mel_basis.grad[:20] = 0 # freeze first 20 triangles    
         self.mel_layer.mel_basis.grad[20:] = 0  # freeze last 20 triangles
#mel_basis is mel_spectrogram info
#mel_basis size is 241:40   ****40 mel filter bank, each with 241 datapoint 
        
    def forward(self, x):        
        # x [Batch_size,16000]
#         self.mel_layer.mel_basis = torch.clamp(self.mel_layer.mel_basis, 0, 1)
        spec = self.mel_layer(x) # [B,F,T]
        # (B, 40, T)
        #print(f'{spec.max()=}')
        #print(f'{spec.min()=}')
        
        #spec = torch.relu(spec)
        
        spec = torch.log(spec+1e-10)
        spec = spec.unsqueeze(1)
#x is training_step_batch['waveforms' [B,16000]
#after self.mel_layer(x) --> 3D [B,F,T]
#after spec.unsqueeze(1) --> 4D bcoz conv1 need 4D [B,1,F,T]

#        print('INPUT SHAPE:', x.shape)
        out = self.conv1(spec)

#        print('BLOCK1 INPUT SHAPE:', out.shape)
        out = self.block1_1(out)
        out = self.block1_2(out)

#        print('BLOCK2 INPUT SHAPE:', out.shape)
        out = self.block2_1(out)
        out = self.block2_2(out)

#        print('BLOCK3 INPUT SHAPE:', out.shape)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

#        print('BLOCK4 INPUT SHAPE:', out.shape)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

#        print('Conv2 INPUT SHAPE:', out.shape)
        out = self.conv2(out)

#        print('Conv3 INPUT SHAPE:', out.shape)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

#        print('Conv4 INPUT SHAPE:', out.shape)
        out = self.conv4(out)   #4D
        out = out.squeeze(2).squeeze(2)  #2D
        spec = spec.squeeze(1)

        return out, spec