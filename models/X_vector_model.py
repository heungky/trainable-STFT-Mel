import torch.nn as nn
from models.tdnn import TDNN
import torch
import torch.nn.functional as F
from .lightning_module import SpeechCommand
import sys
sys.path.insert(0, './package/Installation/')
from nnAudio.features.mel import MelSpectrogram, STFT, MFCC


class X_vector(SpeechCommand):
    def __init__(self, input_dim = 40, num_classes=35,cfg_model=None):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer_cfg = cfg_model.optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.mel_layer = MelSpectrogram(**cfg_model.spec_args)
#        self.mel_layer = MFCC(**cfg_model.spec_args)
        self.fastaudio_filter=None
        
    def forward(self, inputs):

        #inputs is 2D training_step_batch['waveforms' [B,16000]
        #after self.mel_layer(inputs) --> 3D [B,F,T]
        spec = self.mel_layer(inputs)
        spec = torch.log(spec+1e-10)
        spec = spec.transpose(1, 2)
        #become B, T, F
        
       
        tdnn1_out = self.tdnn1(spec)
        tdnn2_out = self.tdnn2(tdnn1_out)        
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.softmax(self.output(x_vec))
        spec = spec.transpose(1,2)
        #become B, F, T
        return predictions,spec
    
    
    