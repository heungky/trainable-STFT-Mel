import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from nnAudio.features.mel import MelSpectrogram
import sys
import matplotlib.pyplot as plt
from torch import Tensor

class SpeechCommand(LightningModule):
    def training_step(self, batch, batch_idx):
        outputs, spec = self(batch['waveforms']) 
        loss = self.criterion(outputs, batch['labels'].squeeze(1).long())
#return outputs for calculate loss, return spec for visual
#for debug 
#torch.save(outputs, 'output.pt')
#torch.save(batch['labels'], 'label.pt')          
#sys.exit()

        acc = sum(outputs.argmax(-1) == batch['labels'].squeeze(1))/outputs.shape[0] #batch wise
        
        self.log('Train/acc', acc, on_step=False, on_epoch=True)
        if self.current_epoch==0:
            if batch_idx == 0:
                self.log_images(spec, 'Train/Spec')        
        self.log('Train/Loss', loss, on_step=False, on_epoch=True)
        return loss
        #log(graph title, take acc as data, on_step: plot every step, on_epch: plot every epoch)
       
    
         
    
    def validation_step(self, batch, batch_idx):       
        outputs, spec = self(batch['waveforms'])
        loss = self.criterion(outputs, batch['labels'].squeeze(1).long())
        
#acc = sum(outputs.argmax(-1) == batch['labels'].squeeze(1))/outputs.shape[0]
#accuracy for 
#self.log('Validation/acc', acc, on_step=False, on_epoch=True)

        self.log('Validation/Loss', loss, on_step=False, on_epoch=True)          
        if self.current_epoch==0:
            if batch_idx == 0:
                self.log_images(spec, 'Validation/Spec')
        #plot log_images for 1st epoch_1st batch
        
        output_dict = {'outputs': outputs,
                       'labels': batch['labels'].squeeze(1)}        
        return output_dict
        
        
                
        
    def validation_epoch_end(self, outputs):
        pred = []
        label = []
        for output in outputs:
            pred.append(output['outputs'])
            label.append(output['labels'])
        label = torch.cat(label, 0)
        pred = torch.cat(pred, 0)
        acc = sum(pred.argmax(-1) == label)/label.shape[0]
        self.log('Validation/acc', acc, on_step=False, on_epoch=True)    
    #use the return value from validation_step: output_dict , to calculate the overall accuracy   #epoch wise 
        
        
        
        
        
    def log_images(self, tensors, key):
        fig, axes = plt.subplots(2,2, figsize=(12,5), dpi=100)
        for ax, tensor in zip(axes.flatten(), tensors):
            ax.imshow(tensor.cpu().detach(), aspect='auto', origin='lower', cmap='jet')
        plt.tight_layout()
        self.logger.experiment.add_figure(f"{key}", fig, global_step=self.current_epoch)
        plt.close(fig)
    #plot images in TensorBoard        
          
    
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)    
    

#python Inheritance
class ModelA(SpeechCommand):
    def __init__(self, no_output_chan, cfg_spec):
        super().__init__()
        print(f"I am model A")   
        self.mel_layer = MelSpectrogram(**cfg_spec)        
        
        self.conv1 = nn.Conv2d(1,no_output_chan,5)    
        self.conv2 = nn.Conv2d(no_output_chan,16,5)
             
        self.fc1 = nn.Linear(16*22*5,120) 
        #have to follow input, x.shape before flattern: 
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,35)
        
        self.criterion = nn.CrossEntropyLoss()       
        
    def forward(self,x):
        spec = self.mel_layer(x) #will take batch['waveforms' in training_step
        spec = torch.log(spec+1e-10) #3-dimension #take log to make data more comparable 
        x = spec.unsqueeze(1)    #4-dimension     
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, spec
       
    
    
    

   
    
    
class ModelB(SpeechCommand):    
    def __init__(self, no_output_chan,cfg_spec):
        super().__init__()
        print(f"I am model B")
        self.mel_layer = MelSpectrogram(**cfg_spec)           
        self.conv1 = nn.Conv2d(1,no_output_chan,5)    
        self.conv2 = nn.Conv2d(no_output_chan,16,5)
        
        self.fc1 = nn.Linear(16*22*5,120) 
        #have to follow input, x.shape before flattern: 
        self.fc2 = nn.Linear(120,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,35)       
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self,x):
        spec = self.mel_layer(x) #will take batch['waveforms'] in training_step
        spec = torch.log(spec+1e-10) 
        x = spec.unsqueeze(1)       
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, spec
    
    
    
    
    
    
    
class SubSpectralNorm(LightningModule):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)
        return x.view(N, C, F, T)
    
class BroadcastedBlock(LightningModule):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)
        return out
    

class TransitionBlock(LightningModule):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNet(SpeechCommand):
    def __init__(self, no_output_chan, cfg_spec):
        super().__init__()
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
        
        self.mel_layer = MelSpectrogram(**cfg_spec)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):        
        # x [B,16000]
        
        spec = self.mel_layer(x) # [B,F,T]
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
        out = self.conv4(out)
        out = out.squeeze(2).squeeze(2)
        spec = spec.squeeze(1)

#        print('OUTPUT SHAPE:', out.shape)
#        OUTPUT SHAPE: torch.Size([8, 35, 1, 1])  out
#crossentropy expect[B, C], so need to squeeze to be 2 dimension
#ref:https://pytorch.org/docs/1.9.1/generated/torch.nn.CrossEntropyLoss.html
#old spec :4D [B,1,F,T] , the return spec is for plot log_images, so need 3D
        return out, spec


    