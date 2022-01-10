import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from nnAudio.features.mel import MelSpectrogram
import sys
import matplotlib.pyplot as plt

class ModelA(LightningModule):
    def __init__(self, no_output_chan):
        super().__init__()
        print(f"I am model A")   
        self.mel_layer = MelSpectrogram(sr=16000, 
                                   n_fft=2048,
                                   win_length=None,
                                   n_mels=100, 
                                   hop_length=512,
                                   window='hann',
                                   center=True,
                                   pad_mode='reflect',
                                   power=2.0,
                                   htk=False,
                                   fmin=0.0,
                                   fmax=None,
                                   norm=1,
                                   trainable_mel=False,
                                   trainable_STFT=False,
                                   verbose=True,)        
        
        self.conv1 = nn.Conv2d(1,no_output_chan,5)    
        self.conv2 = nn.Conv2d(no_output_chan,16,5)
             
        self.fc1 = nn.Linear(16*22*5,120) 
        #have to follow input, x.shape before flattern: 
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,35)
        
        self.criterion = nn.CrossEntropyLoss()       
        
    def forward(self,x):
        spec = self.mel_layer(x) #will take batch['waveforms' in training_step
        spec = torch.log(spec+1e-10) #3-dimension
        x = spec.unsqueeze(1)    #4-dimension     
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, spec
       
    
    
    
    
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
#output_dict        
        
        
        
        
        
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
   
    
    
class ModelB(LightningModule):
    def __init__(self, no_output_chan):
        super().__init__()
        print(f"I am model B")
        self.conv1 = nn.Conv2d(1,no_output_chan,5)    
        self.conv2 = nn.Conv2d(no_output_chan,16,5)
        
        self.fc1 = nn.Linear(16*22*5,120) 
        #have to follow input, x.shape before flattern: 
        self.fc2 = nn.Linear(120,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,35)       

        
    def forward(self,x):
        #print(f"{x.shape=}")
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x