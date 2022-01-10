import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from nnAudio.features.mel import MelSpectrogram
import sys

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
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    def training_step(self, batch, batch_idx):
        mel_output_batch = self.mel_layer(batch['waveforms']) 
        mel_output_batch4 = mel_output_batch.unsqueeze(1) 
        
        outputs = self(mel_output_batch4)
        loss = self.criterion(outputs, batch['labels'].squeeze(1).long())
#         torch.save(outputs, 'output.pt')
#         torch.save(batch['labels'], 'label.pt')        
#for debug   
#         sys.exit()


        acc = sum(outputs.argmax(-1) == batch['labels'].squeeze(1))/outputs.shape[0]
        
        self.log('Train/acc', acc, on_step=False, on_epoch=True)
        self.log('Train/Loss', loss, on_step=False, on_epoch=True)
        return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        mel_output_batch = self.mel_layer(batch['waveforms']) 
        mel_output_batch4 = mel_output_batch.unsqueeze(1) 
        
        outputs = self(mel_output_batch4)
        loss = self.criterion(outputs, batch['labels'].squeeze(1).long())
        
        acc = sum(outputs.argmax(-1) == batch['labels'].squeeze(1))/outputs.shape[0]
        self.log('Validation/acc', acc, on_step=False, on_epoch=True)
        self.log('Validation/Loss', loss, on_step=False, on_epoch=True)    
        
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