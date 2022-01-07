import argparse 

speechcommand_arg = argparse.ArgumentParser(description='User choose GPU or CPU')

speechcommand_arg.add_argument('--device', type=str, help='choose either GPU or CPU')
speechcommand_arg.add_argument('--output_chan', type=int, help='Number of output channel')


device = speechcommand_arg.parse_args().device
no_output_chan = speechcommand_arg.parse_args().output_chan

##speechcommands cnn

import torch
import torchaudio 
import torch.nn as nn
import torch.nn.functional as F
import pandas
import tqdm
from torch.nn.utils.rnn import pad_sequence
from nnAudio.features.mel import MelSpectrogram
import matplotlib.pyplot as plt
from IPython.display import Audio

str2int = \
{'backward': 0,
 'bed': 1,
 'bird': 2,
 'cat': 3,
 'dog': 4,
 'down': 5,
 'eight': 6,
 'five': 7,
 'follow': 8,
 'forward': 9,
 'four': 10,
 'go': 11,
 'happy': 12,
 'house': 13,
 'learn': 14,
 'left': 15,
 'marvin': 16,
 'nine': 17,
 'no': 18,
 'off': 19,
 'on': 20,
 'one': 21,
 'right': 22,
 'seven': 23,
 'sheila': 24,
 'six': 25,
 'stop': 26,
 'three': 27,
 'tree': 28,
 'two': 29,
 'up': 30,
 'visual': 31,
 'wow': 32,
 'yes': 33,
 'zero': 34
}

#create dict from dataset_label to int


def data_processing(data):
    waveforms = []
    labels = []
    
    for batch in data:
        waveforms.append(batch[0].squeeze(0)) #after squeeze => (audio_len) tensor # remove batch dim
        # batch[2] = string
        # str2int = dict
        # str2int[batch[2]] = int
        #torch.Tensor([str2int[batch[2]]]) = tensor
        label = torch.Tensor([str2int[batch[2]]]) # batch[2] is the label key #str --> int --> tensor
        ## print(f"{label=}")
        labels.append(label)
        
    
        
    waveform_padded = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)  
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    
    output_batch = {'waveforms': waveform_padded, 
             'labels': labels,
             }
    return output_batch

mel_layer = MelSpectrogram(sr=16000, 
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

mel_layer.to(device)

batch_size = 8
trainset = torchaudio.datasets.SPEECHCOMMANDS('./',url='speech_commands_v0.02',folder_in_archive='SpeechCommands',download = False, subset = 'training')
testset = torchaudio.datasets.SPEECHCOMMANDS('./',url='speech_commands_v0.02',folder_in_archive='SpeechCommands',download = False, subset = 'testing')

trainloader = torch.utils.data.DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=lambda x: data_processing(x))

#speech_command_transform = Speech_Command_label_Transform(trainset)
testloader = torch.utils.data.DataLoader(testset,
                              batch_size=batch_size,
                              collate_fn=lambda x: data_processing(x))


import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,no_output_chan,5)    
        self.conv2 = nn.Conv2d(no_output_chan,16,5)
        self.fc1 = nn.Linear(16*22*5,120) 
        #have to follow input, x.shape before flattern: 
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,35)
        
    def forward(self,x):
        #print(f"{x.shape=}")
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net = net.to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    Loss = 0
    
    for batch in tqdm.tqdm(trainloader):
        mel_output_batch = mel_layer(batch['waveforms'].to(device)) 
        mel_output_batch4 = mel_output_batch.unsqueeze(1) 
        
        optimizer.zero_grad()
        outputs = net(mel_output_batch4)
        loss = criterion(outputs, batch['labels'].to(device).squeeze(1).long()) 

        loss.backward()
        optimizer.step()
        














