import argparse 
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
from models.ModelA import ModelA
from dataloading_util import data_processing


speechcommand_arg = argparse.ArgumentParser(description='User choose GPU or CPU')

speechcommand_arg.add_argument('--device', type=str, help='choose either GPU or CPU')
speechcommand_arg.add_argument('--output_chan', type=int, help='Number of output channel')


device = speechcommand_arg.parse_args().device
no_output_chan = speechcommand_arg.parse_args().output_chan


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



net = ModelA(no_output_chan)
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
        














