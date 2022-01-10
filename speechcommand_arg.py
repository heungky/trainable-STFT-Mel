import argparse 
##speechcommands cnn

import torch
import torchaudio 
import torch.nn as nn
import torch.nn.functional as F
import pandas
import tqdm
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from IPython.display import Audio
from dataloading_util import data_processing
import models as Model 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime


speechcommand_arg = argparse.ArgumentParser(description='User choose GPU or CPU')

speechcommand_arg.add_argument('--device','-d', type=int, help='choose either GPU [1] or CPU[0]')
speechcommand_arg.add_argument('--output_chan','-c', type=int, help='Number of output channel')
speechcommand_arg.add_argument('--model_type','-t', type=str, help='choose model type')
speechcommand_arg.add_argument('--epoch', '-e', type=int, help='number of epoch')


device = speechcommand_arg.parse_args().device
no_output_chan = speechcommand_arg.parse_args().output_chan
model_type = speechcommand_arg.parse_args().model_type
max_epochs = speechcommand_arg.parse_args().epoch



batch_size = 8

trainset = torchaudio.datasets.SPEECHCOMMANDS('./',url='speech_commands_v0.02',folder_in_archive='SpeechCommands',download = False, subset = 'training')
testset = torchaudio.datasets.SPEECHCOMMANDS('./',url='speech_commands_v0.02',folder_in_archive='SpeechCommands',download = False, subset = 'testing')
validset = torchaudio.datasets.SPEECHCOMMANDS('./',url='speech_commands_v0.02',folder_in_archive='SpeechCommands',download = False, subset = 'validation')



trainloader = torch.utils.data.DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=lambda x: data_processing(x))

#speech_command_transform = Speech_Command_label_Transform(trainset)
testloader = torch.utils.data.DataLoader(testset,
                              batch_size=batch_size,
                              collate_fn=lambda x: data_processing(x))

validloader = torch.utils.data.DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=lambda x: data_processing(x))
#for dataloader, trainset need shuffle

#***********
net = getattr(Model, model_type)(no_output_chan)
# net = net.to(device)
print(type(net))



#added into Model_types.py, using pytorch_lightning
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#for epoch in range(2):
    #Loss = 0
    
    #for batch in tqdm.tqdm(trainloader):
        #mel_output_batch = mel_layer(batch['waveforms'].to(device)) 
        #mel_output_batch4 = mel_output_batch.unsqueeze(1) 
        
        
        #optimizer.zero_grad()
        #outputs = net(mel_output_batch4)
        #loss = criterion(outputs, batch['labels'].to(device).squeeze(1).long()) 

        #loss.backward()
        #optimizer.step()
        
now = datetime.now()        
logger = TensorBoardLogger(save_dir=".", version=1, name=f'output/{now.strftime("%d-%m-%Y-%H-%M-%S")}')


trainer = Trainer(gpus=device, max_epochs=max_epochs,logger=logger, check_val_every_n_epoch=1, num_sanity_val_steps=5)

trainer.fit(net, trainloader, validloader)
#added validloader, in order to reach validation_step









