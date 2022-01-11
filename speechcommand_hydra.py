import hydra
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f"{(cfg.model_type)}")




    batch_size = cfg.batch_size

    trainset = torchaudio.datasets.SPEECHCOMMANDS(**cfg.dataset.train)
    validset = torchaudio.datasets.SPEECHCOMMANDS(**cfg.dataset.val)
    testset = torchaudio.datasets.SPEECHCOMMANDS(**cfg.dataset.test)




    trainloader = torch.utils.data.DataLoader(trainset,                                
                                  collate_fn=lambda x: data_processing(x),
                                             **cfg.dataloader.train)

    #speech_command_transform = Speech_Command_label_Transform(trainset)
    validloader = torch.utils.data.DataLoader(validset,                               
                                  collate_fn=lambda x: data_processing(x),
                                             **cfg.dataloader.val)
    
    testloader = torch.utils.data.DataLoader(testset,   
                                  collate_fn=lambda x: data_processing(x),
                                            **cfg.dataloader.test)


    #for dataloader, trainset need shuffle

    #***********
    net = getattr(Model, cfg.model_type)(cfg.no_output_chan)
    # net = net.to(gpus)
    print(type(net))



    #added into Model_types.py, using pytorch_lightning
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #for epoch in range(2):
        #Loss = 0

        #for batch in tqdm.tqdm(trainloader):
            #mel_output_batch = mel_layer(batch['waveforms'].to(gpus)) 
            #mel_output_batch4 = mel_output_batch.unsqueeze(1) 


            #optimizer.zero_grad()
            #outputs = net(mel_output_batch4)
            #loss = criterion(outputs, batch['labels'].to(gpus).squeeze(1).long()) 

            #loss.backward()
            #optimizer.step()

    now = datetime.now()        
    logger = TensorBoardLogger(save_dir=".", version=1, name=f'{cfg.model_type}-bz={cfg.batch_size}')


    trainer = Trainer(**cfg.trainer, logger = logger)

    trainer.fit(net, trainloader, validloader)
    #added validloader, in order to reach validation_step


if __name__ == "__main__":
    my_app()






