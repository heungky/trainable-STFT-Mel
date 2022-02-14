import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
##speechcommands cnn

import torch
import torchaudio 
from dataset.speechcommands import SPEECHCOMMANDS_12C
import torch.nn as nn
import torch.nn.functional as F
import pandas
import tqdm
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from dataloading_util import data_processing
import models as Model 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
from torch import Tensor
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import WeightedRandomSampler,DataLoader
from nnAudio import Spectrogram

@hydra.main(config_path="conf", config_name="speechcommand_config")
def cnn(cfg : DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))

    cfg.data_root = to_absolute_path(cfg.data_root)

    batch_size = cfg.batch_size

    trainset = SPEECHCOMMANDS_12C(**cfg.dataset.train)
    validset = SPEECHCOMMANDS_12C(**cfg.dataset.val)
    testset = SPEECHCOMMANDS_12C(**cfg.dataset.test)

    # for class weighting, rebalancing silence and unknown class in training set
    class_weights = [1,1,1,1,1,1,1,1,1,1,4.6,1/17]
    sample_weights = [0] * len(trainset)
    #create a list as per length of trainset

    for idx, (data,rate,label,speaker_id, _) in enumerate(trainset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    #apply sample_weights in each data base on their label class in class_weight
    #ref: https://www.youtube.com/watch?v=4JFVhJyTZ44&t=518s
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),replacement=True)
     

        
        
    trainloader = DataLoader(trainset,                                
                                  collate_fn=lambda x: data_processing(x),
                                             **cfg.dataloader.train,sampler=sampler)

    #speech_command_transform = Speech_Command_label_Transform(trainset)
    validloader = DataLoader(validset,                               
                                  collate_fn=lambda x: data_processing(x),
                                             **cfg.dataloader.val)
    
    testloader = DataLoader(testset,   
                                  collate_fn=lambda x: data_processing(x),
                                            **cfg.dataloader.test)     
    
    
    if '_Fastaudio' in cfg.model.model_type:
            cfg.model.args.input_dim = cfg.model.fastaudio.n_mels *101
    elif '_nnAudio' in cfg.model.model_type:
            cfg.model.args.input_dim = cfg.model.spec_args.n_mels *101 
    
            
    #for dataloader, trainset need shuffle
    if cfg.model.model_type=='X_vector':
        net = getattr(Model, cfg.model.model_type)(**cfg.model.args, cfg_model=cfg.model)        
    else:
        print(f'cfg.model ={cfg.model.keys()}')
        net = getattr(Model, cfg.model.model_type)(cfg.model)
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

    #now = datetime.now()        
    logger = TensorBoardLogger(save_dir=".", version=1, name=f'SGD-{cfg.model.model_type}-speechcommand-bz={cfg.batch_size}')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint,
                                          auto_insert_metric_name=False) #save checkpoint
    
    callbacks = [checkpoint_callback, lr_monitor]

    trainer = Trainer(**cfg.trainer, logger = logger, callbacks=callbacks)

    trainer.fit(net, trainloader, validloader)
    trainer.test(net, testloader)
    #added validloader, in order to reach validation_step


if __name__ == "__main__":
    cnn()






