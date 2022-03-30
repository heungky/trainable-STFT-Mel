# Libraries related to PyTorch
import torch
from torch import Tensor
import torchaudio 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler,DataLoader

# Libraries related to PyTorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# custom packages
from dataset.speechcommands import SPEECHCOMMANDS_12C #for 12 classes KWS task
import models as Model 

from dataloading_util import data_processing
from datetime import datetime


@hydra.main(config_path="conf", config_name="KWS_config")
def cnn(cfg : DictConfig) -> None:

    cfg.data_root = to_absolute_path(cfg.data_root)

    batch_size = cfg.batch_size

    trainset = SPEECHCOMMANDS_12C(**cfg.dataset.train)
    validset = SPEECHCOMMANDS_12C(**cfg.dataset.val)
    testset = SPEECHCOMMANDS_12C(**cfg.dataset.test)

    # for class weighting, rebalancing silence(10th class) and unknown(11th class) in training set
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

    validloader = DataLoader(validset,                               
                                  collate_fn=lambda x: data_processing(x),
                                             **cfg.dataloader.val)
    
    testloader = DataLoader(testset,   
                                  collate_fn=lambda x: data_processing(x),
                                            **cfg.dataloader.test)     
    
    
    if '_Fastaudio' in cfg.model.model_type:
            cfg.model.args.input_dim = cfg.model.fastaudio.n_mels *101
            train_setting=cfg.model.fastaudio.freeze
            n_mel=cfg.model.fastaudio.n_mels
            stft = cfg.model.spec_args.trainable
    
    elif '_nnAudio' in cfg.model.model_type:
            cfg.model.args.input_dim = cfg.model.spec_args.n_mels *101 
            train_setting=cfg.model.spec_args.trainable_mel
            n_mel=cfg.model.spec_args.n_mels
            stft = cfg.model.spec_args.trainable_STFT
            
    elif '_maskout' in cfg.model.model_type:
            cfg.model.args.input_dim = cfg.model.spec_args.n_mels *101 
            train_setting=cfg.model.spec_args.trainable_mel
            n_mel=cfg.model.spec_args.n_mels
            stft = cfg.model.spec_args.trainable_STFT            
            
    
    net = getattr(Model, cfg.model.model_type)(cfg.model)
               
    print(type(net))

      
    name = f'SGD-n_mels={n_mel}-{cfg.model.model_type}-mel={train_setting}-STFT={stft}-speechcommand'
    #file name shown in tensorboard logger
    
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    
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






