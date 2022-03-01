# Useful github libries

from AudioLoader.Speech import TIMIT

# Libraries related to PyTorch
import torch
from torch.utils.data import DataLoader, random_split

# Libraries related to PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# custom packages
from tasks.asr import ASR
import models as Model
from utils.text_processing import TextTransform, data_processing

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

# For loading the output class ddictionary
import pickle
from hydra.utils import to_absolute_path

@hydra.main(config_path="conf", config_name="ASR_config")
def main(cfg):  
    # Loading dataset
    
    cfg.data_root = to_absolute_path(cfg.data_root)
    train_dataset = TIMIT(**cfg.dataset.train)
    test_dataset = TIMIT(**cfg.dataset.test)
    train_dataset, valid_dataset = random_split(train_dataset, [4000, 620], generator=torch.Generator().manual_seed(0))

    
    # Auto inferring output mode and output dimension
    if cfg.output_mode == 'char':
        dict_file = 'characters_dict'
        cfg.data_processing.label_key = 'words'
    elif cfg.output_mode == 'ph':
        dict_file = 'phonemics_dict'
        cfg.data_processing.label_key = 'phonemics'        
    elif cfg.output_mode == 'word':
        dict_file = 'words_dict'
        cfg.data_processing.label_key = 'words'        
    else:
        raise ValueError(f'cfg.output_mode={cfg.output_mode} is not supported')
        
    with open(to_absolute_path(dict_file), 'rb') as f:
        output_dict = pickle.load(f)
    cfg.model.args.output_dim = len(output_dict) # number of classes equals to number of entries in the dict
    
    
    if '_Fastaudio' in cfg.model.model_type:
            cfg.model.args.input_dim = cfg.model.fastaudio.n_mels 
            train_setting=cfg.model.fastaudio.freeze
            n_mel=cfg.model.fastaudio.n_mels
            stft = cfg.model.spec_args.trainable
    elif '_nnAudio' in cfg.model.model_type:
            cfg.model.args.input_dim = cfg.model.spec_args.n_mels
            train_setting=cfg.model.spec_args.trainable_mel
            n_mel=cfg.model.spec_args.n_mels
            stft = cfg.model.spec_args.trainable_STFT
    

    text_transform = TextTransform(output_dict, cfg.output_mode) # for text to int conversion layer

    train_loader = DataLoader(train_dataset,
                              **cfg.dataloader.train,
                              collate_fn=lambda x: data_processing(x,
                                                                   text_transform,
                                                                   **cfg.data_processing))
    valid_loader = DataLoader(valid_dataset,
                              **cfg.dataloader.valid,
                              collate_fn=lambda x: data_processing(x,
                                                                   text_transform,
                                                                   **cfg.data_processing))
    test_loader = DataLoader(test_dataset,
                             **cfg.dataloader.test,
                             collate_fn=lambda x: data_processing(x,
                                                                  text_transform,
                                                                  **cfg.data_processing))      

     #for dataloader, trainset need shuffle
    if cfg.model.model_type=='X_vector':
        model = ASR(getattr(Model, cfg.model.model_type)(**cfg.model.args, cfg_model=cfg.model),text_transform,**cfg.pl)        
    else:
        print(f'cfg.model ={cfg.model.keys()} ')
        model=getattr(Model, cfg.model.model_type)(cfg.model,text_transform,cfg.pl.lr)

#     model = ASR(getattr(Model, cfg.model.type)(spec_layer, **cfg.model.args),
#                 text_transform,
#                 **cfg.pl)
    
    
    checkpoint_callback = ModelCheckpoint(monitor="valid_ctc_loss",
                                          filename="{epoch:02d}-{valid_ctc_loss:.2f}-{PER:.2f}",
                                          save_top_k=3,
                                          mode="min")
    name = f'n_mels={n_mel}-{cfg.model.model_type}-mel={train_setting}-STFT={stft}--TIMIT'
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         check_val_every_n_epoch=1)


    trainer.fit(model, train_loader, valid_loader)
#     trainer.test(model, test_loader)
    trainer.test(model, test_loader, ckpt_path="best")    
    
if __name__ == "__main__":
    main()    