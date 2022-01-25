import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
import sys
sys.path.insert(0, './package/Installation/')
from nnAudio.features.mel import MelSpectrogram, STFT
import sys
import matplotlib.pyplot as plt
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class SpeechCommand(LightningModule):
    def training_step(self, batch, batch_idx):
        if self.current_epoch<70:
            self.mel_layer.mel_basis.requires_grad = False
        else:
            self.mel_layer.mel_basis.requires_grad = True
        outputs, spec = self(batch['waveforms']) 
        loss = self.criterion(outputs, batch['labels'].squeeze(1).long())

#return outputs (2D) for calculate loss, return spec (3D) for visual
#for debug 
#torch.save(outputs, 'output.pt')
#torch.save(batch['labels'], 'label.pt')          
#sys.exit()

        acc = sum(outputs.argmax(-1) == batch['labels'].squeeze(1))/outputs.shape[0] #batch wise
        
        self.log('Train/acc', acc, on_step=False, on_epoch=True)
        #if self.current_epoch==0:
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
        #if self.current_epoch==0:
        if batch_idx == 0:
            fig, axes = plt.subplots(1,1)
            
            if self.fastaudio_filter!=None:
                fbank_matrix = self.fastaudio_filter.get_fbanks()
                f_central = self.fastaudio_filter.f_central
                band = self.fastaudio_filter.band
                debug_dict = {'fbank_matrix': fbank_matrix,
                              'f_central': f_central,
                              'band': band}
                
                torch.save(debug_dict, f'debug_dict_e{self.current_epoch}.pt')
                for idx, i in enumerate(fbank_matrix.t().detach().cpu().numpy()):
                    axes.plot(i)
                self.logger.experiment.add_figure(
                    'Validation/fastaudio_MelFilterBanks',
                    fig,
                    global_step=self.current_epoch)
                
            elif self.fastaudio_filter==None:
            
                mel_filter_banks = torch.clamp(self.mel_layer.mel_basis, 0, 1)
                for i in mel_filter_banks:
                    axes.plot(i.cpu())

                self.logger.experiment.add_figure(
                    'Validation/MelFilterBanks',
                    fig,
                    global_step=self.current_epoch)
                
#     these is for plot mel filter band in nnAudio 
#     fbank_matrix contain all filterbank value
            
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
        optimizer = optim.SGD(self.parameters(),lr=1e-4, momentum=0.9,weight_decay =0.001)
              
#        def step_function(step):
#            if step< 848*5:
#                return step*100/(848*5)
#            else:
#                return 100
                
#        scheduler1 = {
#            'scheduler': LambdaLR(optimizer, lr_lambda= step_function),
#            'interval': 'step',
#            'frequency': 1,
#       }
        if self.optimizer_cfg.warmup=='cosine':
            scheduler = {
                'scheduler': CosineAnnealingWarmupRestarts(optimizer,
                                                           first_cycle_steps=5000,
                                                           cycle_mult=1.0,
                                                           max_lr=0.1,
                                                           min_lr=0.001,
                                                           warmup_steps=848*5,
                                                           gamma=0.5) ,
                'interval': 'step',
                'frequency': 1,}
            return [optimizer] , [scheduler]            
        elif self.optimizer_cfg.warmup=='constant':
            return [optimizer]
        else:
            raise ValueError(f"Please choose the correct warmup type."
                             f"{self.optimizer.warmup} is not supported")
#for learning rate schedular 
#warmup_steps set to 5 epochs, gamma value refer to decay %
#if interval = step, refer to each feedforward step


#return 2 lists
#if use constant learning rate: no cosineannealing --> exclude out the scheduler2 return


    