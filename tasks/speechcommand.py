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
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sklearn.metrics import precision_recall_fscore_support
from dataset.speechcommands import idx2name, name2idx
from sklearn.metrics import confusion_matrix
import numpy as np
import re
import itertools


class SpeechCommand(LightningModule):
    def training_step(self, batch, batch_idx):
#        if self.current_epoch<70:
#           self.mel_layer.mel_basis.requires_grad = False
#        else:
#        self.mel_layer.mel_basis.requires_grad = True
        outputs, spec = self(batch['waveforms']) 
        loss = self.criterion(outputs, batch['labels'].long())

#return outputs (2D) for calculate loss, return spec (3D) for visual
#for debug 
#torch.save(outputs, 'output.pt')
#torch.save(batch['labels'], 'label.pt')          
#sys.exit()

        acc = sum(outputs.argmax(-1) == batch['labels'])/outputs.shape[0] #batch wise
        
        self.log('Train/acc', acc, on_step=False, on_epoch=True)
        #if self.current_epoch==0:
        if batch_idx == 0:
            self.log_images(spec, 'Train/Spec')
            cm = plot_confusion_matrix(batch['labels'].cpu(),
                                       outputs.argmax(-1).cpu(),
                                       name2idx.keys(),
                                       title='Train: Confusion matrix',
                                       normalize=False)
            self.logger.experiment.add_figure('Train/confusion_maxtrix', cm, global_step=self.current_epoch)            
        self.log('Train/Loss', loss, on_step=False, on_epoch=True)
        
        return loss
#log(graph title, take acc as data, on_step: plot every step, on_epch: plot every epoch)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)
        with torch.no_grad():
            torch.clamp_(self.mel_layer.mel_basis, 0, 1)                    
    
    def validation_step(self, batch, batch_idx):               
        outputs, spec = self(batch['waveforms'])
        loss = self.criterion(outputs, batch['labels'].long())        
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
                
#                 torch.save(debug_dict, f'debug_dict_e{self.current_epoch}.pt')
                for idx, i in enumerate(fbank_matrix.t().detach().cpu().numpy()):
                    axes.plot(i)
                self.logger.experiment.add_figure(
                    'Validation/fastaudio_MelFilterBanks',
                    fig,
                    global_step=self.current_epoch)
                
            elif self.fastaudio_filter==None:
            
                mel_filter_banks = self.mel_layer.mel_basis
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
                       'labels': batch['labels']}        
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
        
        cm = plot_confusion_matrix(label.cpu(),
                                   pred.argmax(-1).cpu(),
                                   name2idx.keys(),
                                   title='Validation: Confusion matrix',
                                   normalize=False)
        self.logger.experiment.add_figure('Validation/confusion_maxtrix', cm, global_step=self.current_epoch)
        
        self.log('Validation/acc', acc, on_step=False, on_epoch=True)    
#use the return value from validation_step: output_dict , to calculate the overall accuracy   #epoch wise 
                              
    def test_step(self, batch, batch_idx):               
        outputs, spec = self(batch['waveforms'])
        loss = self.criterion(outputs, batch['labels'].long())        
#acc = sum(outputs.argmax(-1) == batch['labels'].squeeze(1))/outputs.shape[0]
#accuracy for 
#self.log('Validation/acc', acc, on_step=False, on_epoch=True)

        self.log('Test/Loss', loss, on_step=False, on_epoch=True)          
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
                
#                 torch.save(debug_dict, f'debug_dict_e{self.current_epoch}.pt')
                for idx, i in enumerate(fbank_matrix.t().detach().cpu().numpy()):
                    axes.plot(i)
                self.logger.experiment.add_figure(
                    'Test/fastaudio_MelFilterBanks',
                    fig,
                    global_step=self.current_epoch)
                
            elif self.fastaudio_filter==None:
            
                mel_filter_banks = self.mel_layer.mel_basis
                for i in mel_filter_banks:
                    axes.plot(i.cpu())

                self.logger.experiment.add_figure(
                    'Test/MelFilterBanks',
                    fig,
                    global_step=self.current_epoch)
                
#     these is for plot mel filter band in nnAudio 
#     fbank_matrix contain all filterbank value
            
            self.log_images(spec, 'Test/Spec')
#plot log_images for 1st epoch_1st batch
        
        output_dict = {'outputs': outputs,
                       'labels': batch['labels']}        
        return output_dict

    
    def test_epoch_end(self, outputs):
        pred = []
        label = []
        for output in outputs:
            pred.append(output['outputs'])
            label.append(output['labels'])
        label = torch.cat(label, 0)
        pred = torch.cat(pred, 0)
        
        result_dict = {}
        for key in [None, 'micro', 'macro', 'weighted']:
            result_dict[key] = {}
            p, r, f1, _ = precision_recall_fscore_support(label.cpu(), pred.argmax(-1).cpu(), average=key, zero_division=0)
            result_dict[key]['precision'] = p
            result_dict[key]['recall'] = r
            result_dict[key]['f1'] = f1
            
        barplot(result_dict, 'precision')
        barplot(result_dict, 'recall')
        barplot(result_dict, 'f1')
            
        acc = sum(pred.argmax(-1) == label)/label.shape[0]
        self.log('Test/acc', acc, on_step=False, on_epoch=True)
        
        self.log('Test/micro_f1', result_dict['micro']['f1'], on_step=False, on_epoch=True)
        self.log('Test/macro_f1', result_dict['macro']['f1'], on_step=False, on_epoch=True)
        self.log('Test/weighted_f1', result_dict['weighted']['f1'], on_step=False, on_epoch=True)
        
        cm = plot_confusion_matrix(label.cpu(),
                                   pred.argmax(-1).cpu(),
                                   name2idx.keys(),
                                   title='Test: Confusion matrix',
                                   normalize=False)
        self.logger.experiment.add_figure('Test/confusion_maxtrix', cm, global_step=self.current_epoch)        
        
        torch.save(result_dict, "result_dict.pt")        
        
        return result_dict
        
    def log_images(self, tensors, key):
        fig, axes = plt.subplots(2,2, figsize=(12,5), dpi=100)
        for ax, tensor in zip(axes.flatten(), tensors):
            ax.imshow(tensor.cpu().detach(), aspect='auto', origin='lower', cmap='jet')
        plt.tight_layout()
        self.logger.experiment.add_figure(f"{key}", fig, global_step=self.current_epoch)
        plt.close(fig)
#plot images in TensorBoard        
    
    
    def configure_optimizers(self):
        model_param = []
        for name, params in self.named_parameters():
            if 'mel_layer.' in name:
                pass
            else:
                model_param.append(params)          
        optimizer = optim.SGD([
                                {"params": self.mel_layer.parameters(),
                                 "lr": 1e-3,
                                 "momentum": 0.9,
                                 "weight_decay": 0.001},
                                {"params": model_param,
                                 "lr": 1e-3,
                                 "momentum": 0.9,
                                 "weight_decay": 0.001}            
                              ])
#for applying diff lr in model and mel filter bank        
            
              
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
                             f"{self.optimizer_cfg.warmup} is not supported")
#for learning rate schedular 
#warmup_steps set to 5 epochs, gamma value refer to decay %
#if interval = step, refer to each feedforward step


#return 2 lists
#if use constant learning rate: no cosineannealing --> exclude out the scheduler2 return


def barplot(result_dict, title, figsize=(4,12), minor_interval=0.2, log=False):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    metric = {}
    for idx, item in enumerate(result_dict[None][title]):
        metric[idx2name[idx]] = item
    xlabels = list(metric.keys())
    values = list(metric.values())
    if log:
        values = np.log(values)
    ax.barh(xlabels, values)
    ax.tick_params(labeltop=True, labelright=False)
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_minor_locator(MultipleLocator(minor_interval))
    ax.set_ylim([-1,len(xlabels)])
    ax.set_title(title)
    ax.grid(axis='x')
    ax.grid(b=True, which='minor', linestyle='--')
    fig.savefig(f'{title}.png', bbox_inches='tight')
    fig.tight_layout() # prevent edge from missing
#         fig.set_tight_layout(True)
    return fig
          
    
def plot_confusion_matrix(correct_labels,
                          predict_labels,
                          labels,
                          title='Confusion matrix',
                          normalize=False):
    ''' 
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary 

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=range(len(labels)))
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=160, facecolor='w', edgecolor='k')
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
#     classes = ['\n'.join(l) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=6, rotation=0,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=6, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
#     summary = tfplot.figure.to_summary(fig, tag=tensor_name)
#     return summary

    return fig