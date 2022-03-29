import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import TriStageLRSchedule
from utils.text_processing import GreedyDecoder
import fastwer
import contextlib
import pandas as pd
import sys


class Timit(pl.LightningModule):
    def __init__(self,
                 text_transform,
                 lr):
        super().__init__()
        self.text_transform = text_transform        
        self.lr = lr
       

    def training_step(self, batch, batch_idx):
        x = batch['waveforms']
        out, spec = self(x)
        pred = out
        pred = torch.log_softmax(pred, -1) 
        #CTC loss requires log_softmax
                
        loss = F.ctc_loss(pred.transpose(0, 1), #[B, T_i, num_class]
                          batch['labels'],
                          batch['input_lengths'],   #prediction T_i
                          batch['label_lengths'])   #T_l 
                          #if loss nan:check T_i > T_l
        if torch.isnan(loss):       
            torch.save(pred, './ASRpred.pt')
            torch.save(batch['labels'], './ASRbatch_label.pt')
            torch.save(batch['input_lengths'], './input_lengths.pt')
            torch.save(batch['label_lengths'], './label_lengths.pt')
            sys.exit()
        self.log("train_ctc_loss", loss)
        return loss
  
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if self.fastaudio_filter==None:
            optimizer.step(closure=optimizer_closure)
            with torch.no_grad():
                torch.clamp_(self.mel_layer.mel_basis, 0, 1)    
    
    def validation_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            out, spec  = self(x)
            
            pred = out
            pred = torch.log_softmax(pred, -1) 
            #CTC loss requires log_softmax  

            loss = F.ctc_loss(pred.transpose(0, 1),
                              batch['labels'],
                              batch['input_lengths'],  #original length before padding
                              batch['label_lengths'])
            valid_metrics = {"valid_ctc_loss": loss}

            pred = pred.cpu().detach()
            decoded_preds, decoded_targets = GreedyDecoder(pred,
                                                           batch['labels'],
                                                           batch['label_lengths'],
                                                           self.text_transform)
            PER_batch = fastwer.score(decoded_preds, decoded_targets)/100            
            valid_metrics['valid_PER'] = PER_batch
            if batch_idx==0:
                self.log_images(spec, f'Valid/spectrogram')
                self._log_text(decoded_preds, 'Valid/texts_pred', max_sentences=4)
                if self.current_epoch==0: # log ground truth
                    self._log_text(decoded_targets, 'Valid/texts_label', max_sentences=4)

            self.log_dict(valid_metrics)
        if batch_idx == 0:
            fig, axes = plt.subplots(1,1)
            
            if self.fastaudio_filter!=None:
                fbank_matrix = self.fastaudio_filter.get_fbanks()
                f_central = self.fastaudio_filter.f_central
                band = self.fastaudio_filter.band
                debug_dict = {'fbank_matrix': fbank_matrix,
                              'f_central': f_central,
                              'band': band}
                
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
                
        #for plotting mel filter band in nnAudio 
        #fbank_matrix contain all Fastaudio filterbank value (mel bases)
        if batch_idx == 0:
            if self.fastaudio_filter!=None:
                fig, axes = plt.subplots(2,2)
                for ax, kernel_num in zip(axes.flatten(), [2,10,20,50]):
                    ax.plot(self.mel_layer.wsin[kernel_num,0].cpu())   #STFT() called in Fastaudio model
                    ax.set_ylim(-1,1)
                    fig.suptitle('sin')

                self.logger.experiment.add_figure(
                        'Validation/sin',
                        fig,
                        global_step=self.current_epoch)

                fig, axes = plt.subplots(2,2)
                for ax, kernel_num in zip(axes.flatten(), [2,10,20,50]):
                    ax.plot(self.mel_layer.wcos[kernel_num,0].cpu())
                    ax.set_ylim(-1,1)
                    fig.suptitle('cos')

                self.logger.experiment.add_figure(
                        'Validation/cos',
                        fig,
                        global_step=self.current_epoch)
               
            
            elif self.fastaudio_filter==None:    
                fig, axes = plt.subplots(2,2)
                for ax, kernel_num in zip(axes.flatten(), [2,10,20,50]):
                    ax.plot(self.mel_layer.stft.wsin[kernel_num,0].cpu())  #STFT is included in Melspectrogram()
                    ax.set_ylim(-1,1)
                    fig.suptitle('sin')

                self.logger.experiment.add_figure(
                        'Validation/sin',
                        fig,
                        global_step=self.current_epoch)

                fig, axes = plt.subplots(2,2)
                for ax, kernel_num in zip(axes.flatten(), [2,10,20,50]):
                    ax.plot(self.mel_layer.stft.wcos[kernel_num,0].cpu())
                    ax.set_ylim(-1,1)
                    fig.suptitle('cos')

                self.logger.experiment.add_figure(
                        'Validation/cos',
                        fig,
                        global_step=self.current_epoch)
        #plotting wsin and wcos to show STFT trainable
    def test_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            out, spec = self(x)
            pred = out
            pred = torch.log_softmax(pred, -1) #CTC loss requires log_softmax
            loss = F.ctc_loss(pred.transpose(0, 1),
                              batch['labels'],
                              batch['input_lengths'],
                              batch['label_lengths'])
            valid_metrics = {"test_ctc_loss": loss}

            pred = pred.cpu().detach()
            decoded_preds, decoded_targets = GreedyDecoder(pred,
                                                           batch['labels'],
                                                           batch['label_lengths'],
                                                           self.text_transform)
            PER_batch = fastwer.score(decoded_preds, decoded_targets)/100            
            valid_metrics['test_PER'] = PER_batch
            if batch_idx<4:
                self.log_images(spec, f'Test/spectrogram')
                self._log_text(decoded_preds, 'Test/texts_pred', max_sentences=1)
                if batch_idx==0: # log ground truth
                    self._log_text(decoded_targets, 'Test/texts_label', max_sentences=1)

            self.log_dict(valid_metrics)     

            
    def _log_text(self, texts, tag, max_sentences=4):
        text_list=[]
        for idx in range(min(len(texts),max_sentences)): # visualize 4 samples or the batch whichever is smallest
            # Avoid using <> tag, which will have conflicts in html markdown
            text_list.append(texts[idx])
        s = pd.Series(text_list, name="IPA")
        self.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.current_epoch)

    def log_images(self, tensor, key):
        for idx, spec in enumerate(tensor):
            fig, ax = plt.subplots(1,1)
            ax.imshow(spec.cpu().detach().t(), aspect='auto', origin='lower')    
            self.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=self.current_epoch)         


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return [optimizer]