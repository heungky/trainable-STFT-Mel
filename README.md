# Welcome

We explore the possibility of maximizing the information represented in spectrograms by making the spectrogram basis functions trainable.

A number of experiments are conducted in which we compare the performance of trainable _short-time Fourier transform (STFT)_ and _Mel basis functions_ provided by FastAudio and nnAudio on two tasks: _keyword spotting (KWS)_ and _automatic speech recognition (ASR)_. 

_Broadcasting-residual network (BC-ResNet)_ as well as a _Simple model (constructed with a linear layer)_ are used for these two tasks.

In our experiments, we explore four different training settings: \
A Both gMel and gSTFT are non-trainable. \
B gMel is trainable while gSTFT is fixed.\
C gMel is fixed while gSTFT is trainable.\
D Both gMel and gSTFT are trainable.

# Introduction

```
trainable-STFT-Mel
├── conf
│     ├─model
│     │     ├─BC_ResNet.yaml
│     │     ├─BC_ResNet_ASR.yaml
│     │     ├─BC_ResNet_maskout.yaml
│     │     │   
│     │     ├─Linearmodel.yaml
│     │     ├─Linearmodel_ASR.yaml
│     │     ├─Linearmodel_maskout.yaml
│     │     │   
│     │
│     ├─ASR_config.yaml
│     └─KWS_config.yaml
│
├── models
│     ├─nnAudio_model.py
│     └─fastaudio_model.py
├── tasks
│     ├─speechcommand.py
│     ├─speechcommand_maskout.py
│     ├─Timit.py
│     ├─Timit_maskout.py
│     │
├──train_KWS_hydra.py
├──train_ASR_hydra.py
├──phonemics_dict
├──requirements.txt
```

* `conf` contains the `.yaml` configuration files.
* `models` contains the model architectures.
* `tasks` contains the lightning modules for KWS and ASR.
* `train_KWS_hydra.py` and `train_ASR_hydra.py` are training script of KWS and ASR respectively.
* `phonemics_dict` is the phoneme labels provided in TIMIT which used for phoneme recognition.

# Requirement

Python `3.8.10` is required to run this repo. 

You can install all required libraries at once via 
```bash
pip install -r requirements.txt
```

# Training the model
```bash
python train_KWS_hydra.py 
```
```bash
python train_ASR_hydra.py 
```
Note: 
* If this is your 1st time to train the model, you need to set `download` setting to `True` via
```bash
python train_KWS_hydra.py download=True
```

* If you use CPU instead of GPU to train the model, set gpus to 0 via 
```bash
python train_KWS_hydra.py gpus=0
```

Default:
* nnAudio BC_ResNet model: `model=BC_ResNet`
* setting A (Both gMel and gSTFT are non-trainable):
`model.spec_args.trainable_mel=False` `model.spec_args.trainable_STFT=False`
* 40 number of Mel bases: `model.spec_args.n_mels=40`
* use 1 gpus

## Multiple training with KWS/ASR task under four different settings

### For model with nnAudio front-end
```bash
python train_KWS_hydra.py -m gpus=<arg> model=<arg> model.spec_args.trainable_mel=True,False model.spec_args.trainable_STFT=True,False
```

### For model with Fastaudio front-end
```bash
python train_KWS_hydra.py -m gpus=<arg> model=<arg> model.fastaudio.freeze=True,False model.spec_args.trainable=True,False
```

Note: simply replace `train_KWS_hydra.py` with `train_ASR_hydra.py` for ASR task.

## Multiple training with KWS/ASR task under different number of Mel bases

### For model with nnAudio front-end
```bash
python train_KWS_hydra.py -m gpus=<arg> model=<arg> model.spec_args.n_mels=10,20,30,40 
```

### For model with FastAudio front-end
```bash
python train_KWS_hydra.py -m gpus=<arg> model=<arg> model.fastaudio.n_mels=10,20,30,40
```

Note: simply replace `train_KWS_hydra.py` with `train_ASR_hydra.py` for ASR task.

## Train model with KWS/ASR task under masked STFT bins 

```bash
python train_KWS_hydra.py gpus=<arg> model=<arg> model.maskout_start=<arg> model.maskout_end=<arg>
```
Applicable model: 
* KWS nnAudio BC_ResNet
* KWS nnAudio Simple
* ASR nnAudio Simple

Note: simply replace `train_KWS_hydra.py` with `train_ASR_hydra.py` for ASR task.

## Train model with KWS/ASR task under randomely initialize mel bases

```bash
python train_KWS_hydra.py gpus=<arg> model=<arg> model.random_mel=True
```

Applicable model: 
* KWS nnAudio BC_ResNet
* ASR nnAudio BC_ResNet
* KWS nnAudio Simple
* ASR nnAudio Simple

Note: simply replace `train_KWS_hydra.py` with `train_ASR_hydra.py` for ASR task.