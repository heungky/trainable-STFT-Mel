import torch
import torch.nn as nn


str2int = \
{'backward': 0,
 'bed': 1,
 'bird': 2,
 'cat': 3,
 'dog': 4,
 'down': 5,
 'eight': 6,
 'five': 7,
 'follow': 8,
 'forward': 9,
 'four': 10,
 'go': 11,
 'happy': 12,
 'house': 13,
 'learn': 14,
 'left': 15,
 'marvin': 16,
 'nine': 17,
 'no': 18,
 'off': 19,
 'on': 20,
 'one': 21,
 'right': 22,
 'seven': 23,
 'sheila': 24,
 'six': 25,
 'stop': 26,
 'three': 27,
 'tree': 28,
 'two': 29,
 'up': 30,
 'visual': 31,
 'wow': 32,
 'yes': 33,
 'zero': 34
}

def data_processing(data):
    waveforms = []
    labels = []
    
    for batch in data:
        waveforms.append(batch[0].squeeze(0)) #after squeeze => (audio_len) tensor # remove batch dim
        # batch[2] = string
        # str2int = dict
        # str2int[batch[2]] = int
        #torch.Tensor([str2int[batch[2]]]) = tensor
        label = torch.Tensor([str2int[batch[2]]]) # batch[2] is the label key #str --> int --> tensor
        ## print(f"{label=}")
        labels.append(label)
        
        
    waveform_padded = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)  
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    
    output_batch = {'waveforms': waveform_padded, 
             'labels': labels,
             }
    return output_batch

        