import copy
import torch
import numpy as np
from scipy import signal
from librosa.filters import mel
from scipy.signal import get_window
import torch
import torch.nn as nn
import torch.nn.functional as F


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)



class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
    
    

def filter_bank_mean(num_rep, codes_mask, max_len_long):
    '''
    num_rep (B, L)
    codes_mask (B, L)
    
    output: filterbank (B, L, max_len_fake)
    
    zero pad in codes must be real zero
    '''
    
    num_rep = num_rep.unsqueeze(-1) # (B, L, 1)
    codes_mask = codes_mask.unsqueeze(-1) # (B, L, 1)
    num_rep = num_rep * codes_mask
    
    right_edge = num_rep.cumsum(dim=1)
    left_edge = torch.zeros_like(right_edge)
    left_edge[:, 1:, :] = right_edge[:, :-1, :]
    right_edge = right_edge.ceil()
    left_edge = left_edge.floor()
    
    index = torch.arange(1, max_len_long+1, device=num_rep.device).view(1, 1, -1)
    
    lower = index - left_edge

    right_edge_flip = max_len_long - right_edge
    
    upper = (index - right_edge_flip).flip(dims=(2,))
    
    # triangular pooling
    fb = F.relu(torch.min(lower, upper)).float()
    
    # mean pooling
    fb = (fb > 0).float()
    
    norm = fb.sum(dim=-1, keepdim=True)
    norm[norm==0] = 1.0
    
    fb = fb / norm
    
    return fb * codes_mask    