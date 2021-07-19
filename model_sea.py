import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import ConvNorm, LinearNorm
from torch.nn.parameter import Parameter



class GroupNorm_Mask(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x, mask):
        B, C, L = x.size()
        assert C % self.num_groups == 0
        
        x = x.view(B, self.num_groups, C//self.num_groups, L)
        mask = mask.view(B, 1, 1, L)
        x = x * mask
        
        mean = x.mean(dim=2, keepdim=True).sum(dim=3, keepdim=True) / mask.sum(dim=3, keepdim=True)
        var = (((x - mean)**2)*mask).mean(dim=2, keepdim=True).sum(dim=3, keepdim=True) / mask.sum(dim=3, keepdim=True)
        
        x = (x - mean) / (var + self.eps).sqrt()
        
        x = x.view(B, C, L)
        
        return x * self.weight.view(1,-1,1) + self.bias.view(1,-1,1)
    
    
    
class M43_Sequential(nn.Sequential):
    def forward(self, inputs, mask):
        inputs = self._modules['0'](inputs)
        inputs = self._modules['1'](inputs, mask)
        return inputs
    


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        
        self.dim_freq = hparams.dim_freq_sea
        self.dim_enc = hparams.dim_enc_sea
        self.chs_grp = hparams.chs_grp
        self.dim_neck = hparams.dim_neck_sea
        
        convolutions = []        
        for i in range(5):
            conv_layer = M43_Sequential(
                ConvNorm(self.dim_freq if i==0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=1, stride=1,
                         padding=0,
                         dilation=1, w_init_gain='relu'),
                GroupNorm_Mask(self.dim_enc//self.chs_grp, self.dim_enc))
            convolutions.append(conv_layer)
             
        conv_layer = M43_Sequential(
                ConvNorm(self.dim_enc,
                         128,
                         kernel_size=1, stride=1,
                         padding=0,
                         dilation=1, w_init_gain='relu'),
                GroupNorm_Mask(128//self.chs_grp, 128))
        convolutions.append(conv_layer)   
        
        conv_layer = M43_Sequential(
                ConvNorm(128,
                         32,
                         kernel_size=1, stride=1,
                         padding=0,
                         dilation=1, w_init_gain='relu'),
                GroupNorm_Mask(32//self.chs_grp, 32))
        convolutions.append(conv_layer)           
        
        conv_layer = M43_Sequential(
                ConvNorm(32,
                         self.dim_neck,
                         kernel_size=1, stride=1,
                         padding=0,
                         dilation=1, w_init_gain='relu'),
                GroupNorm_Mask(1, self.dim_neck))
        convolutions.append(conv_layer)   
            
        self.convolutions = nn.ModuleList(convolutions)
        

    def forward(self, x, mask):
                
        for conv in self.convolutions:
            x = F.relu(conv(x, mask))
            
        codes = x.permute(0, 2, 1) * mask.unsqueeze(-1)

        return codes
      
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.dim_enc = hparams.dim_enc_sea
        self.dim_emb = hparams.dim_spk
        self.dim_freq = hparams.dim_freq_sp
        self.dim_neck = hparams.dim_neck_sea
        
        self.lstm = nn.LSTM(self.dim_neck+self.dim_emb, 
                            1024, 3, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, self.dim_freq)

    def forward(self, x):
        
        outputs = self.lstm(x)[0]
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    

    
    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, hparams):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, x, c_trg):
        
        x = x.transpose(2,1)
        codes = self.encoder(x)
        
        encoder_outputs = torch.cat((codes, 
                                     c_trg.unsqueeze(1).expand(-1,x.size(-1),-1)), dim=-1)
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs
    
    def encode(self, x, mask):
        x = x.transpose(2,1)
        codes = self.encoder(x, mask)
        return codes
    
    def decode(self, codes, c_trg):
        encoder_outputs = torch.cat((codes, 
                                     c_trg.unsqueeze(1).expand(-1,codes.size(1),-1)), dim=-1)
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs
    
    
    
class Encoder_2(nn.Module):
    """Encoder module:
    """
    def __init__(self, hparams):
        super().__init__()
        
        self.dim_freq = hparams.dim_freq_sea
        self.dim_enc = hparams.dim_enc_sea
        self.chs_grp = hparams.chs_grp
        self.dim_neck = hparams.dim_neck_sea
        
        convolutions = []        
        for i in range(5):
            conv_layer = M43_Sequential(
                ConvNorm(self.dim_freq if i==0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                GroupNorm_Mask(self.dim_enc//self.chs_grp, self.dim_enc))
            convolutions.append(conv_layer)
             
        conv_layer = M43_Sequential(
                ConvNorm(self.dim_enc,
                         128,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                GroupNorm_Mask(128//self.chs_grp, 128))
        convolutions.append(conv_layer)   
        
        conv_layer = M43_Sequential(
                ConvNorm(128,
                         32,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                GroupNorm_Mask(32//self.chs_grp, 32))
        convolutions.append(conv_layer)           
        
        conv_layer = M43_Sequential(
                ConvNorm(32,
                         self.dim_neck,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                GroupNorm_Mask(1, self.dim_neck))
        convolutions.append(conv_layer)   
            
        self.convolutions = nn.ModuleList(convolutions)
        

    def forward(self, x, mask):
                
        for i in range(len(self.convolutions)-1):
            x = F.relu(self.convolutions[i](x, mask))
            
        x = self.convolutions[-1](x, mask)    
            
        codes = x.permute(0, 2, 1) * mask.unsqueeze(-1)

        return codes    