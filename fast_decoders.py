import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt_modules.misc import sequence_mask


class DecodeFunc_Sp(object):
    """
    Decoding functions
    """
    def __init__(self, hparams, type_out):
                
        if type_out == 'Sp':
            self.dim_freq = hparams.dim_freq
            self.max_decoder_steps = hparams.dec_steps_sp
        elif type_out == 'Tx':
            self.dim_freq = hparams.dim_code
            self.max_decoder_steps = hparams.dec_steps_tx
        else:
            raise ValueError
        
        self.gate_threshold = hparams.gate_threshold
        self.type_out = type_out
        
    def __call__(self, tgt, memory_bank, memory_lengths, decoder, postnet):
        
        dec_outs, attns = decoder(tgt, memory_bank, step=None, 
                                  memory_lengths=memory_lengths)
        spect_gate = postnet(dec_outs)
        spect, gate = spect_gate[:, :, 1:], spect_gate[:, :, :1]
        
        return spect, gate
    
    
    def infer(self, tgt_real, memory_bank, memory_lengths, decoder, postnet):
        B = memory_bank.size(1)
        device = memory_bank.device
        
        spect_outputs = torch.zeros((self.max_decoder_steps, B, self.dim_freq), 
                                    dtype=torch.float, device=device)
        gate_outputs = torch.zeros((self.max_decoder_steps, B, 1), 
                                   dtype=torch.float, device=device)
        tgt_words = torch.zeros([B, 1], 
                                 dtype=torch.float, device=device)
        
        current_pred = torch.zeros([1, B, self.dim_freq], 
                                    dtype=torch.float, device=device)
        
        for t in range(self.max_decoder_steps):
            
            dec_outs, _ = decoder(current_pred, 
                                  memory_bank, t, 
                                  memory_lengths=memory_lengths,
                                  tgt_words=tgt_words)
            spect_gate = postnet(dec_outs)
            
            spect, gate = spect_gate[:, :, 1:], spect_gate[:, :, :1]
            spect_outputs[t:t+1] = spect
            gate_outputs[t:t+1] = gate
            
            stop = (torch.sigmoid(gate) - self.gate_threshold + 0.5).round()
            current_pred = spect.data
            tgt_words = stop.squeeze(-1).t()
            
            if t == self.max_decoder_steps - 1:
                print(f"Warning! {self.type_out} reached max decoder steps")
            
            if (stop == 1).all():
                break
        
        stop_quant = (torch.sigmoid(gate_outputs.data) - self.gate_threshold + 0.5).round().squeeze(-1) 
        len_spect = (stop_quant.cumsum(dim=0)==0).sum(dim=0)
        
        return spect_outputs, len_spect, gate_outputs