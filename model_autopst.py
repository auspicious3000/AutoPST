import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import filter_bank_mean

from fast_decoders import DecodeFunc_Sp

from model_sea import Encoder_2 as Encoder_Code_2

from override_decoder import OnmtDecoder_1 as OnmtDecoder

from onmt_modules.misc import sequence_mask
from onmt_modules.embeddings import PositionalEncoding
from onmt_modules.encoder_transformer import TransformerEncoder as OnmtEncoder

      
        
class Prenet(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.1):
        super().__init__() 
        
        mlp = nn.Linear(dim_input, dim_output, bias=True)
        pe = PositionalEncoding(dropout, dim_output, 1600)
        
        self.make_prenet = nn.Sequential()
        self.make_prenet.add_module('mlp', mlp)
        self.make_prenet.add_module('pe', pe)
        
        self.word_padding_idx = 1
        
    def forward(self, source, step=None):
        
        for i, module in enumerate(self.make_prenet._modules.values()):
            if i == len(self.make_prenet._modules.values()) - 1:
                source = module(source, step=step)
            else:
                source = module(source)
                
        return source
    
    
    
class Decoder_Sp(nn.Module):
    """
    Speech Decoder
    """
    def __init__(self, hparams):
        super().__init__() 
        
        self.dim_freq = hparams.dim_freq
        self.max_decoder_steps = hparams.dec_steps_sp
        self.gate_threshold = hparams.gate_threshold
        
        prenet = Prenet(hparams.dim_freq, hparams.dec_rnn_size)
        self.decoder = OnmtDecoder.from_opt(hparams, prenet)

        self.postnet = nn.Linear(hparams.dec_rnn_size, 
                                 hparams.dim_freq+1, bias=True)
        
    def forward(self, tgt, tgt_lengths, memory_bank, memory_lengths):
        
        dec_outs, attns = self.decoder(tgt, memory_bank, step=None, 
                                       memory_lengths=memory_lengths,
                                       tgt_lengths=tgt_lengths)
        spect_gate = self.postnet(dec_outs)
        spect, gate = spect_gate[:, :, 1:], spect_gate[:, :, :1]
        
        return spect, gate
    
    
    
class Encoder_Tx_Spk(nn.Module):
    """
    Text Encoder
    """
    def __init__(self, hparams):
        super().__init__() 
        
        prenet = Prenet(hparams.dim_code+hparams.dim_spk, 
                        hparams.enc_rnn_size)
        self.encoder = OnmtEncoder.from_opt(hparams, prenet)
        
    def forward(self, src, src_lengths, spk_emb):
        
        spk_emb = spk_emb.unsqueeze(0).expand(src.size(0),-1,-1)
        src_spk = torch.cat((src, spk_emb), dim=-1)
        enc_states, memory_bank, src_lengths = self.encoder(src_spk, src_lengths)
        
        return enc_states, memory_bank, src_lengths
    
    
    
class Decoder_Tx(nn.Module):
    """
    Text Decoder with stop 
    and num_rep prediction
    """
    def __init__(self, hparams):
        super().__init__()        
        
        self.dim_code = hparams.dim_code
        self.max_decoder_steps = hparams.dec_steps_tx
        self.gate_threshold = hparams.gate_threshold
        self.dim_rep = hparams.dim_rep
        
        prenet = Prenet(hparams.dim_code, hparams.dec_rnn_size)
        self.decoder = OnmtDecoder.from_opt(hparams, prenet)

        self.postnet_1 = nn.Linear(hparams.dec_rnn_size, 
                                   hparams.dim_code+1, bias=True)
        
        self.postnet_2 = nn.Linear(hparams.dec_rnn_size, 
                                   self.dim_rep, bias=True)
        
    def forward(self, tgt, tgt_lengths, memory_bank, memory_lengths):
        
        dec_outs, attns = self.decoder(tgt, memory_bank, step=None, 
                                       memory_lengths=memory_lengths,
                                       tgt_lengths=tgt_lengths)
        gate_text = self.postnet_1(dec_outs)
        rep = self.postnet_2(dec_outs)
        gate, text = gate_text[:, :, :1], gate_text[:, :, 1:]
        
        return text, gate, rep
        
        
    
class Generator_1(nn.Module):
    '''
    sync stage 1
    '''
    def __init__(self, hparams):
        super().__init__() 
        
        self.encoder_cd = Encoder_Code_2(hparams)
        self.encoder_tx = Encoder_Tx_Spk(hparams)
        self.decoder_sp = Decoder_Sp(hparams)   
        self.encoder_spk = nn.Linear(hparams.dim_spk, 
                                     hparams.enc_rnn_size, bias=True)
        self.fast_dec_sp = DecodeFunc_Sp(hparams, 'Sp')
        
        
    def pad_sequences_rnn(self, cd_short, num_rep, len_long):
        B, L, C = cd_short.size()
        out_tensor = torch.zeros((B, len_long.max(), C), device=cd_short.device)
        '''
        len_long = len_spect + 1
        '''
        for i in range(B):
            code_sync = cd_short[i].repeat_interleave(num_rep[i], dim=0)
            out_tensor[i, :len_long[i]-1, :] = code_sync
            
        return out_tensor 

        
    def forward(self, cep_in, mask_long, codes_mask, num_rep, len_short,
                      tgt_spect, len_spect, 
                      spk_emb):
        
        cd_long = self.encoder_cd(cep_in, mask_long)
        fb = filter_bank_mean(num_rep, codes_mask, cd_long.size(1))
        
        cd_short = torch.bmm(fb.detach(), cd_long)
        
        cd_short_sync = self.pad_sequences_rnn(cd_short, num_rep, len_spect)
        
        spk_emb_1 = self.encoder_spk(spk_emb)
        
        # text to speech
        _, memory_tx, _ = self.encoder_tx(cd_short_sync.transpose(1,0), len_spect, 
                                          spk_emb)
        memory_tx_spk = torch.cat((spk_emb_1.unsqueeze(0), memory_tx), dim=0)
        self.decoder_sp.decoder.init_state(memory_tx_spk, None, None)
        spect_out, gate_sp_out \
        = self.decoder_sp(tgt_spect, len_spect, memory_tx_spk, len_spect+1)
        
        return spect_out, gate_sp_out
    
    
    def infer_onmt(self, cep_in, mask_long,
                   len_spect, 
                   spk_emb):
        
        cd_long = self.encoder_cd(cep_in, mask_long)
        
        spk_emb_1 = self.encoder_spk(spk_emb)
        
        # text to speech
        _, memory_tx, _ = self.encoder_tx(cd_long.transpose(1,0), len_spect,
                                          spk_emb)
        memory_tx_spk = torch.cat((spk_emb_1.unsqueeze(0), memory_tx), dim=0)
        self.decoder_sp.decoder.init_state(memory_tx_spk, None, None)
        spect_output, len_spect_out, stop_sp_output \
        = self.fast_dec_sp.infer(None, memory_tx_spk, len_spect+1, 
                                 self.decoder_sp.decoder, 
                                 self.decoder_sp.postnet)
        
        return spect_output, len_spect_out
    
    
    
class Generator_2(nn.Module):
    '''
    async stage 2
    '''
    def __init__(self, hparams):
        super().__init__() 
        
        self.encoder_cd = Encoder_Code_2(hparams)
        self.encoder_tx = Encoder_Tx_Spk(hparams)
        self.decoder_sp = Decoder_Sp(hparams)   
        self.encoder_spk = nn.Linear(hparams.dim_spk, 
                                     hparams.enc_rnn_size, bias=True)
        self.fast_dec_sp = DecodeFunc_Sp(hparams, 'Sp')
        
        
    def forward(self, cep_in, mask_long, codes_mask, num_rep, len_short,
                      tgt_spect, len_spect, 
                      spk_emb):
        
        cd_long = self.encoder_cd(cep_in, mask_long)
        fb = filter_bank_mean(num_rep, codes_mask, cd_long.size(1))
        
        cd_short = torch.bmm(fb.detach(), cd_long.detach())
        
        spk_emb_1 = self.encoder_spk(spk_emb)
        
        # text to speech
        _, memory_tx, _ = self.encoder_tx(cd_short.transpose(1,0), len_short, 
                                          spk_emb)
        memory_tx_spk = torch.cat((spk_emb_1.unsqueeze(0), memory_tx), dim=0)
        self.decoder_sp.decoder.init_state(memory_tx_spk, None, None)
        spect_out, gate_sp_out \
        = self.decoder_sp(tgt_spect, len_spect, memory_tx_spk, len_short+1)
        
        return spect_out, gate_sp_out
    
    
    def infer_onmt(self, cep_in, mask_long, len_spect,
                   spk_emb):
        
        cd_long = self.encoder_cd(cep_in, mask_long)
        
        spk_emb_1 = self.encoder_spk(spk_emb)
        
        # text to speech
        _, memory_tx, _ = self.encoder_tx(cd_long.transpose(1,0), len_spect, 
                                          spk_emb)
        memory_tx_spk = torch.cat((spk_emb_1.unsqueeze(0), memory_tx), dim=0)
        self.decoder_sp.decoder.init_state(memory_tx_spk, None, None)
        spect_output, len_spect_out, stop_sp_output \
        = self.fast_dec_sp.infer(None, memory_tx_spk, len_spect+1, 
                                 self.decoder_sp.decoder, 
                                 self.decoder_sp.postnet)
        
        return spect_output, len_spect_out