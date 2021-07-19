import os
import time
import pickle
import datetime
import itertools
import numpy as np
import torch
import torch.nn.functional as F

from onmt_modules.misc import sequence_mask
from model_autopst import Generator_1 as Predictor



class Solver(object):

    def __init__(self, data_loader, config, hparams):
        """Initialize configurations."""

        
        self.data_loader = data_loader
        self.hparams = hparams
        self.gate_threshold = hparams.gate_threshold
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')
        self.num_iters = config.num_iters
        self.log_step = config.log_step
        
        # Build the model
        self.build_model()
    
            
    def build_model(self):
        
        self.P = Predictor(self.hparams)
        
        self.optimizer = torch.optim.Adam(self.P.parameters(), 0.0001, [0.9, 0.999])
        
        self.P.to(self.device)
        
        self.BCELoss = torch.nn.BCEWithLogitsLoss().to(self.device)    
    
                
    def train(self):
        # Set data loader
        data_loader = self.data_loader
        data_iter = iter(data_loader)
        
        
        # Print logs in specified order
        keys = ['P/loss_tx2sp', 'P/loss_stop_sp']
        
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            try:
                sp_real, cep_real, cd_real, _, num_rep_sync, len_real, _, len_short_sync, spk_emb = next(data_iter)
            except:
                data_iter = iter(data_loader)
                sp_real, cep_real, cd_real, _, num_rep_sync, len_real, _, len_short_sync, spk_emb = next(data_iter)
                
            
            sp_real = sp_real.to(self.device)
            cep_real = cep_real.to(self.device)
            cd_real = cd_real.to(self.device)
            len_real = len_real.to(self.device)
            spk_emb = spk_emb.to(self.device)
            num_rep_sync = num_rep_sync.to(self.device)
            len_short_sync = len_short_sync.to(self.device)
            
            
            # real spect masks
            mask_sp_real = ~sequence_mask(len_real, sp_real.size(1))
            mask_long = (~mask_sp_real).float()
            
            len_real_mask = torch.min(len_real + 10, 
                                      torch.full_like(len_real, sp_real.size(1)))
            loss_tx2sp_mask = sequence_mask(len_real_mask, sp_real.size(1)).float().unsqueeze(-1)
            
            # text input masks
            codes_mask = sequence_mask(len_short_sync, num_rep_sync.size(1)).float()
            
            
            # =================================================================================== #
            #                                    2. Train                                         #
            # =================================================================================== #
            
            self.P = self.P.train()
            
            
            sp_real_sft = torch.zeros_like(sp_real)
            sp_real_sft[:, 1:, :] = sp_real[:, :-1, :]    
            
            
            spect_pred, stop_pred_sp = self.P(cep_real.transpose(2,1),
                                              mask_long,
                                              codes_mask,
                                              num_rep_sync,
                                              len_short_sync+1,
                                              sp_real_sft.transpose(1,0), 
                                              len_real+1,
                                              spk_emb)
                        
            
            loss_tx2sp = (F.mse_loss(spect_pred.permute(1,0,2), sp_real, reduction='none')
                          * loss_tx2sp_mask).sum() / loss_tx2sp_mask.sum()
                          
            loss_stop_sp = self.BCELoss(stop_pred_sp.squeeze(-1).t(), mask_sp_real.float())
            
            loss_total = loss_tx2sp + loss_stop_sp
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            

            # Logging
            loss = {}
            loss['P/loss_tx2sp'] = loss_tx2sp.item()
            loss['P/loss_stop_sp'] = loss_stop_sp.item()
            

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)
                
                
            # Save model checkpoints.
            if (i+1) % 10000 == 0:
                torch.save({'model': self.P.state_dict(),
                            'optimizer': self.optimizer.state_dict()}, f'./assets/{i+1}-A.ckpt')
                print('Saved model checkpoints into assets ...')  