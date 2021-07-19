import os
import pickle 
import torch
import numpy as np
       
from numpy.random import uniform
from torch.utils import data
from torch.utils.data.sampler import Sampler
from multiprocessing import Process, Manager  



class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, hparams):
        """Initialize and preprocess the Utterances dataset."""
        self.meta_file = hparams.meta_file
        
        self.feat_dir_1 = hparams.feat_dir_1
        self.feat_dir_2 = hparams.feat_dir_2
        self.feat_dir_3 = hparams.feat_dir_3
        
        self.step = 4
        self.split = 0
         
        self.max_len_pad = hparams.max_len_pad
        
        meta = pickle.load(open(self.meta_file, "rb"))
        
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  # <-- can be shared between processes.
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        # very importtant to do dataset = list(dataset)            
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the {} Utterances training dataset...'.format(self.num_tokens))
        
        
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            for j, tmp in enumerate(sbmt):
                if j < 2: 
                    # fill in speaker name and embedding
                    uttrs[j] = tmp
                else:    
                    # fill in data
                    sp_tmp = np.load(os.path.join(self.feat_dir_1, tmp))
                    cep_tmp = np.load(os.path.join(self.feat_dir_2, tmp))[:, 0:14]
                    cd_tmp = np.load(os.path.join(self.feat_dir_3, tmp))
                    
                    assert len(sp_tmp) == len(cep_tmp) == len(cd_tmp)           

                    uttrs[j] = ( np.clip(sp_tmp, 0, 1), cep_tmp, cd_tmp )
            dataset[idx_offset+k] = uttrs
    
    
    def segment_np(self, cd_long, tau=2):
        
        cd_norm = np.sqrt((cd_long ** 2).sum(axis=-1, keepdims=True))
        G = (cd_long @ cd_long.T) / (cd_norm @ cd_norm.T)
        
        L = G.shape[0]
        
        num_rep = []
        num_rep_sync = []
        
        prev_boundary = 0
        rate = np.random.uniform(0.8, 1.3)
        
        for t in range(1, L+1):
            if t==L:
                num_rep.append(t - prev_boundary)
                num_rep_sync.append(t - prev_boundary)
                prev_boundary = t
            if t < L:
                q = np.random.uniform(rate-0.1, rate)
                tmp = G[prev_boundary, max(prev_boundary-20, 0):min(prev_boundary+20, L)]
                if q <= 1:
                    epsilon = np.quantile(tmp, q)
                    if np.all(G[prev_boundary, t:min(t+tau, L)] < epsilon):
                        num_rep.append(t - prev_boundary)
                        num_rep_sync.append(t - prev_boundary)
                        prev_boundary = t
                else:
                    epsilon = np.quantile(tmp, 2-q)
                    if np.all(G[prev_boundary, t:min(t+tau, L)] < epsilon):
                        num_rep.append(t - prev_boundary)    
                    else:
                        num_rep.extend([t-prev_boundary-0.5, 0.5])
                        
                    num_rep_sync.append(t - prev_boundary)    
                    prev_boundary = t
                    
        num_rep = np.array(num_rep)
        num_rep_sync = np.array(num_rep_sync)
        
        return num_rep, num_rep_sync
            
        
    def __getitem__(self, index):
        """Return M uttrs for one spkr."""
        dataset = self.train_dataset
        
        list_uttrs = dataset[index]
        
        emb_org = list_uttrs[1]
        
        uttr = np.random.randint(2, len(list_uttrs))
        melsp, melcep, cd_real = list_uttrs[uttr]
        
        num_rep, num_rep_sync = self.segment_np(cd_real)
        
        return melsp, melcep, cd_real, num_rep, num_rep_sync, len(melsp), len(num_rep), len(num_rep_sync), emb_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    

class MyCollator(object):
    def __init__(self, hparams):
        self.max_len_pad = hparams.max_len_pad
        
    def __call__(self, batch):
        new_batch = []
        
        l_short_max = 0
        l_short_sync_max = 0
        l_real_max = 0
        
        for token in batch:
            sp_real, cep_real, cd_real, rep, rep_sync, l_real, l_short, l_short_sync, emb = token
            
            if l_short > l_short_max:
                l_short_max = l_short  
                
            if l_short_sync > l_short_sync_max:
                l_short_sync_max = l_short_sync
            
            if l_real > l_real_max:
                l_real_max = l_real
            
            sp_real_pad = np.pad(sp_real, ((0,self.max_len_pad-l_real),(0,0)), 'constant')
            cep_real_pad = np.pad(cep_real, ((0,self.max_len_pad-l_real),(0,0)), 'constant')
            cd_real_pad = np.pad(cd_real, ((0,self.max_len_pad-l_real),(0,0)), 'constant')
            
            rep_pad = np.pad(rep, (0,self.max_len_pad-l_short), 'constant')
            rep_sync_pad = np.pad(rep_sync, (0,self.max_len_pad-l_short_sync), 'constant')
            
            new_batch.append( (sp_real_pad, cep_real_pad, cd_real_pad, rep_pad, rep_sync_pad, l_real, l_short, l_short_sync, emb) ) 
            
        batch = new_batch  
        
        a, b, c, d, e, f, g, h, i = zip(*batch)
        
        sp_real = torch.from_numpy(np.stack(a, axis=0))[:,:l_real_max+1,:]
        cep_real = torch.from_numpy(np.stack(b, axis=0))[:,:l_real_max+1,:]
        cd_real = torch.from_numpy(np.stack(c, axis=0))[:,:l_real_max+1,:]
        num_rep = torch.from_numpy(np.stack(d, axis=0))[:,:l_short_max+1]
        num_rep_sync = torch.from_numpy(np.stack(e, axis=0))[:,:l_short_sync_max+1]
        
        len_real = torch.from_numpy(np.stack(f, axis=0))
        len_short = torch.from_numpy(np.stack(g, axis=0))
        len_short_sync = torch.from_numpy(np.stack(h, axis=0))
        
        spk_emb = torch.from_numpy(np.stack(i, axis=0))
        
        return sp_real, cep_real, cd_real, num_rep, num_rep_sync, len_real, len_short, len_short_sync, spk_emb


    
class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.
    """
    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self):
        self.sample_idx_array = torch.arange(self.num_samples, dtype=torch.int64).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array))]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)        
    
    
    
def worker_init_fn(x):
    return np.random.seed((torch.initial_seed()) % (2**32))    

def get_loader(hparams):
    """Build and return a data loader."""
    
    dataset = Utterances(hparams)
    
    my_collator = MyCollator(hparams)
    
    sampler = MultiSampler(len(dataset), hparams.samplier, shuffle=hparams.shuffle)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=hparams.batch_size,
                                  sampler=sampler,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=False,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=my_collator)
    return data_loader