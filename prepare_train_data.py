import os
import pickle
import numpy as np
import scipy.fftpack
import soundfile as sf
from utils import pySTFT
from scipy import signal
from librosa.filters import mel
from utils import butter_highpass

import torch
import torch.nn.functional as F
from model_sea import Generator as Model 
from hparams_sea import hparams

    
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

mfcc_mean, mfcc_std, dctmx = pickle.load(open('assets/mfcc_stats.pkl', 'rb'))
spk2emb = pickle.load(open('assets/spk2emb_82.pkl', 'rb'))

rootDir = "assets/vctk16-train-wav"
targetDir_sp = 'assets/vctk16-train-sp-mel'
targetDir_cep = 'assets/vctk16-train-cep-mel'
targetDir_cd = 'assets/vctk16-train-teacher'

device = 'cuda:0'

G = Model(hparams).eval().to(device)
            
g_checkpoint = torch.load('assets/sea.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'], strict=True)


metadata = []
dirName, subdirList, _ = next(os.walk(rootDir))

for subdir in sorted(subdirList):
    print(subdir)
    
    if not os.path.exists(os.path.join(targetDir_sp, subdir)):
        os.makedirs(os.path.join(targetDir_sp, subdir))
    if not os.path.exists(os.path.join(targetDir_cep, subdir)):
        os.makedirs(os.path.join(targetDir_cep, subdir)) 
    if not os.path.exists(os.path.join(targetDir_cd, subdir)):
        os.makedirs(os.path.join(targetDir_cd, subdir))     
    
    submeta = []
    submeta.append(subdir)
    submeta.append(spk2emb[subdir])
    
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    
    for fileName in sorted(fileList):
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        D = pySTFT(y * 0.96).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel))

        # mel sp
        S = (D_db + 80) / 100 
                
        # mel cep
        cc_tmp = S.dot(dctmx)
        cc_norm = (cc_tmp - mfcc_mean) / mfcc_std
        S = np.clip(S, 0, 1)
        
        # teacher code
        cc_torch = torch.from_numpy(cc_norm[:,0:20].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            codes = G.encode(cc_torch, torch.ones_like(cc_torch[:,:,0])).squeeze(0)
            
        np.save(os.path.join(targetDir_cd, subdir, fileName[:-4]), 
                codes.cpu().numpy(), allow_pickle=False)
        np.save(os.path.join(targetDir_sp, subdir, fileName[:-4]), 
                S.astype(np.float32), allow_pickle=False)
        np.save(os.path.join(targetDir_cep, subdir, fileName[:-4]), 
                cc_norm.astype(np.float32), allow_pickle=False)
        
        submeta.append(subdir+'/'+fileName[:-4]+'.npy')
            
    metadata.append(submeta)
        
with open('./assets/train_vctk.meta', 'wb') as handle:
    pickle.dump(metadata, handle)  