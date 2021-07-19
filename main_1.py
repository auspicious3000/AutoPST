import os
import argparse
import torch

from solver_1 import Solver
from data_loader import get_loader
from hparams_autopst import hparams, hparams_debug_string



def str2bool(v):
    return v.lower() in ('true')

def main(config):

    # Data loader
    data_loader = get_loader(hparams)
    
    # Solver for training 
    solver = Solver(data_loader, config, hparams)

    solver.train()
    
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Training configuration.
    parser.add_argument('--num_iters', type=int, default=1000000)

    # Miscellaneous.
    parser.add_argument('--device_id', type=int, default=0)

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    
    config = parser.parse_args()
    print(config)
    print(hparams_debug_string())
    main(config)