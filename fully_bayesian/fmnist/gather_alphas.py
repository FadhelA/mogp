import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F


from simulationmodel import BayesFFNN
from sampling_utils import GBFRYInitLearnableAlpha

if not os.path.isdir('results/summary'):
    os.makedirs('results/summary')
    
p_list = [500, 1000, 2000]
num_trials = 5
input_size = 28 * 28
num_hidden = 1
output_size = 10
datasize = 60000
alphas = {p:[] for p in p_list}

for i, p in enumerate(p_list):
    for j in range(num_trials):
        nets = []
        ckpts = torch.load(f"./results/sghmc/gbfry-{p}-trial{j}.net")
        for ckpt in ckpts:
            net = BayesFFNN(input_size, num_hidden, p, output_size)
            net.set_prior(GBFRYInitLearnableAlpha())
            net.load_state_dict(ckpt)          
            alphas[p].append(net.hidden_layers[0].var_dist.alpha.cpu().data.numpy()[0])

torch.save(alphas, 'results/alphas.data')