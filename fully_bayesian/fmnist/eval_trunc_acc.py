import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn

from fast_nist import FastFMNIST

from simulationmodel import BayesFFNN
from sampling_utils import lam_dist, GBFRYInitLearnableAlpha

if not os.path.isdir('results'):
    os.makedirs('results')

train = FastFMNIST("../data", train=True, download=True, device='cuda')
trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
test = FastFMNIST("../data", train=False, download=True, device='cuda')
testset = torch.utils.data.DataLoader(test, batch_size=100)

input_size = 28 * 28
num_hidden = 1
output_size = 10
datasize = 60000

def compute_accuracy(models):
    for model in models:
        model.eval()

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for X, y in testset:
            pred_dists = []
            for model in models:
                pred_dists.append(model(X.view(-1, 28*28)))
            pred_dist = torch.stack(pred_dists).logsumexp(0)
            correct += (pred_dist.argmax(-1)==y).float().sum()
            total += y.shape[0]

    acc = (correct/total).cpu().data.numpy()
    return round(acc*100, 3)

def truncate_quantile(model, q):
    model.eval()
    for layer in model.hidden_layers:
        layer.trunc_eps = torch.quantile(layer.get_variances()**0.5, q)

    model.output_layer.trunc_eps = \
            torch.quantile(model.output_layer.get_variances()**0.5, q)

def transform_normal(model):
    model.eval()
    for layer in model.hidden_layers:
        if layer.var_dist.is_static:
            w = torch.sum(layer.weight.detach()**2, axis=0).reshape(1,-1)
            layer.transformed_variances.data = layer.transformed_variances.data * w
            w[w==0] = 1
            layer.weight.data = layer.weight.data / (w**0.5)

    layer = model.output_layer
    if layer.var_dist.is_static:
        w = torch.sum(layer.weight.detach()**2, axis=0).reshape(1,-1)
        layer.transformed_variances.data = layer.transformed_variances.data * w
        w[w==0] = 1
        layer.weight.data = layer.weight.data / (w**0.5)

q_list = [0.0, .5, .8, .9, .95, .99]
p_list = [500, 1000, 2000]
model_list = ['normal', 'horseshoe', 'gbfry']
num_trials = 5
acc = {model:np.zeros((len(p_list), len(q_list), num_trials))
            for model in model_list}
for model in model_list:
    acc[model] = np.zeros((len(p_list), len(q_list), num_trials))

for model in tqdm(model_list):
    for i, p in enumerate(p_list):
        for j in range(num_trials):
            nets = []
            ckpts = torch.load(f"./results/sghmc/{model}-{p}-trial{j}.net")
            for ckpt in ckpts:
                net = BayesFFNN(input_size, num_hidden, p, output_size)
                if model == 'horseshoe':
                    net.set_prior(lam_dist(p, 'horseshoe'))
                elif model.startswith('gbfry'):
                    net.set_prior(GBFRYInitLearnableAlpha())
                net.load_state_dict(ckpt)
                nets.append(net.cuda())

            for k, q in enumerate(q_list):
                for net in nets:
                    if model == 'normal':
                        transform_normal(net)
                    if q > 0.0:
                        truncate_quantile(net, q)
                acc[model][i,k,j] = compute_accuracy(nets)

torch.save(acc, 'results/trunc_acc.data')