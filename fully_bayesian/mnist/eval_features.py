import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fast_nist import FastMNIST
import matplotlib.pyplot as plt

from simulationmodel import BayesFFNN, BayesLinear
from sampling_utils import lam_dist, GBFRYInitLearnableAlpha

if not os.path.isdir('results/summary'):
    os.makedirs('results/summary')

input_size = 28 * 28
num_hidden = 1
output_size = 10
datasize = 60000

test = FastMNIST("../data", train=False, download=True, device='cuda')
testset = torch.utils.data.DataLoader(test, batch_size=10000)
X, Y = list(testset)[0]

X_tr, Y_tr = X[:5000], Y[:5000]
X_te, Y_te = X[5000:], Y[5000:]

def test_model(feat_tr, y_tr, feat_te, y_te, num_train):
    mlp = nn.Sequential(
            nn.Linear(n_neurons, 100), nn.ReLU(),
            nn.Linear(100, n_neurons)).cuda()
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    train_idxs = torch.randperm(len(feat_tr))[:num_train]
    feat_tr = feat_tr[train_idxs]
    y_tr = y_tr[train_idxs]

    opt = torch.optim.Adam(params=mlp.parameters(), lr=1e-4)
    num_iter = 1000
    for i in range(1, num_iter+1):
        mlp.zero_grad()
        logits = mlp(feat_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = np.round((mlp(feat_tr).argmax(-1)==y_tr).
                             float().mean().item(), 4)
    with torch.no_grad():
        test_acc = np.round((mlp(feat_te).argmax(-1)==y_te).
                            float().mean().item(), 4)

    return train_acc, test_acc

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

# Load trained models
hidden_neurons = [0]

def forward_hook(m, i, o):
    hidden_neurons[0] = o

p_list = [500, 1000, 2000]
model_list = ['normal', 'horseshoe', 'gbfry']
keys = ['imgs', 'idxs']
collected = {}
for key in keys:
    collected[key] = {model:[] for model in model_list}
n_neurons = 30
n_figures = 30
num_trials = 5
num_tests_per_trial = 5
num_trains = [200, 400, 600, 800]
acc = {model:np.zeros((len(p_list), len(num_trains), num_trials, num_tests_per_trial, 2))
        for model in model_list}

for i, p in enumerate(p_list):
    for model in model_list:
        imgs_all = []
        idxs_all = []
        for j in range(num_trials):
            print(f'processing {model}-{p}-trial{j}...')
            neurons = []
            variances = []
            neurons_test = []
            ckpts = torch.load(f'./results/sghmc/{model}-{p}-trial{i}.net')
            for ckpt in ckpts:
                net = BayesFFNN(input_size, num_hidden, p, output_size).cuda()
                if model == 'horseshoe':
                    net.set_prior(lam_dist(p, 'horseshoe'))
                elif model == 'gbfry':
                    net.set_prior(GBFRYInitLearnableAlpha())
                transform_normal(net)
                net.load_state_dict(ckpt)
                net.hidden_layers[0].register_forward_hook(forward_hook)
                with torch.no_grad():
                    net(X_tr.view(-1, 784))
                    neurons.append(hidden_neurons[0])
                    variances.append(net.output_layer.get_variances().flatten())
                    net(X_te.view(-1, 784))
                    neurons_test.append(hidden_neurons[0])

            neurons = torch.stack(neurons).mean(0)
            variances = torch.stack(variances).mean(0)
            neuron_idxs = torch.argsort(-variances)[:n_neurons]
            figure_idxs = torch.argsort(-neurons[:,neuron_idxs], axis=0)[:n_figures].view(-1)
            neurons_test = torch.stack(neurons_test).mean(0)

            imgs_all.append(X_tr[figure_idxs].cpu().data.numpy())
            idxs_all.append(figure_idxs.cpu().data.numpy())

            feat_tr = neurons[figure_idxs][..., neuron_idxs]
            y_tr = Y_tr[figure_idxs]
            feat_te = neurons_test[...,neuron_idxs]

            for k, num_train in enumerate(num_trains):
                for l in range(num_tests_per_trial):
                    train_acc, test_acc = test_model(
                            feat_tr, y_tr, feat_te, Y_te, num_train)
                    acc[model][i,k,j,l,0] = train_acc
                    acc[model][i,k,j,l,1] = test_acc
                    print(f'trial {l}, num_train {num_train}, train acc {train_acc:.4f}, test acc {test_acc:.4f}')

collected['acc'] = acc
torch.save(collected, 'results/features.data')
