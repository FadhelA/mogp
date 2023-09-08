import torch
import torch.nn as nn
import sys
sys.path.append('../')

import os
import numpy as np
from tqdm import tqdm
from fast_nist import FastMNIST
from copy import deepcopy

from simulationmodel import BayesFFNN, BayesLinear
from sampling_utils import *

train = FastMNIST("../data", train=True, download=True, device='cuda')
trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
test = FastMNIST("../data", train=False, download=True, device='cuda')
testset = torch.utils.data.DataLoader(test, batch_size=100)

# sghmc parameters
alpha = 0.1
temp = 1.0

# Create FFNN models with the given dimensions
input_size = 28 * 28
num_hidden = 1
output_size = 10
datasize = 60000
num_batch = datasize//100

epochs = 100
T = num_batch * epochs
base_lr = 0.5

if not os.path.isdir('./results/sghmc'):
    os.makedirs('./results/sghmc')

def learning_rate(epoch, batch_idx, min_lr=0.1):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % T)
    cos_inner /= T
    cos_out = np.cos(cos_inner) + 1
    return max(min_lr, 0.5 * cos_out*base_lr)

def update_params(net, epoch, lr):
    for p in net.parameters():
        if not hasattr(p, 'buf'):
            p.buf = torch.zeros_like(p)

        if p.requires_grad:
            d_p = p.grad.data
            buf_new = (1 - alpha)*p.buf - lr*d_p
            if epoch > 0.5 * epochs:
                eps = torch.randn_like(p)
                buf_new += (2.0*lr*alpha*temp/datasize)**(0.5)*eps
            p.data.add_(buf_new)
            p.buf = buf_new

# Train the models
def train_model(net, epochs, net_name, p):
    net.float()
    net.train()
    samples = []
    for epoch in tqdm(range(epochs)):

        net.train()
        for bid, (X, y) in enumerate(trainset):
            net.zero_grad()
            pred_dist = net(X.view(-1, 28 * 28))

            l = nn.CrossEntropyLoss(reduction="mean")
            log_prior_term = net.log_prior()/datasize
            loss = l(pred_dist, y)-log_prior_term
            loss.backward()
            update_params(net, epoch, learning_rate(epoch, bid))

        line = f"{net_name}_{p} "
        line += f"training loss = {loss:.4f} lr {learning_rate(epoch, bid):.2e} "
        if net_name == 'gbfry':
            alpha = net.hidden_layers[0].var_dist.alpha.cpu().data.numpy()[0]
            line += f"alpha {alpha:.4f} "

        net.eval()
        accurate = 0
        total = 0
        with torch.no_grad():
            for X, y in testset:
                pred_dist = net(X.view(-1, 28 * 28))
                pred_y = torch.argmax(pred_dist, dim=-1)
                accurate += torch.sum(pred_y == y).cpu().data.numpy()
                total += y.shape[0]

        if epoch > epochs // 2 and (epoch+1) % 2 == 0:
            samples.append(deepcopy(net.state_dict()))

        acc = round(accurate / total * 100, 1)
        line += f"test acc {acc:.4f}"
        tqdm.write(line)

    return samples

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='normal',
            choices=['normal', 'horseshoe', 'gbfry', 'gbfry_alpha'])
    parser.add_argument('--num_trials', type=int, default=5)
    args = parser.parse_args()

    p_list = [500, 1000]

    for p in p_list:
        iid_rvs = lam_dist(p, 'iid')
        if args.model == 'normal':
            for i in range(args.num_trials):
                normal_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
                normal_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)
                normal_ffnn = normal_ffnn.cuda()
                samples = train_model(normal_ffnn, epochs, 'normal', p)
                torch.save(samples,
                        f'./results/sghmc/normal-{p}-trial{i}.net')
        elif args.model == 'horseshoe':
            for i in range(args.num_trials):
                horseshoe_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
                horseshoe_rvs = lam_dist(p, 'horseshoe')
                horseshoe_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)
                horseshoe_ffnn.set_prior(horseshoe_rvs)
                horseshoe_ffnn = horseshoe_ffnn.cuda()
                samples = train_model(horseshoe_ffnn, epochs, 'horseshoe', p)
                torch.save(samples,
                        f'./results/sghmc/horseshoe-{p}-trial{i}.net')
        elif args.model == 'gbfry':
            for i in range(args.num_trials):
                gbfry_ffnn = BayesFFNN(input_size, num_hidden, p, output_size)
                gbfry_rvs = GBFRYInitLearnableAlpha()
                gbfry_ffnn.init_weights(iid_rvs, kappa=1, sigma_b=1)
                gbfry_ffnn.set_prior(gbfry_rvs)
                gbfry_ffnn = gbfry_ffnn.cuda()
                samples = train_model(gbfry_ffnn, epochs, f'gbfry', p)
                torch.save(samples,
                        f'./results/sghmc/gbfry-{p}-trial{i}.net')
