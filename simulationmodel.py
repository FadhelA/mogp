# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:49:41 2021

Script to sample from the proposed model

@author: caron
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F

from numpy.linalg import svd

from sampling_utils import lam_sampler, IIDInit, InvGammaInit, BetaInit, HorseshoeInit

torch.set_default_tensor_type(
    torch.DoubleTensor
)  # set default to double - otherwise, need to add .float() when using numpy double arrays

class BayesLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 kappa: float = 1., sigma_b: float = 1.
                ):
        super(BayesLinear, self).__init__(in_features, out_features, bias)

        self.var_dist = IIDInit(in_features)
        self.register_buffer('kappa', torch.tensor(kappa))
        self.register_buffer('sigma_b', torch.tensor(sigma_b))
        self.trunc_eps = 0

        self.init_weights()

    def init_weights(self, dist=None, kappa=None, sigma_b=None) -> None:
        if kappa is not None:
            self.kappa = torch.tensor(kappa)
        if sigma_b is not None:
            self.sigma_b = torch.tensor(sigma_b)
        if dist is not None:
            self.var_dist = dist

        self.weight = nn.Parameter(torch.from_numpy(ss.norm.rvs(
            scale=self.kappa, size=(self.out_features, self.in_features)
        )))
        self.bias = nn.Parameter(torch.from_numpy(ss.norm.rvs(scale=self.sigma_b, size=self.out_features)))

        self.register_parameter('transformed_variances', nn.Parameter(
            self.var_dist.transform(self.var_dist.rvs((1, self.in_features)))
        ))

        if self.var_dist.is_static:
            self.transformed_variances.requires_grad_(False)
        else:
            self.transformed_variances.requires_grad_(True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lam = self.get_variances().sqrt()
        active = torch.ones_like(lam)
        if not self.training:
            active = lam >= self.trunc_eps
        return F.linear(input, self.weight*lam*active, self.bias)
    
    def get_variances(self):
        #print(self.transformed_variances.detach())
        return self.var_dist.map_to_domain(self.transformed_variances)

    def log_prior(self):
        weight_term = -torch.sum(self.weight**2)/2/self.kappa
        bias_term = 0
        if self.sigma_b > 0:
            bias_term = -torch.sum(self.bias**2)/2/self.sigma_b
        var_term = 0
        if not self.var_dist.is_static:
            var_term = torch.sum(self.var_dist.log_pdf(self.get_variances()))
        return weight_term+bias_term+var_term
    
    def set_prior(self, dist):
        self.transformed_variances = nn.Parameter(dist.transform(self.get_variances()))
        self.var_dist = dist


class BayesFFNN(nn.Module):
    """Feed Forward Neural Network"""

    def __init__(self, input_size, num_hidden_layers, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.L = num_hidden_layers + 1
        self.p = hidden_size
        self.output_size = output_size

        # Create first hidden layer
        self.input_layer = BayesLinear(input_size, hidden_size)

        # Create remaining hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(BayesLinear(hidden_size, hidden_size))

        # Create output layer
        self.output_layer = BayesLinear(hidden_size, output_size)

    def forward(self, x):

        # Input to hidden
        output = self.input_layer(x)
        output = relu(output)

        # Hidden to hidden
        for layer in self.hidden_layers:
            output = layer(output)
            output = relu(output)

        # Output
        output = self.output_layer(output)
        return output

    def init_weights(self, dist, kappa=1, sigma_b=1):
        num_hidden_layers = self.L - 1
        p = self.p

        # Initialise Input layer (usual update)
        self.input_layer.init_weights(IIDInit(self.input_size), kappa, sigma_b)

        # Initialise Hidden layers
        for i in range(num_hidden_layers):
            self.hidden_layers[i].init_weights(dist, kappa, sigma_b)

        # Initialise Output layer
        self.output_layer.init_weights(dist, kappa, sigma_b)
        
    def set_prior(self, dist):
        num_hidden_layers = self.L - 1
        p = self.p

        # Initialise Hidden layers
        for i in range(num_hidden_layers):
            self.hidden_layers[i].set_prior(dist)

        # Initialise Output layer
        self.output_layer.set_prior(dist)      

    def log_prior(self):
        res = self.input_layer.log_prior()
        for layer in self.hidden_layers:
            res += layer.log_prior()
        res += self.output_layer.log_prior()

        return res
    
    def truncate(self, eps):
        # Initialise Hidden layers
        for layer in self.hidden_layers:
            layer.trunc_eps = eps

        # Initialise Output layer
        self.output_layer.trunc_eps = eps

class FFNN(nn.Module):
    """Feed Forward Neural Network"""

    def __init__(self, input_size, num_hidden_layers, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.L = num_hidden_layers + 1
        self.p = hidden_size
        self.output_size = output_size

        # Create first hidden layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Create remaining hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # Input to hidden
        output = self.input_layer(x)
        output = relu(output)

        # Hidden to hidden
        for layer in self.hidden_layers:
            output = layer(output)
            output = relu(output)

        # Output
        output = self.output_layer(output)
        return output

    def init_weights(self, lam_rvs, kappa=1, sigma_b=1):
        num_hidden_layers = self.L - 1
        p = self.p

        # Initialise Input layer (usual update)
        custom_weight = torch.from_numpy(ss.norm.rvs(scale=kappa / np.sqrt(self.input_size), size=(p, self.input_size)))
        custom_bias = torch.from_numpy(ss.norm.rvs(scale=sigma_b, size=p))
        self.input_layer.weight.data = custom_weight
        self.input_layer.bias.data = custom_bias

        # Initialise Hidden layers
        for i in range(num_hidden_layers):
            lam = torch.from_numpy(lam_rvs(p))  # sample the variances
            v = torch.from_numpy(ss.norm.rvs(scale=kappa, size=(p, p)))
            custom_weight = v * torch.sqrt(lam.reshape((1, -1)))
            self.hidden_layers[i].weight.data = custom_weight

            custom_bias = torch.from_numpy(ss.norm.rvs(scale=sigma_b, size=p))
            self.hidden_layers[i].bias.data = custom_bias

        # Initialise Output layer
        lam = torch.from_numpy(lam_rvs(p))
        v = torch.from_numpy(ss.norm.rvs(scale=kappa, size=(self.output_size, p)))
        custom_weight = v * torch.sqrt(lam.reshape((1, -1)))
        self.output_layer.weight.data = custom_weight

        custom_bias = torch.from_numpy(ss.norm.rvs(scale=sigma_b, size=self.output_size))
        self.output_layer.bias.data = custom_bias

if __name__ == "__main__":
    # Create a FFNN with the given dimensions
    input_size = 1
    p = 2000
    num_hidden = 3
    output_size = 100
    nn = FFNN(input_size, num_hidden, p, output_size)

    # Sample the weights and outputs using IID
    lam_rvs = lam_sampler(p, 'iid')  # lam_rvs is a function to sample the variances
    nn.init_weights(lam_rvs, kappa=np.sqrt(2))  # this initialises the weights of the nn

    # Show outputs for multiple 1D inputs
    x_torch = torch.from_numpy(np.asmatrix(np.arange(-1, 1, step=0.001)).T)
    y_torch = nn.forward(x_torch)

    x = x_torch.detach().numpy()
    y = y_torch.detach().numpy()
    plt.figure()
    plt.plot(x, y)
    plt.show()

    s, d1, v = svd(y)  # SVD to look at the the feature extraction properties

    # print('correlation between outputs=', np.corrcoef(y.T))

    # Using another prior
    lam_rvs = lam_sampler(p, 'horseshoe')
    nn.init_weights(lam_rvs, kappa=np.sqrt(2))

    # Plot input/output
    y_torch = nn.forward(x_torch)
    x = x_torch.detach().numpy()
    y = y_torch.detach().numpy()

    plt.figure()
    plt.plot(x, y)
    print('correlation between outputs=', np.corrcoef(y.T))

    s, d2, v = svd(y)
    plt.figure()
    plt.plot(np.arange(np.min([p, output_size])) + 1, np.cumsum(np.sort(d1)[::-1]) / np.sum(np.sort(d1)[::-1]))
    plt.plot(np.arange(np.min([p, output_size])) + 1, np.cumsum(np.sort(d2)[::-1]) / np.sum(np.sort(d2)[::-1]))
    plt.xlim([0, 50])
    plt.legend(['iid', 'noniid'])
    plt.ylabel('pct of variance explained')
    plt.xlabel('eigenvalue')
    plt.show()
