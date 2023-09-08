import math

import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

from torch.distributions import Normal, Gamma, Pareto, Beta, Uniform

import numpy as np
from scipy.special import logsumexp
from torch import Tensor

class CRMLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 alpha: float=.1, tau: float=2., mu: float=2.,
                 he_init=False
                ):

        self.alpha = alpha
        self.tau = tau
        self.mu = mu
        self.he_init = he_init

        super(CRMLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def reset_parameters(self) -> None:
        super(CRMLinear, self).reset_parameters()

        if self.he_init:
            return

        var_mat = sample_etbfry(self.alpha, self.tau, self.mu, shape=(1, self.in_features))

        self.weight.data = torch.randn(size=(self.out_features, self.in_features))*var_mat.sqrt()

    def reset_mask(self, eta=None):
        if eta is None:
            self.mask = torch.ones_like(self.weight.data)
        else:
            self.mask = (torch.abs(self.weight.data) > eta).float()

    def reset_mask_neurons(self, eta=None):
        if eta is None:
            self.mask = torch.ones_like(self.weight.data, requires_grad=False)
        else:
            neurons_mask = (self.weight.data**2).sum(axis=1).view(-1,1) > eta
            self.mask = neurons_mask*torch.ones_like(self.weight.data, requires_grad=False)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight*self.mask, self.bias)

def sample_etbfry(alpha, tau, mu=1., shape=(5,1)):
    tau = np.maximum(tau, 1.01)

    out_features = shape[0]
    in_features = shape[1]

    c = mu*(tau-1)/(tau-alpha)

    eta = c**alpha/math.gamma(1-alpha)

    # gamma_mat = Gamma((1-alpha)*torch.ones(shape),
    #                      torch.ones(shape)).sample()

    s_mat = Uniform(torch.zeros(shape),
                    torch.ones(shape)
                   ).sample()

    log_tl = np.log(alpha*in_features*tau / eta / (tau-alpha)) / alpha

    tens = torch.ones((2, out_features, in_features))
    tens[0, :, :] = torch.log(s_mat)
    tens[1, :, :] = torch.log(1-s_mat) + alpha*logsumexp((log_tl, 1))
    log_w = -1/alpha * torch.logsumexp(tens, axis=0)

    pareto_mat = Pareto(scale=torch.ones(shape), alpha = tau*torch.ones(shape)).sample()

    return c*torch.exp(log_w)*pareto_mat
