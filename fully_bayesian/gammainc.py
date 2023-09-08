import torch
import torch.nn as nn
from torch.special import gammainc, gammaln
from torch.autograd import Function

class GammaInc(Function):

    @staticmethod
    def forward(ctx, a, x):
        ctx.save_for_backward(a, x)
        output = gammainc(a, x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, x = ctx.saved_tensors
        grad_a = grad_x = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output * ((gammainc(a+1e-5, x) - gammainc(a, x))/1e-5)
        if ctx.needs_input_grad[1]:
            grad_x = grad_output * (x**(a-1)*torch.exp(-x) / torch.exp(gammaln(a)))

        return grad_a, grad_x

mygammainc = GammaInc.apply

#a = torch.Tensor([2.0])
#x = nn.Parameter(torch.exp(torch.randn(1)))
#
#print(torch.autograd.grad(mygammainc(a,x), x))
#print(torch.autograd.grad(torch.exp(mygammainc(a,x)), x))
#print(torch.autograd.grad(gammainc(a,x), x))
#print(torch.autograd.grad(torch.exp(gammainc(a,x)), x))
