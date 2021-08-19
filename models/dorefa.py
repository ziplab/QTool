
import torch
import torch.nn as nn
import numpy as np

__EPS__ = 1e-5

##########
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ratio=1.0):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

##########
class LSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        #n = int(n)
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

##############################################################
def GradientScale(x, scale):
    if int(scale) == 1:
        return x
    yGrad = x * scale
    return (x - yGrad).detach() + yGrad

def ClampWithScale(x, min=0, max=1):
    filtered = (x >= min) & (x <= max)
    n_pass = filtered.sum().item()
    if n_pass == 0:
      return torch.clamp(x, min=min, max=max)

    scale = x.numel() / n_pass
    scale = np.sqrt(scale)
    y = torch.clamp(x, min=min, max=max)
    return GradientScale(y, scale)

##############################################################
## Dorefa-net (https://arxiv.org/pdf/1606.06160.pdf)
class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_levels, clip_val=1.0, adaptive='none'):
        n = float(num_levels - 1)
        scale = n / clip_val
        out = torch.round(input * scale) / scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

## 
class DorefaParamsBinarizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, adaptive='none'):
        E = x.detach().abs().mean()
        y = torch.ones_like(x)
        y.masked_fill_(x < 0, -1)  # use compare rather than sign() to handle the zero
        y.mul_(E)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

## TTN (https://arxiv.org/pdf/1612.01064v1)
class TTN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wp, wn, thre):
        y = x.clone()
        quant_group = thre.size(0)
        y = y.reshape(quant_group, -1)
        thre_y = y.abs().max(dim=1, keepdim=True)[0] * thre
        thre_y = thre_y.expand_as(y)
        a = (y > thre_y).float()
        b = (y <-thre_y).float()
        y = a * wp - b * wn
        y = y.reshape(x.shape)
        ctx.save_for_backward(a, b, wp, wn)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        a, b, wp, wn = ctx.saved_tensors
        grad_out_shape = grad_out.shape
        grad_out = grad_out.reshape(a.shape)
        c = torch.ones_like(a) - a - b
        grad_wp = (a*grad_out).sum(dim=1, keepdim=True)
        grad_wn = (b*grad_out).sum(dim=1, keepdim=True)
        grad_in = (wp*a + wn*b* + 1.0*c) * grad_out
        return grad_in.reshape(grad_out_shape), grad_wp, grad_wn, None


## 
def non_uniform_scale(x, codec):
    BTxX = codec * x
    BTxX = BTxX.sum()
    BTXB = (codec * codec).sum()
    basis = BTxX / BTXB
    basis = torch.where(BTXB == 0, x.mean().to(torch.float32), basis)
    return basis

class Quant_Distribution_Loss(nn.Module):
    def __init__(self):
        super(Quant_Distribution_Loss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        m = input * target
        n = target * target
        k = m.sum() / n.sum()
        return (k - 1).abs()


