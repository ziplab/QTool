
import torch

##
class XnorActivation(torch.autograd.Function):
    '''
    Binarize the activations and calculate the mean across channel / resolution dimension.
    '''
    @staticmethod
    def forward(self, x, reduce_type='channel', grad_type='None'):
        b, c, h, w = x.shape
        if reduce_type == 'resolution':
            E = x.detach().abs().reshape(b, c, -1).mean(2, keepdim=True).reshape(b, c, 1, 1)
        elif reduce_type == 'channel':
            E = x.detach().abs().mean(1, keepdim=True)
        else:
            E = 1 # avoid runtime cost to calculate mean
        y = torch.ones_like(x)
        y.masked_fill_(x < 0, -1)
        y.mul_(E)

        if grad_type == 'STE':
            grad_mask = torch.ones_like(x)
            grad_mask.masked_fill_(x.ge(1), 0)
            grad_mask.masked_fill_(x.le(-1), 0)
        elif grad_type == "Triangle":
            grad_mask = 2 - 2 * x.clone().abs()
            grad_mask.masked_fill_(x.ge(1), 0)
            grad_mask.masked_fill_(x.le(-1), 0)
        else:
            grad_mask = torch.ones_like(x)
        self.save_for_backward(grad_mask)
        return y

    @staticmethod
    def backward(self, grad_output):
        grad_mask, = self.saved_tensors
        return grad_output * grad_mask, None, None

## 
class XnorWeight(torch.autograd.Function):
    '''
    Binarize the weight
    '''
    @staticmethod
    def forward(self, x, quant_group=1, grad_type='None'):
        E = x.detach().abs().reshape(quant_group, -1).mean(1, keepdim=True).reshape(quant_group, 1, 1, 1)
        y = torch.ones_like(x)
        y.masked_fill_(x < 0, -1)  # use compare rather than sign() to handle the zero
        y.mul_(E)

        if grad_type == 'STE':
            grad_mask = torch.ones_like(x)
            grad_mask.masked_fill_(x.ge(1), 0)
            grad_mask.masked_fill_(x.le(-1), 0)
        #elif grad_type == "Triangle":
        #    grad_mask = 2 - 2 * x.clone().abs()
        #    grad_mask.masked_fill_(x.ge(1), 0)
        #    grad_mask.masked_fill_(x.le(-1), 0)
        else:
            grad_mask = torch.ones_like(x)
        self.save_for_backward(grad_mask)
        return y

    @staticmethod
    def backward(self, grad_output):
        grad_mask, = self.saved_tensors
        return grad_output * grad_mask, None, None

