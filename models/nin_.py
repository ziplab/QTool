
import torch.nn as nn
import torch
import torch.nn.functional as F

from .quant import custom_conv
from .layers import norm, actv

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.body = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                norm(192, args),
                actv(args),
                custom_conv(192, 160, kernel_size=1, stride=1, padding=0, args=args,),
                norm(160, args),
                actv(args),
                custom_conv(160,  96, kernel_size=1, stride=1, padding=0, args=args),
                norm(96, args),
                actv(args),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                custom_conv( 96, 192, kernel_size=5, stride=1, padding=2, args=args),
                norm(192, args),
                actv(args),
                custom_conv(192, 192, kernel_size=1, stride=1, padding=0, args=args),
                norm(192, args),
                actv(args),
                custom_conv(192, 192, kernel_size=1, stride=1, padding=0, args=args),
                norm(192, args),
                actv(args),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                custom_conv(192, 192, kernel_size=3, stride=1, padding=1, args=args),
                norm(192, args),
                actv(args),
                custom_conv(192, 192, kernel_size=1, stride=1, padding=0, args=args),
                norm(192, args),
                actv(args),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                norm(10, args),
                actv(args),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), self.args.num_classes)
        return x

def nin(args=None):
    model = Net(args)
    return model

