

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from .quant import conv3x3, conv1x1
from .layers import norm, actv
from torch.nn.modules.utils import _quadruple

# Prone-net: Point-wise and Reshape only Neural Network
class Prone(nn.Module):
    def __init__(self, out_channel, in_channel=3, stride=4, groups=1, kernel_size=3, force_fp=True, args=None, feature_stride=1, keepdim=True):
        super(Prone, self).__init__()
        self.stride = stride * 2
        self.in_channel = in_channel * self.stride * self.stride
        self.out_channel = out_channel * 4
        self.keepdim = keepdim
        self.conv = conv1x1(self.in_channel, self.out_channel, groups=groups, args=args, force_fp=force_fp)

        self.bn_before_restore = False
        if hasattr(args, 'keyword'):
            self.bn_before_restore = 'bn_before_restore' in args.keyword
        if self.bn_before_restore:
            self.bn = norm(self.out_channel, args)

    def forward(self, x):
        # padding zero when cannot just be divided
        B, C, H, W = x.shape
        if H % self.stride != 0:
            pad = (self.stride - (H % self.stride)) // 2
            x = F.pad(x, _quadruple(pad), mode="constant", value=0)

        B, C, H, W = x.shape
        #print(B, C, H, W, self.stride)
        # consider to employ the PixelShuffle layer instead
        x = x.reshape(B, C, H // self.stride, self.stride, W // self.stride, self.stride)
        x = x.transpose(4, 3).reshape(B, C, 1, H // self.stride, W // self.stride, self.stride * self.stride)
        x = x.transpose(2, 5).reshape(B, C * self.stride * self.stride, H // self.stride, W // self.stride)
        x = self.conv(x)
        if self.bn_before_restore:
            x = self.bn(x)
        
        if self.keepdim: # restore the channel dimension, however resolution might be altered
            B, C, H, W = x.shape
            x = x.reshape(B, C//4, 4, H, W, 1)
            x = x.transpose(2, 5).reshape(B, C//4, H, W, 2, 2)
            x = x.transpose(4, 3).reshape(B, C//4, H * 2, W * 2)
        return x

def qprone(in_channel, out_channel, stride=1, groups=1, padding=1, args=None, force_fp=False, feature_stride=1, kernel_size=3, keepdim=True):
    assert kernel_size in [3], "Only kernel size = 3 support"
    assert stride in [1, 2], "Stride must be 1 or 2"
    #assert groups in [1], "groups must be 1"
    return Prone(out_channel, in_channel, stride, groups, kernel_size, force_fp, args, feature_stride, keepdim)

