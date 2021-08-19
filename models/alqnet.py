import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

plugin_enable=False
try:
    import subprocess
    import plugin
    print("Import plugin successfully")
    plugin_enable=True
except (ImportError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError, PermissionError) as e:
    print("Failing to import plugin, %r" % e)
    plugin_enable=False

# add on date 2019.12.26
__EPS__ = 1e-5

## LQ-net
class LqNet_fm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, basis, codec_vector, codec_index, thrs_multiplier, training=True, half_range=False, auxil=None, adaptive='none'):
        num_levels = codec_vector.shape[0]
        bit = codec_vector.shape[1]
        quant_group = basis.shape[1]

        # calculate levels and sort
        levels = torch.matmul(codec_vector, basis)  # [num_levels * bit] * [bit * quant_group]
        levels, sort_id = torch.sort(levels, 0, descending=False)

        # calculate threshold
        thrs = torch.matmul(thrs_multiplier, levels) # [(num_levels - 1) * num_levels] * [num_levels * quant_group]

        # pre-processing of the inputs, according to adaptive
        if adaptive == 'mean':
            mean = inputs.mean([1,2,3], keepdim=True) + __EPS__
            inputs = inputs / mean
        if adaptive == 'var':
            std = inputs.std([1,2,3], keepdim=True) + __EPS__
            inputs = inputs / std
        if adaptive == 'min':
            lower = inputs.min()
            inputs = inputs - lower

        # feature map: b * c * h * w   | b * c
        if quant_group != 1:
            x = inputs.transpose(1, 0)  # bchw --> cbhw | bc --> cb
        else:
            x = inputs # keep origin shape

        # calculate output y and its binary codec
        x_shape = x.shape
        x = x.reshape(quant_group, -1)
        y = levels[0].unsqueeze(1).expand_as(x)  # output
        codec = codec_index[sort_id[0]].unsqueeze(1).expand_as(x)
        for i in range(num_levels - 1):
            g = x > thrs[i].unsqueeze(1).expand_as(x)
            y = torch.where(g, levels[i + 1].unsqueeze(1).expand_as(x), y)
            codec = torch.where(g, codec_index[sort_id[i+1]].unsqueeze(1).expand_as(codec), codec)

        # y is ready here, means forward has been finished
        y = y.reshape(x_shape)
        if quant_group != 1:
            y = y.transpose(1,0) # cbhw --> bchw
        if adaptive == 'mean':
            y = y * mean
        if adaptive == 'var':
            y = y * std
        if adaptive == 'min':
            y = y + lower
        if not training:
            return y, basis

        # contine to compute the gradident of basis to avoid saving buffer to backward
        code = codec.new_ones(bit, x.shape[0], x.shape[1], dtype=torch.int8)
        for i in range(bit):
            code[i] = codec / (2**i) - codec / (2**(i+1)) * 2
            if not half_range:
                code[i] = code[i] * 2 - 1
        codec = None

        # calculate BTxX
        BTxX = inputs.new_zeros(bit, quant_group, 1)
        for i in range(bit):
            BTxXi0 = code[i].float() * x
            BTxXi0 = BTxXi0.sum(dim=1, keepdim=True)
            BTxX[i] = BTxXi0
        BTxX = BTxX.reshape(bit, quant_group)
        x = None

        # BTxB
        BTxB = inputs.new_zeros(bit*bit, quant_group, 1)
        for i in range(bit):
            for j in range(i+1):
                value = (code[i] * code[j]).float().sum(dim=1, keepdim=True)
                if i == j:
                    value = torch.where(value == 0, value.new_ones(value.shape) * 0.00001, value)
                else:
                    BTxB[j*bit + i] = value
                BTxB[i*bit + j] = value
        BTxB = BTxB.reshape(bit*bit, quant_group).reshape(bit, bit, quant_group).float()

        # inverse
        BTxB_transpose = BTxB.transpose(0, 2).transpose(1, 2)
        try:
            BTxB_inv = torch.inverse(BTxB_transpose)
        except RuntimeError:
            logging.info("LqNet_fm matrix has not inverse %r" % BTxB_transpose)
            raise RuntimeError("LqNet_fm matrix has no inverse for weight %r" % BTxB_transpose)
        BTxB_inv = BTxB_inv.transpose(1, 2).transpose(0, 2)

        new_basis = BTxB_inv * BTxX.expand_as(BTxB_inv)
        new_basis = new_basis.sum(dim=1, keepdim=True)
        new_basis = new_basis.squeeze(1)
        auxil.data = new_basis
        basis = 0.9 * basis + 0.1 * new_basis
        ctx.save_for_backward(inputs, levels[num_levels - 1])
        return y, basis

    @staticmethod
    def backward(ctx, grad_output, grad_basis):
        inputs, clip = ctx.saved_tensors
        quant_group = clip.size(0)
        if quant_group != 1:
            x = inputs.transpose(1,0)
        else:
            x = inputs
        x_shape = x.shape
        x = x.reshape(quant_group, -1)
        clip = clip.unsqueeze(1).expand_as(x)
        x = x >= clip
        x = x.reshape(x_shape)
        if quant_group != 1:
          x = x.transpose(1,0)
        #x = x.reshape(grad_output.shape)
        grad_input = grad_output.clone()
        grad_input.masked_fill_(x, 0)
        return grad_input, None, None, None, None, None, None, None, None

## LQ-net
class LqNet_wt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, basis, codec_vector, codec_index, thrs_multiplier, training=True, half_range=False, auxil=None, adaptive='none'):
        num_levels = codec_vector.shape[0]
        bit = codec_vector.shape[1]
        quant_group = basis.shape[1]

        # calculate levels and sort
        levels = torch.matmul(codec_vector, basis)
        levels, sort_id = torch.sort(levels, 0, descending=False)

        # calculate threshold
        thrs = torch.matmul(thrs_multiplier, levels)

        # calculate output y and its binary codec
        origin_shape = inputs.shape
        x = inputs.reshape(quant_group, -1)

        # pre-processing of the inputs, according to adaptive
        if adaptive == 'mean':
            mean = x.mean(1, keepdim=True)
            x = x - mean
        if adaptive == 'var':
            std = x.std(1, keepdim=True) + __EPS__
            x = x / std
        if adaptive == 'mean-var':
            mean = x.mean(1, keepdim=True)
            std = x.std(1, keepdim=True) + __EPS__
            x = (x - mean) / std

        y = levels[0].unsqueeze(1).expand_as(x)  # output
        codec = codec_index[sort_id[0]].unsqueeze(1).expand_as(x)
        for i in range(num_levels - 1):
            g = x > thrs[i].unsqueeze(1).expand_as(x)
            y = torch.where(g, levels[i + 1].unsqueeze(1).expand_as(x), y)
            codec = torch.where(g, codec_index[sort_id[i+1]].unsqueeze(1).expand_as(codec), codec)

        if adaptive == 'mean':
            y = y + mean
            mean = None
        if adaptive == 'var':
            y = y * std
            std = None
        if adaptive == 'mean-var':
            y = y * std + mean
            std = None
            mean = None

        y = y.reshape(origin_shape)
        if not training:
            return y, basis

        # contine to compute the gradident of basis to avoid saving buffer to backward
        code = codec.new_ones(bit, x.shape[0], x.shape[1], dtype=torch.int8)
        for i in range(bit):
            code[i] = codec / (2**i) - codec / (2**(i+1)) * 2
            if not half_range:
                code[i] = code[i] * 2 - 1
        codec = None

        # calculate BTxX
        BTxX = x.new_zeros(bit, quant_group, 1)
        for i in range(bit):
            BTxXi0 = code[i].float() * x
            BTxXi0 = BTxXi0.sum(dim=1, keepdim=True)
            BTxX[i] = BTxXi0
        BTxX = BTxX.reshape(bit, quant_group)
        BTxX = BTxX + (0.0001 * basis)
        x = None

        # BTxB
        BTxB = inputs.new_zeros(bit*bit, quant_group, 1)
        for i in range(bit):
            for j in range(i+1):
                value = (code[i] * code[j]).float().sum(dim=1, keepdim=True)
                if i == j:
                    value = (value + 0.0001) * 1.000001
                else:
                    BTxB[j*bit + i] = value 
                BTxB[i*bit + j] = value 
        BTxB = BTxB.reshape(bit*bit, quant_group).reshape(bit, bit, quant_group).float()

        # inverse
        BTxB_transpose = BTxB.transpose(0, 2).transpose(1, 2)
        try:
            BTxB_inv = torch.inverse(BTxB_transpose)
        except RuntimeError:
            logging.info("LqNet_wt matrix has not inverse %r" % BTxB_transpose)
            raise RuntimeError("LqNet_wt matrix has no inverse for weight %r" % BTxB_transpose)
        BTxB_inv = BTxB_inv.transpose(1, 2).transpose(0, 2)

        new_basis = BTxB_inv * BTxX.expand_as(BTxB_inv)
        new_basis = new_basis.sum(dim=1, keepdim=True)
        new_basis = new_basis.squeeze(1)
        auxil.data = new_basis
        basis = 0.9 * basis + 0.1 * new_basis
        return y, basis

    @staticmethod
    def backward(ctx, grad_output, grad_basis):
        return grad_output, None, None, None, None, None, None, None, None

