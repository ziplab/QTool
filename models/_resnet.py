import torch
import torch.nn as nn
import logging
import numpy as np

from .quant import conv3x3, conv1x1, conv0x0
from .layers import norm, actv, concat, flatten

'''
BasicBlock:
    different variants on architectures are supported (mainly controled by the order string
'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, args=None, feature_stride=1):
        super(BasicBlock, self).__init__()
        self.args = args

        for i in range(2):
            setattr(self, 'relu%d' % (i+1), nn.ModuleList([actv(args) for j in range(args.base)]))
        if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
            self.fix_relu = actv(args)
            setattr(self, 'relu2', nn.ModuleList([nn.Sequential() for j in range(args.base)]))

        self.bn1 = nn.ModuleList([norm(planes, args) for j in range(args.base)])
        self.bn2 = nn.ModuleList([norm(planes, args) for j in range(args.base)])

        # downsample branch
        self.enable_skip = stride != 1 or inplanes != planes or 'identify_norm' in args.keyword
        real_skip = 'real_skip' in args.keyword
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(stride))
        else:
            downsample.append(nn.Sequential())
        if inplanes != planes:
            if 'bacs' in args.keyword:
                downsample.append(norm(inplanes, args, feature_stride=feature_stride))
                downsample.append(actv(args))
                downsample.append(qconv1x1(inplanes, planes, stride=1, args=args, force_fp=real_skip))
                if 'fix' in args.keyword:
                    downsample.append(norm(planes, args, feature_stride=feature_stride*stride))
            elif 'bcas' in args.keyword:
                downsample.append(norm(inplanes, args, feature_stride=feature_stride))
                downsample.append(qconv1x1(inplanes, planes, stride=1, args=args, force_fp=real_skip))
                downsample.append(actv(args))
                if 'fix' in args.keyword: # remove the ReLU in skip connection
                    downsample.append(norm(planes, args))
            else:
                downsample.append(qconv1x1(inplanes, planes, args=args, force_fp=real_skip))
                downsample.append(norm(planes, args, feature_stride=feature_stride*stride))
                if 'fix' not in args.keyword:
                    downsample.append(actv(args))
        elif 'identify_norm' in args.keyword:
            downsample.append(norm(inplanes, args))
        if 'singleconv' in args.keyword: # pytorch official branch employ single convolution layer
            for i, n in enumerate(downsample):
                if isinstance(n, nn.AvgPool2d):
                    downsample[i] = nn.Sequential()
                if isinstance(n, nn.Conv2d) and inplanes != planes:
                    downsample[i] = qconv1x1(inplanes, planes, stride=stride, padding=extra_padding, args=args, force_fp=real_skip)
        if 'DCHR' in args.keyword: # double channel and halve resolution
            if inplanes != planes:
                downsample = []
                number = planes // inplanes
                if stride != 1:
                    downsample.append(concat(nn.ModuleList([nn.AvgPool2d(stride) for i in range(number)])))
                else:
                    downsample.append(concat(nn.ModuleList([nn.Sequential() for i in range(number)])))
        self.skip = nn.Sequential(*downsample)

        self.conv1 = nn.ModuleList([fconv3x3(inplanes, planes, stride=stride, groups=1, padding=extra_padding+1, args=args) for j in range(args.base)])
        self.conv2 = nn.ModuleList([sconv3x3(planes, planes, stride=1, groups=1, args=args) for j in range(args.base)])

        # scales
        if args.base == 1:
            self.scales = [1]
        else:
            self.scales = nn.ParameterList([nn.Parameter(torch.ones(1) / args.base, requires_grad=True) for i in range(args.base)])

        # Fixup initialization (https://arxiv.org/abs/1901.09321)
        if 'fixup' in args.keyword:
            self.bn1 = nn.ModuleList([nn.Sequential()])
            self.bn2 = nn.ModuleList([nn.Sequential()])
            for i, n in enumerate(self.skip):
                if isinstance(n, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.skip[i] = nn.Sequential()

            self.fixup_scale = nn.Parameter(torch.ones(1))
            if 'bias' in args.keyword:
                self.fixup_bias1a = nn.Parameter(torch.zeros(1))
                self.fixup_bias1b = nn.Parameter(torch.zeros(1))
                self.fixup_bias2a = nn.Parameter(torch.zeros(1))
                self.fixup_bias2b = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        if self.shrink is not None:
            x = self.shrink(x)

            if self.skip_block:
                return x

        if not self.enable_skip:
            residual = x

        if 'fixup' in self.args.keyword:
            if 'bias' in self.args.keyword:
                x = x + self.fixup_bias1a

        if self.enable_skip:
            residual = self.skip(x)

        result = None
        for conv1, conv2, bn1, bn2, relu1, relu2, scale in zip(self.conv1, self.conv2, \
                self.bn1, self.bn2, self.relu1, self.relu2, self.scales):
            if 'fixup' in self.args.keyword and 'bias' in self.args.keyword:
                out = self.seq(x, conv1, relu1, bn1, self.fixup_bias1b, True) + self.fixup_bias2a
            else:
                out = self.seq(x, conv1, relu1, bn1, residual, self.addition_skip)
            output = self.seq(out, conv2, relu2, bn2, out, self.addition_skip)
            if result is None:
                result = scale * output
            else:
                result = result + scale * output
        output = result

        if 'fixup' in self.args.keyword:
            output = output * self.fixup_scale
            if 'bias' in self.args.keyword:
                output = output + self.fixup_bias2b

        if not self.addition_skip:
            if 'fix' in self.args.keyword and ('bacs' in self.args.keyword or 'bcas' in self.args.keyword):
                output = self.fix_bn(output)
            output = output + residual
            if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
                output = self.fix_relu(output)

        return output

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, args=None, feature_stride=1):
        super(BottleNeck, self).__init__()
        self.args = args

        # Bi-Real structure or original structure
        if 'origin' in args.keyword:
            self.addition_skip = False
        else:
            self.addition_skip = True

        if self.addition_skip and args.verbose:
                logging.info("warning: add addition skip, not the origin resnet")

        for i in range(3):
            setattr(self, 'relu%d' % (i+1), nn.ModuleList([actv(args) for j in range(args.base)]))
        if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
            setattr(self, 'relu3', nn.ModuleList([nn.Sequential() for j in range(args.base)]))
            self.fix_relu = actv(args)

        if 'cbas' in args.keyword:
            self.seq = seq_c_b_a_s
        elif 'cbsa' in args.keyword: # default Pytorch
            self.seq = seq_c_b_s_a
        elif 'cabs' in args.keyword: # group-net
            self.seq = seq_c_a_b_s
        elif 'bacs' in args.keyword:
            self.seq = seq_b_a_c_s
        elif 'bcas' in args.keyword:
            self.seq = seq_b_c_a_s
        else:
            self.seq = None

        if 'bacs' in args.keyword:
            self.bn1 = nn.ModuleList([norm(inplanes, args) for j in range(args.base)])
            self.bn3 = nn.ModuleList([norm(planes, args) for j in range(args.base)])
            if 'fix' in self.args.keyword:
                self.fix_bn = norm(planes * self.expansion, args)
        else:
            self.bn1 = nn.ModuleList([norm(planes, args) for j in range(args.base)])
            self.bn3 = nn.ModuleList([norm(planes * self.expansion, args) for j in range(args.base)])
        self.bn2 = nn.ModuleList([norm(planes, args) for j in range(args.base)])

        keepdim = True
        qconv3x3 = conv3x3
        qconv1x1 = conv1x1
        extra_padding = 0
        # Prone network on
        if 'prone' in args.keyword:
            bn_before_restore = 'bn_before_restore' in args.keyword
            qconv3x3 = qprone

            if 'no_prone_downsample' in args.keyword and stride != 1 and keepdim:
                qconv3x3 = conv3x3

            if 'preBN' in args.keyword:
                raise NotImplementedError("preBN not supported for the Prone yet")

            if 'bn_before_restore' in args.keyword:
                if qconv3x3 == qprone:
                    self.bn2 = nn.ModuleList([nn.Sequential() for j in range(args.base)])

            if stride != 1 and (args.input_size // feature_stride) % (2*stride) != 0:
                extra_padding = ((2*stride) - ((args.input_size // feature_stride) % (2*stride))) // 2
                logging.warning("extra pad for Prone is added to be {}".format(extra_padding))
        # Prone network off

        # downsample branch
        self.enable_skip = stride != 1 or inplanes != planes * self.expansion or 'identify_norm' in args.keyword
        real_skip = 'real_skip' in args.keyword
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(stride))
        else:
            downsample.append(nn.Sequential())
        if inplanes != planes * self.expansion:
            if 'bacs' in args.keyword:
                downsample.append(norm(inplanes, args))
                downsample.append(actv(args))
                downsample.append(qconv1x1(inplanes, planes * self.expansion, stride=1, args=args, force_fp=real_skip))
                if 'fix' in args.keyword:
                    downsample.append(norm(planes * self.expansion, args))
            else:
                downsample.append(qconv1x1(inplanes, planes * self.expansion, stride=1, args=args, force_fp=real_skip))
                downsample.append(norm(planes * self.expansion, args))
                if 'fix' not in args.keyword:
                    downsample.append(actv(args))
        elif 'identify_norm' in args.keyword:
            downsample.append(norm(inplanes, args))
        if 'singleconv' in args.keyword:
            for i, n in enumerate(downsample):
                if isinstance(n, nn.AvgPool2d):
                    downsample[i] = nn.Sequential()
                if isinstance(n, nn.Conv2d) and inplanes != planes * self.expansion:
                    downsample[i] = qconv1x1(inplanes, planes * self.expansion, stride=stride, padding=extra_padding, args=args, force_fp=real_skip)
        if 'DCHR' in args.keyword: # double channel and halve resolution
            if inplanes != planes:
                downsample = []
                number = planes // inplanes
                if stride != 1:
                    downsample.append(concat(nn.ModuleList([nn.AvgPool2d(stride) for i in range(number)])))
                else:
                    downsample.append(concat(nn.ModuleList([nn.Sequential() for i in range(number)])))
        self.skip = nn.Sequential(*downsample)

        self.conv1 = nn.ModuleList([qconv1x1(inplanes, planes, stride=1, args=args) for j in range(args.base)])
        self.conv2 = nn.ModuleList([qconv3x3(planes, planes, stride=stride, groups=1, padding=extra_padding+1, args=args) for j in range(args.base)])
        self.conv3 = nn.ModuleList([qconv1x1(planes, planes * self.expansion, stride=1, args=args) for j in range(args.base)])

        if args.base == 1:
            self.scales = [1]
        else:
            self.scales = nn.ParameterList([nn.Parameter(torch.ones(1) / args.base, requires_grad=True) for i in range(args.base)])

        if 'fixup' in args.keyword:
            assert args.base == 1, 'Base should be 1 in Fixup'
            self.bn1 = nn.ModuleList([nn.Sequential()])
            self.bn2 = nn.ModuleList([nn.Sequential()])
            self.bn3 = nn.ModuleList([nn.Sequential()])
            for i, n in enumerate(self.skip):
                if isinstance(n, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.skip[i] = nn.Sequential()

            self.fixup_scale = nn.Parameter(torch.ones(1))
            if 'bias' in args.keyword:
                self.fixup_bias1a = nn.Parameter(torch.zeros(1))
                self.fixup_bias1b = nn.Parameter(torch.zeros(1))
                self.fixup_bias2a = nn.Parameter(torch.zeros(1))
                self.fixup_bias2b = nn.Parameter(torch.zeros(1))
                self.fixup_bias3a = nn.Parameter(torch.zeros(1))
                self.fixup_bias3b = nn.Parameter(torch.zeros(1))
            else:
                pass

    def forward(self, x):
        if not self.enable_skip:
            residual = x

        if 'fixup' in self.args.keyword:
            if 'bias' in self.args.keyword:
                x = x + self.fixup_bias1a

        if self.enable_skip:
            residual = self.skip(x)

        result = None
        for conv1, conv2, conv3, bn1, bn2, bn3, relu1, relu2, relu3, scale in zip(self.conv1, self.conv2, self.conv3, \
                    self.bn1, self.bn2, self.bn3, self.relu1, self.relu2, self.relu3, self.scales):
            if 'fixup' in self.args.keyword and 'bias' in self.args.keyword:
                out = self.seq(x, conv1, relu1, bn1, self.fixup_bias1b, True) + self.fixup_bias2a
                out = self.seq(out, conv2, relu2, bn2, self.fixup_bias2b, True) + self.fixup_bias3a
            else:
                out = self.seq(x, conv1, relu1, bn1, residual, self.addition_skip)
                out = self.seq(out, conv2, relu2, bn2, out, self.addition_skip)
            output = self.seq(out, conv3, relu3, bn3, out, self.addition_skip)
            if result is None:
                result = scale * output
            else:
                result = result + scale * output
        output = result

        if 'fixup' in self.args.keyword:
            output = output * self.fixup_scale
            if 'bias' in self.args.keyword:
                output = output + self.fixup_bias3b

        if not self.addition_skip:
            if 'fix' in self.args.keyword and ('bacs' in self.args.keyword or 'bcas' in self.args.keyword):
                output = self.fix_bn(output)
            output += residual
            if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
                output = self.fix_relu(output)

        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, args):
        super(ResNet, self).__init__()
        self.args = args
        assert args is not None, "args is None"
        self.layer_count = len(layers)
        self.inplanes = 64
        self.width_alpha = getattr(args, 'width_alpha', 1.0)
        self.inplanes = int(self.inplanes * self.width_alpha)
        self.input_channel = self.inplanes
        self.feature_stride = 1

        self.layers = []
        if 'cifar10' in args.keyword or 'cifar100' in args.keyword:
            self.layers.append(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False))
            self.layers.append(norm(self.input_channel, args))
            self.layers.append(actv(args))
        else:
            self.layers.append(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False))
            self.layers.append(norm(self.input_channel, args))
            self.layers.append(actv(args))
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        outplanes = self.inplanes
        for i in range(self.layer_count):
            index = i + 1
            channel_scale = 2 ** i
            outplanes = self.input_channel * channel_scale
            stride = 1 if i == 0 else 2
            if layers[i] == 0:
                continue
            setattr(self, 'layer%d' % index, self._make_layer(block, outplanes, layers[i], stride=stride, feature_stride=self.feature_stride))
            self.feature_stride = self.feature_stride * stride

        self.layers.append(nn.AdaptiveAvgPool2d((1,1)))
        self.layers.append(flatten())
        self.layers.append(nn.Linear(outplanes * block.expansion, args.num_classes))
        self.network = nn.Sequential(*self.layers)

    def _make_layer(self, block, planes, blocks, stride=1, feature_stride=1):
        strides = [stride] + [1]*(blocks-1)
        for stride in strides:
            self.layers.append(block(self.inplanes, planes, stride, self.args, feature_stride))
            feature_stride = feature_stride * stride
            self.inplanes = planes * block.expansion

    def forward(self, x):
        x = self.network(x)
        return x


def resnet18(args):
    model = ResNet(BasicBlock, [2, 2, 2, 2], args)
    return model

def resnet20(args):
    model = ResNet(BasicBlock, [3, 3, 3], args)
    return model

def resnet20_(args):
    model = ResNet(BasicBlock, [2, 2, 3, 2], args)
    return model

def resnet32(args):
    model = ResNet(BasicBlock, [5, 5, 5], args)
    return model

def resnet34(args):
    model = ResNet(BasicBlock, [3, 4, 6, 3], args)
    return model

def resnet34_(args):
    model = ResNet(BasicBlock, [0, 2, 2, 2, 2, 2, 2, 2, 2], args)
    return model

def resnet44(args):
    model = ResNet(BasicBlock, [7, 7, 7], args)
    return model

def resnet50(args):
    model = ResNet(BottleNeck, [3, 4, 6, 3], args)
    return model

def resnet56(args):
    model = ResNet(BasicBlock, [9, 9, 9], args)
    return model

def resnet101(args):
    model = ResNet(BottleNeck, [3, 4, 23, 3], args)
    return model



