
import torch
import torch.nn as nn
from .quant import conv3x3

def seq_b_a_c(x, bn, relu, conv):
    x = bn(x)
    x = relu(x)
    x = conv(x)
    return x

def seq_c_b_a(x, bn, relu, conv):
    x = conv(x)
    x = bn(x)
    x = relu(x)
    return x

def seq_a_b_c(x, bn, relu, conv):
    x = relu(x)
    x = bn(x)
    x = conv(x)
    return x

class VGG_SMALL(nn.Module):
    def __init__(self, args):
        super(VGG_SMALL, self).__init__()

        qconv3x3 = conv3x3
        for i in range(6):
            setattr(self, 'relu%d' % (i+1), nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

        if 'bac' in args.keyword: # default in LQ-net paper
            self.seq = seq_b_a_c
        elif 'cba' in args.keyword:
            self.seq = seq_c_b_a
        elif 'abc' in args.keyword:
            self.seq = seq_a_b_c
        else:
            self.seq = None

        if 'bac' in args.keyword or 'abc' in args.keyword:
            self.relu1 = nn.Sequential()  # add to fix bug
            self.bn1 = nn.Sequential()
            self.bn3 = nn.BatchNorm2d(128)
            self.bn5 = nn.BatchNorm2d(256)
        elif 'cba' in args.keyword:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm2d(512)

        self.bn2 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.conv2 = qconv3x3(128, 128, args=args)
        self.conv3 = qconv3x3(128, 256, args=args)
        self.conv4 = qconv3x3(256, 256, args=args)
        self.conv5 = qconv3x3(256, 512, args=args)
        self.conv6 = qconv3x3(512, 512, args=args)

        self.classifier = nn.Linear(512 * 4 * 4, args.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.seq(x, self.bn1, self.relu1, self.conv1)
        x = self.seq(x, self.bn2, self.relu2, self.conv2)
        x = self.pool1(x)
        x = self.seq(x, self.bn3, self.relu3, self.conv3)
        x = self.seq(x, self.bn4, self.relu4, self.conv4)
        x = self.pool2(x)
        x = self.seq(x, self.bn5, self.relu5, self.conv5)
        x = self.seq(x, self.bn6, self.relu6, self.conv6)
        x = self.pool3(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_small(args):
    return VGG_SMALL(args)

