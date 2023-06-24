# this is an implementation of vgg16 network in pytorch
# since it can be used as a feature extraction network in faster-rcnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List

# M means maxpooling layer
# the number in the list means the number of filters in a convolutional layer
vgg16_layers_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):
    def __init__(self, layers_cfg:List[int]):
        super(VGG, self).__init__()
        self.features = self._make_layers(layers_cfg)
        self.classifier = nn.Linear(512, 10)

    def _make_layers(self, layers_cfg:List[int]) -> nn.Sequential:
        layers = []
        in_channels = 3
        for x in layers_cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                   nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                   nn.BatchNorm2d(x),
                   nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def vgg16() -> VGG:
    return VGG(vgg16_layers_cfg)
