# implementation of residual network (resnet50) since it could be very much
# used as the backbone of the faster-rcnn model

# In theory, we expect having a deeper network should only help but in reality,
# the deeper network has higher training error, and thus test error. And that is 
# the problem that resnet is trying to solve.

# The approach is to add a shortcut or a skip connection that allows information to flow,
# well just say, more easily from one layer to the next’s next layer, i.e.,
# you bypass data along with normal CNN flow from one layer to the next layer after the immediate next.

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List

class ResNetBlock(nn.Module):
    """
    Residual block for resnet50 (expansion = 4)

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        stride (int): stride of the convolutional layer (default: 1)
    """
    def __init__(self, in_channels:int, out_channels:int, stride:int=1):
        super(ResNetBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels*self.expansion:
            # if the input and output channels are not the same due to stride or the other condition
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )

    def forward(self, x:Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class ResNetArch(nn.Module):
    def __init__(self, layers:List[int], img_channels:int=3, num_classes:int=10):
        super(ResNetArch, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        # ResNet50 has 3 layers, each layer has 3, 4, 6, 3 blocks respectively
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        # average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # fully connected layer
        self.fc = nn.Linear(512*4, num_classes)

    def _make_layer(self, out_channels:int, num_blocks:int, stride:int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

def resnet50() -> ResNetArch:
    return ResNetArch([3,4,6,3])


def resnet101() -> ResNetArch:
    return ResNetArch([3,4,23,3])

