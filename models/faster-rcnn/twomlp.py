# this is an implementation of a two multi-layer perceptron head for the FPN-based models

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class TwoMLP(nn.Module):
    """
    Two Multi-Layer Perceptron head. This head is used to prepare the features
    for the classification and regression heads.

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """
    __annotations__ = {
        'fc6': nn.Linear,
        'fc7': nn.Linear,
    }

    def __init__(self, in_channels:int, representation_size:int):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x:Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

