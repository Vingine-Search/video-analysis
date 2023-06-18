# this implements the extra self-gating layer in the s3dg network

import torch as th
import torch.nn as nn

class SelfGatting(nn.Module):
    """
    Self-Gatting layer that learns the weights for the input tensor.

    Args:
        input_dim (int): number of input channels

    Note:
        The weights are learned by a linear layer that takes the average of the input tensor
        and then applies a sigmoid activation function to the output.
    """

    __annotations__ = {
        "fc": nn.Linear,
        "segm": nn.Sigmoid,
    }

    def __init__(self, input_dim: int):
        super(SelfGatting, self).__init__()
        # fc is fully connected linear layer
        self.fc = nn.Linear(input_dim, input_dim)
        self.segm = nn.Sigmoid()

    def forward(self, x: th.Tensor) -> th.Tensor:
        spatio_temporal_avg = th.mean(x, dim=(2, 3, 4), keepdim=True)
        weights = self.fc(spatio_temporal_avg)
        weights = self.segm(weights)
        return weights[:, :, None, None, None] * x


