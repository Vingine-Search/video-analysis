# this implements a separable 3d convolutional network as a building block for
# for the s3dg network

import torch as th
import torch.nn as nn


class SepConv3D(nn.Module):
    """
    Separable 3D convolutional layer with batch normalization and ReLU activation.
    It's Separable in the sense that all other layers of the network excpet for the first
    and the last are 2D conv for the spatial dimensions followed by a 1D conv for the temporal dimension.

    Args:
        input_dim (int): number of input channels
        output_dim (int): number of output channels
        kernel_size (int): size of the convolutional kernel
        stride (int): stride of the convolution
        padding (int): padding of the convolution
    """

    __annotations__ = {
        "spatial_conv": nn.Conv3d,
        "temporal_conv": nn.Conv3d,
        "bn_spatial": nn.BatchNorm3d,
        "bn_temporal": nn.BatchNorm3d,
        "relu_spatial": nn.ReLU,
        "relu_temporal": nn.ReLU,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(SepConv3D, self).__init__()
        # spatial conv, batching and activation
        self.spatial_conv = nn.Conv3d(
            input_dim, input_dim, (kernel_size, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0)
        )
        self.bn_spatial = nn.BatchNorm3d(input_dim, eps=0.001, momentum=0.001, affine=True)
        self.relu_spatial = nn.ReLU(inplace=True)

        # temporal conv, batching and activation
        self.temporal_conv = nn.Conv3d(
            input_dim, output_dim, (1, 1, kernel_size), stride=(1, 1, stride), padding=(0, 0, padding)
        )
        self.bn_temporal = nn.BatchNorm3d(output_dim, eps=0.001, momentum=0.001, affine=True)
        self.relu_temporal = nn.ReLU(inplace=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # apply the spatial part
        x = self.spatial_conv(x)
        x = self.bn_spatial(x)
        x = self.relu_spatial(x)

        # apply the temporal part
        x = self.temporal_conv(x)
        x = self.bn_temporal(x)
        x = self.relu_temporal(x)
        return x


