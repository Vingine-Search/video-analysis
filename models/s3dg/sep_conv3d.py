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
        "conv_s": nn.Conv3d,
        "conv_t": nn.Conv3d,
        "bn_s": nn.BatchNorm3d,
        "bn_t": nn.BatchNorm3d,
        "relu_s": nn.ReLU,
        "relu_t": nn.ReLU,
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
        self.conv_s = nn.Conv3d(input_dim, output_dim, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(output_dim, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU(inplace=True)

        # temporal conv, batching and activation
        self.conv_t = nn.Conv3d(output_dim, output_dim, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(output_dim, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU(inplace=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # apply the spatial part
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        # apply the temporal part
        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


