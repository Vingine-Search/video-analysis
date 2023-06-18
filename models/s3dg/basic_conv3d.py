# this implements a basic 3d convolutional network as a building block for
# for the s3dg network

import torch as th
import torch.nn as nn


class BasicConv3D(nn.Module):
    """
    Basic 3D convolutional layer with batch normalization and ReLU activation.

    Args:
        input_dim (int): number of input channels
        output_dim (int): number of output channels
        kernel_size (int): size of the convolutional kernel
        stride (int): stride of the convolution
        padding (int): padding of the convolution

    Note:
        S3D-G paper uses kernel size 3, stride 1, and padding 1 for all conv layers.
        It also uses the top-heavy and bottom-heavy 3D Conv layers meaning that it uses the
        basic 3D conv layer only for the first and the last layers of the network unlike the I3D Network
    """

    __annotations__ = {
        "conv": nn.Conv3d,
        "bn": nn.BatchNorm3d,
        "relu": nn.ReLU,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(BasicConv3D, self).__init__()
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=stride, padding=padding)

        # more can be found here: https://youtu.be/DtEq44FTPM4
        # batch norm: tries to normalize the output of the previous layer to have zero mean and unit variance
        # it's called batch since it normalized the values with respect to the batch of inputs
        # params:
        # eps: a small value added to the denominator to improve numerical stability
        # momentum: the value used for the running_mean and running_var computation
        # affine: a boolean value that when set to True, this module has learnable affine parameters (scale and bias)
        # equation is ((x - mean) / sqrt(var + eps)) * weight + bias
        # used to:
        # 1. speed up training
        # 2. decrease the sensitivity to the initial values of the weights
        # 3. regularize the model (a little bit)
        self.bn = nn.BatchNorm3d(output_dim, eps=0.001, momentum=0.001, affine=True)

        # ReLU: Rectified Linear Unit activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


