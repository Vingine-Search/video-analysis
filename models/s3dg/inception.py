# this implements the core of the s3dg i.e the Inception block
# it combines other building blocks i.e BasicConv3D, SepConv3D and SelfGatting

import torch as th
import torch.nn as nn

from .basic_conv3d import BasicConv3D
from .sep_conv3d import SepConv3D
from .self_gating import SelfGatting

class InceptionBlock(nn.Module):
    """
    Inception block that consists of 4 branches:
        1. 1x1x1 conv
        2. 1x1x1 conv + sep conv (3x1x1 + 1x3x3)
        3. 1x1x1 conv + sep conv (3x1x1 + 1x3x3)
        4. 3x3x3 max pooling + 1x1x1 conv
    can have self gatting after each branch or not
    then the outputs of the branches are concatenated

    Args:
        input_dim (int): number of input channels
        num_outputs_b0_s0 (int): number of output channels for branch 0, stage 0 (1x1x1 conv)

        num_outputs_b1_s0 (int): number of output channels for branch 1, stage 0 (1x1x1 conv)
        num_outputs_b1_s1 (int): number of output channels for branch 1, stage 1 (3x1x1 + 1x3x3)

        num_outputs_b2_s0 (int): number of output channels for branch 2, stage 0 (1x1x1 conv)
        num_outputs_s2_s1 (int): number of output channels for branch 2, stage 1 (3x1x1 + 1x3x3)

        num_outputs_b3_s0 (int): number of output channels for branch 3, stage 0 (3x3x3 max pooling + 1x1x1 conv)
        gatting (bool): whether to use self-gatting or not
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs_b0_s0: int,
        num_outputs_b1_s0: int,
        num_outputs_b1_s1: int,
        num_outputs_b2_s0: int,
        num_outputs_s2_s1: int,
        num_outputs_b3_s0: int,
        gatting: bool = False,
    ):
        super(InceptionBlock, self).__init__()
        self.gatting = gatting
        self.branch0 = BasicConv3D(input_dim, num_outputs_b0_s0, 1, stride=1, padding=0)
        self.branch1 = nn.Sequential(
            BasicConv3D(input_dim, num_outputs_b1_s0, 1, stride=1, padding=0),
            SepConv3D(num_outputs_b1_s0, num_outputs_b1_s1, 3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3D(input_dim, num_outputs_b2_s0, 1, stride=1, padding=0),
            SepConv3D(num_outputs_b2_s0, num_outputs_s2_s1, 3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            BasicConv3D(input_dim, num_outputs_b3_s0, 1, stride=1, padding=0),
        )
        if gatting:
            self.gatting_branch0 = SelfGatting(num_outputs_b0_s0)
            self.gatting_branch1 = SelfGatting(num_outputs_b1_s1)
            self.gatting_branch2 = SelfGatting(num_outputs_s2_s1)
            self.gatting_branch3 = SelfGatting(num_outputs_b3_s0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        if self.gatting:
            b0 = self.gatting_branch0(b0)
            b1 = self.gatting_branch1(b1)
            b2 = self.gatting_branch2(b2)
            b3 = self.gatting_branch3(b3)
        return th.cat((b0, b1, b2, b3), 1)


