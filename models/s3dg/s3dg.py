# this is an implementation of the S3DG video summarization algorithm

import torch as th
import torch.nn.functional as F
import torch.nn as nn

from .basic_conv3d import BasicConv3D
from .sep_conv3d import SepConv3D
from .self_gating import SelfGatting
from .inception import InceptionBlock

class S3DG(nn.Module):
    """
    S3D model from the paper: https://arxiv.org/abs/1712.04851

    Args:
        num_class (int): number of classes
        gatting (bool): whether to use self-gatting or not
    """

    def __init__(
        self,
        num_classes: int,
        gatting: bool = True,
    ):
        super(S3DG, self).__init__()

        self.num_class = num_classes
        self.gatting = gatting

        self.base = nn.Sequential(
            # part 1
            SepConv3D(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # part 2
            BasicConv3D(64, 64, kernel_size=1, stride=1),
            SepConv3D(64, 192, kernel_size=3, stride=1, padding=1),
            SelfGatting(192) if gatting else nn.Identity(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # part 3
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            # part 4
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            # part 5
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),
        )
        self.fc = nn.Sequential(
            nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), y.size(2))
        logits = th.mean(y, 2)
        return logits
