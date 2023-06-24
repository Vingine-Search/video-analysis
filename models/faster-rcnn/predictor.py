# this file is to define the predictor for the fast rcnn model
# which predicts the object class and bounding box regression

import torch as th
import torch.nn as nn

from torch import Tensor
from typing import Tuple

class Predictor(nn.Module):
    """
    Standard classification + bounding box regression layers for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels:int, num_classes:int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        # 4 for the 4 coordinates of the bounding box
        # num_classes * 4 because we have 4 coordinates for each class
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4) 

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() == 4:
            th._assert(list(x.shape[2:]) == [1, 1], f"x has the wrong shape")
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

