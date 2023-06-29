# this is an implementation of the region proposal network head (RPN) in Faster R-CNN
# which is used to predict the classification scores and the bounding box regression deltas
# for the anchors at each spatial position

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class RPNHead(nn.Module):
    """
    This class is the head of the RPN. It takes the output of the backbone network and
    produces the logits and the bounding box regression deltas.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    
    __annotations__ = {
        "conv": nn.Conv2d,
        "cls_logits": nn.Conv2d,
        "bbox_pred": nn.Conv2d,
    }

    def __init__(self, in_channels: int, num_anchors: int):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # cls_logits is conv layer to predict the classification scores for each anchor box
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)

        # bbox_pred is conv layer to predict the bounding box regression deltas for each anchor box
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
    
    def forward(self, x: List[th.Tensor]) -> Tuple[List[th.Tensor], List[th.Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


