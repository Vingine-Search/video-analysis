# this script is used to define the totality of RPN (RPNCore(ANCORGENI+...) + RPNHead)

import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Optional

from .imagelist import ImageList
from .rpnhead import RPNHead
from .anchorgeni import AnchorGeni
from .rpncore import RPNCore

from ...config.config import reader

cfg = reader()

# read rpn config
pre_nms_topn = {
    "training": cfg["rpn"]["train"]["pre_nms_topn"],
    "testing": cfg["rpn"]["test"]["pre_nms_topn"]
}
post_nms_topn = {
    "training": cfg["rpn"]["train"]["post_nms_topn"],
    "testing": cfg["rpn"]["test"]["post_nms_topn"]
}
nms_thresh = cfg["rpn"]["nms_thresh"]
fg_thresh = cfg["rpn"]["fg_thresh"]
bg_thresh = cfg["rpn"]["bg_thresh"]
batch_size_perimage = cfg["rpn"]["batch_size_perimage"]
positive_fraction = cfg["rpn"]["positive_fraction"]


class RPN(nn.Module):
    """
    Full Region Proposal Network (RPN)
    """
    def __init__(self):
        super(RPN, self).__init__()
        # Anchor Box Generator
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchorgeni = AnchorGeni(anchor_sizes, aspect_ratios)
        # RPN Head
        rpn_head = RPNHead(256, anchorgeni.num_anchors_per_location()[0])
        # Create RPN
        self.rpn = RPNCore(
            anchorgeni, rpn_head,
            fg_thresh, bg_thresh, batch_size_perimage, positive_fraction,
            pre_nms_topn, post_nms_topn, nms_thresh
        )

    def forward(
        self, 
        images: ImageList, 
        fpn_feature_maps: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
        """
        Forward pass of RPN

        Args:
            images (tensor): Tensor of images.
            fpn_feature_maps (dict): Dict of tensors from the FPN.
            targets (list[dict]): List of ground-truth boxes for each image.

        Returns:
            boxes (list[tensor]): List of tensors of predicted boxes for each image.
            losses (dict): Dict of losses.
            fpn_feature_maps (dict): Dict of tensors from the FPN.
        """
        if targets:
            boxes, losses = self.rpn(images, fpn_feature_maps, targets)
        else:
            boxes, losses = self.rpn(images, fpn_feature_maps)
        return boxes, losses, fpn_feature_maps

