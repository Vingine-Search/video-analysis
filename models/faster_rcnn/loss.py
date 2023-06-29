# this provides some loss functions for faster-rcnn and other loss functions
# the relation detection network as well as mentiond by the
# "Large-scale Visual Relationship Understanding" paper

import torch as th
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional, Tuple


def fastrcnn_loss(
        class_logits: Tensor,
        box_regression:Tensor, 
        labels: List[Tensor],
        regression_targets: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor): [N, num_classes] classification logits for all anchors
        box_regression (Tensor): [N, num_classes * 4] bbox regression deltas for all anchors
        labels (list[Tensor]): ground-truth boxes present in the image (for each image)
        regression_targets (List[Tensor]): [N, num_classes * 4] regression targets for all anchors

    Returns:
        classification_loss (Tensor): scalar tensor containing the loss for classification
        box_loss (Tensor): scalar tensor containing the loss for bbox regression
    """
    thlabels = th.cat(labels, dim=0)
    thregression_targets = th.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, thlabels)
    # for the box loss, only consider positive anchors (not background)
    sampled_pos_inds_subset = th.nonzero(thlabels > 0).squeeze(1)
    labels_pos = thlabels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4) # (N, num_classes, 4)
    box_loss = F.smooth_l1_loss( # less sensitive to outliers
        box_regression[sampled_pos_inds_subset, labels_pos], # (num_pos, 4)
        thregression_targets[sampled_pos_inds_subset], # also (num_pos, 4)
        beta=(1/9),
        size_average=False,
    )
    box_loss = box_loss / thlabels.numel()
    return classification_loss, box_loss

def reldn_loss(
    prd_cls_scores: Tensor,
    prd_labels_int32: List[Tensor],
    fg_only:bool=False
):
    """
    computes the cross entropy loss for the relation detection network

    Args:
        prd_cls_scores (Tensor): [N, num_classes] classification logits for all anchors
        prd_labels_int32 (List[Tensor]): ground-truth boxes present in the image (for each image)
        fg_only (bool): whether to only consider foreground classes

    Returns:
        Tuple of:
        - **loss_cls_prd** (Tensor): scalar tensor containing the loss for classification
        - **accuracy_cls_prd** (Tensor): scalar tensor containing the accuracy for classification
    """
    device = prd_cls_scores.device
    prd_labels = th.cat(prd_labels_int32, 0).to(device) # (N, )
    loss_cls_prd = F.cross_entropy(prd_cls_scores, prd_labels)
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)
    return loss_cls_prd, accuracy_cls_prd
