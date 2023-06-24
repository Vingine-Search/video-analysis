# helper functions for the faster-rcnn model

import numpy as np
import torch as th

from torch import Tensor
from typing import Tuple, List

from config import reader

cfg = reader()
cfg_device = cfg["device"]

def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    """
    This function is used to make the output of the model be in the same format as the labels.

    Args:
        layer (tensor): Tensor of shape (N, AxHxW, C) where A is the number of anchors.
        N (int): Batch size.
        A (int): Number of anchors.
        C (int): Number of classes.
        H (int): Height of the feature map.
        W (int): Width of the feature map.

    Returns:
        
    """
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    This function is used to concatenate the outputs of the model.

    Args:
        box_cls (list): List of tensors of shape (N, AxHxW, C) where A is the number of anchors.
        box_regression (list): List of tensors of shape (N, AxHxW, 4) where A is the number of anchors.

    Returns:
        (tuple): tuple containing:
        box_cls (tensor): Tensor of shape (N, AxHxW, C) where A is the number of anchors.
        box_regression (tensor): Tensor of shape (N, AxHxW, 4) where A is the number of anchors.
    """
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the same format as the labels.
    for _cls, _reg in zip(box_cls, box_regression): # box_cls_perlevel, box_regression_perlevel
        N, AxC, H, W = _cls.shape
        Ax4 = _reg.shape[1]
        A = Ax4 // 4
        C = AxC // A
        _cls = permute_and_flatten(_cls, N, A, C, H, W)
        box_cls_flattened.append(_cls)
        _reg = permute_and_flatten(_reg, N, A, 4, H, W)
        box_regression_flattened.append(_reg)
    # concatenate on the first dimension (representing the feature levels),
    # to take into account the way the labels were generated (with all feature maps being concatenated as well)
    thbox_cls = th.cat(box_cls_flattened, dim=1).flatten(0, -2)
    thbox_regression = th.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return thbox_cls, thbox_regression


def boxes_union(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    This function is used to compute the union of two sets of boxes.

    Args:
        boxes1 (tensor): Tensor of shape (N, 4) where N is the number of boxes.
        boxes2 (tensor): Tensor of shape (N, 4) where N is the number of boxes.

    Returns:
        (tensor): Tensor of shape (N, 4) where N is the number of boxes.
    """
    assert boxes1.shape == boxes2.shape
    boxes1 = boxes1.cpu().numpy()
    boxes2 = boxes2.cpu().numpy()
    xmin = np.minimum(boxes1[:, 0], boxes2[:, 0])
    ymin = np.minimum(boxes1[:, 1], boxes2[:, 1])
    xmax = np.maximum(boxes1[:, 2], boxes2[:, 2])
    ymax = np.maximum(boxes1[:, 3], boxes2[:, 3])
    return th.from_numpy(np.vstack((xmin, ymin, xmax, ymax)).transpose()).to(cfg_device)
