# this is main file for faster rcnn model
# it combines all the other modules (backbone(vgg16 or resnet101), anchorgeni, rpn, roi_pooling)

import torch as th
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from ...config import reader
from ..backbone.vgg import vgg16
from ..backbone.resnet import resnet101
from .rpn import RPN
from .roi import RoIHead
from _utils import (postprocess_boxes, flatten_targets, unflatten_targets)

from torch import Tensor
from typing import Dict, List, Optional, Tuple

cfg = reader()
cfg_device = cfg["device"]

# transform parameters
min_size = cfg['rcnn_transformer']['min_size']
max_size = cfg['rcnn_transformer']['max_size']
image_mean = cfg['rcnn_transformer']['image_mean']
image_std = cfg['rcnn_transformer']['image_std']

class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.fpn = resnet_fpn_backbone(backbone_name='resnet101', pretrained=True, trainable_layers=5)
        self.rpn = RPN()
        self.roi = RoIHead()
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    def forward(self, images, targets=None):
        original_image_sizes = th.jit.annotate(List[Tuple[int, int]], [])
        # add images width and height to original_image_sizes
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        if targets: # for training
            targets = flatten_targets(targets)
        images, targets = self.transform(images, targets) # look below :)
        fpn_feature_maps = self.fpn(images.tensors.to(cfg_device)) # extract feature maps from backbone
        if targets:
            # apply region proposal network (RPN) to the feature maps
            proposals, rpn_losses, fpn_feature_maps = self.rpn(images, fpn_feature_maps, targets)
            targets = unflatten_targets(targets)
            # apply region of interest (RoI) pooling to the feature maps
            detections, detector_losses = self.roi(fpn_feature_maps, proposals, images.image_sizes, targets)
            losses = {}
            losses.update(detector_losses)
            losses.update(rpn_losses)
        else: # for inference
            losses = {}
            proposals, rpn_losses, fpn_feature_maps = self.rpn(images, fpn_feature_maps)
            detections, detector_losses = self.roi(fpn_feature_maps, proposals, images.image_sizes)
            detections = postprocess_boxes(detections, images.image_sizes, original_image_sizes)
        return detections, losses


# why GeneralizedRCNNTransform is needed?
# it is used to transform images and targets to the same size and format for training and inference
# operations:
# 1. resize: scales the image so that its shorter side is equal to a predefined size,
#    while preserving the aspect ratio of the image
# 2. normalize: subtracts the mean RGB values and divides by the standard deviation of the dataset,
#    which helps to reduce the effect of lighting variations in the images.
# 3. data augmentation: 
#   3.1. random horizontal flipping: flips the image horizontally with a probability of 0.5,
#        which helps to increase the diversity of the training data.
#   3.2. random cropping: selects a rectangular region of the image and resizes it to the desired output size.
#   3.3  photometric distortion: which help to simulate variations in lighting conditions and improve the robustness of the model.
# its output is a dict containing the transformed images and targets
