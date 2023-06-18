# helper functions and classes for the faster-rcnn model

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    __annotations__ = {
        "tensors": Tensor,
        "image_sizes": List[Tuple[int, int]],
    }

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: th.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


