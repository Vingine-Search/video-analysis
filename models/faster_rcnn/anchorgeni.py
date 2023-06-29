# this is an implementation of the Anchor Generator module
# that is used in the Region Proposal Network module

import torch as th
import torch.nn as nn

from torch import Tensor
from typing import Tuple, List

from .imagelist import ImageList


class AnchorGeni(nn.Module):
    """
    Generates anchors for a set of feature maps and image sizes.

    Args:
        sizes (Tuple[Tuple[int]]): sizes of the anchors for each feature map.
        aspect_ratios (Tuple[Tuple[float]]): aspect ratios of the anchors for each feature map.

    Note:
        sizes and aspect_ratios should have the same length.
    """

    __annotations__ = {
        "cell_anchors": Tuple[Tensor],
    }
    
    # TODO: static type checking for sizes and aspect_ratios
    def __init__(
        self,
        sizes = ((128, 256, 512),),
        aspect_ratios = ((0.5, 1.0, 2.0),),
    ):
        super().__init__()
        # make sure that sizes and aspect_ratios are tuples of tuples
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        # generate a base anchor for each size and aspect ratio
        self.cell_anchors = [self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)]

    def generate_anchors(
        self,
        scales: Tuple[int],
        aspect_ratios: Tuple[float],
        dtype: th.dtype = th.float32,
        device: th.device = th.device("cpu"),
    ) -> Tensor:
        """
        Generate anchors for a given size and aspect ratio.

        Args:
            scales (Tuple[int]): scales of the anchors.
            aspect_ratios (Tuple[float]): aspect ratios of the anchors.
            dtype (torch.dtype): the desired data type of returned tensor.
            device (torch.device): the desired device of returned tensor.

        Returns:
            anchors (torch.Tensor): a tensor of shape (N, 4), where N = len(scales) * len(aspect_ratios).
        """
        # change list to tensor
        th_scales = th.as_tensor(scales, dtype=dtype, device=device)
        th_aspect_ratios = th.as_tensor(aspect_ratios, dtype=dtype, device=device)
        # compute the width and height ratios of each anchor
        h_ratios = th.sqrt(th_aspect_ratios)
        w_ratios = 1 / h_ratios
        # compute the scaled width and height of each anchor
        ws = (w_ratios[:, None] * th_scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * th_scales[None, :]).view(-1)
        base_anchors = th.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()


    def change_cell_anchors(self, dtype: th.dtype, device: th.device) -> None:
        """
        Change the data type and device of cell_anchors.

        Args:
            dtype (torch.dtype): the desired data type of cell_anchors.
            device (torch.device): the desired device of cell_anchors.
        """
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self) -> List[int]:
        """
        Return the number of anchors per location for each feature map.
        """
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        """
        Rebasing anchors on the cell anchors and shifts on a regular grid.

        Args:
            grid_sizes (List[List[int]]): the sizes of the grids / feature maps.
            strides (List[List[Tensor]]): the strides between each anchor box per grid.

        Returns:
            anchors after being rebased on the cell anchors and shifts on a regular grid.
        """
        anchors = []
        cell_anchors = self.cell_anchors
        th._assert(cell_anchors is not None, "cell_anchors should not be None")
        th._assert( len(grid_sizes) == len(strides) == len(cell_anchors),
            "feature maps passed and the number of sizes / aspect ratios specified should match",
        )
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            # basic setup
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            # e.g tensor([0, 1, 2, 3, 4]) * 2 = tensor([0, 2, 4, 6, 8])
            shifts_x = th.arange(0, grid_width, dtype=th.float32, device=device) * stride_width
            shifts_y = th.arange(0, grid_height, dtype=th.float32, device=device) * stride_height
            # create x and y grids of shifts
            shift_y, shift_x = th.meshgrid(shifts_y, shifts_x, indexing="ij")
            # flatten each grid and stack 2 of them
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = th.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shifted anchors (K, A, 4)
            # and the reshaping collapses the first two dimensions to get a tensor of shape (K*A, 4)
            anchors.append((shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4))
        return anchors

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        """
        Args:
            image_list (ImageList): a list of images into a single tensor.
            feature_maps (List[Tensor]): a list of feature maps for each image.

        Returns:
            anchors (List[Tensor]): a list of anchors.
        """
        grid_sizes = [list(feature_map.shape[-2:]) for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # compute the strides of each feature map
        strides = [[th.tensor(image_size[0] // g[0], dtype=th.int64, device=device),
             th.tensor(image_size[1] // g[1], dtype=th.int64, device=device)] for g in grid_sizes]
        # change the data type and device of cell_anchors to be of the same
        # data type and device as the feature maps
        self.change_cell_anchors(dtype, device)
        # compute anchors over all feature maps (all anchors offsited correctly with the strides list)
        return self.grid_anchors(grid_sizes, strides)

