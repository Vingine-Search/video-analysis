# this is an implementation of the region proposal network (RPN) in Faster R-CNN

# IoU: intersection over union
# NMS: non-maximum suppression
# GT box: ground truth box

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection._utils import (
    _topk_min,
    BoxCoder,
    Matcher,
    BalancedPositiveNegativeSampler,
)
from torchvision.ops import boxes as box_ops

from .rpnhead import RPNHead
from .anchorgeni import AnchorGeni
from .imagelist import ImageList
from ._utils import (
    permute_and_flatten,
    concat_box_prediction_layers,
)

from torch import Tensor
from functools import reduce
from typing import Dict, List, Optional, Tuple


class RPNCore(nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchorgeni (AnchorGeni): module that generates the anchors for a set of feature maps.
        rpnhead (nn.Module): module that computes the objectness and regression deltas
        fg_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_perimage (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_topn (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_topn (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        score_thresh (float): during inference, only return proposals with a classification score
        greater than this threshold
    """

    __annotations__ = {
        'anchorgeni': AnchorGeni,
        'head': RPNHead,
        'box_coder': BoxCoder, # encode and decode boxes
        'proposal_matcher': Matcher, # match proposals with GT boxes (it uses IoU as a metric)
        'sampler': BalancedPositiveNegativeSampler, # sample positive and negative anchors
    }

    def __init__(
        self,
        # Modules
        anchorgeni: AnchorGeni,
        rpnhead: RPNHead,
        # Training
        fg_thresh: float,
        bg_thresh: float,
        batch_size_perimage: int,
        positive_fraction: float,
        # Inference
        pre_nms_topn: Dict[str, int],
        post_nms_topn: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        # Modules
        self.anchorgeni = anchorgeni
        self.head = rpnhead
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        # used during training
        self.box_similarity = box_ops.box_iou # IoU as a metric for matching proposals with GT boxes
        self.proposal_matcher = Matcher(fg_thresh, bg_thresh, allow_low_quality_matches=True,)
        self.sampler = BalancedPositiveNegativeSampler(batch_size_perimage, positive_fraction)
        # used during testing
        self._pre_nms_topn = pre_nms_topn
        self._post_nms_topn = post_nms_topn
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_topn(self) -> int:
        """
        Return the number of proposals to keep before applying NMS during training or testing
        """
        if self.training:
            return self._pre_nms_topn['training']
        return self._pre_nms_topn['testing']

    def post_nms_topn(self) -> int:
        """
        Return the number of proposals to keep after applying NMS during training or testing
        """
        if self.training:
            return self._post_nms_topn['training']
        return self._post_nms_topn['testing']

    def assign_targets_to_anchors(
        self, 
        anchors: List[Tensor],
        targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Match anchors with ground-truth boxes, sample them, and compute the corresponding
        regression targets.

        Args:
            anchors (List[Tensor]): anchors for a set of feature maps
            targets (List[Dict[str, Tensor]]): ground-truth boxes for a set of images

        Returns:
            Tuple[List[Tensor], List[Tensor]]: the matched GT boxes and the labels for each anchor
        """
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            # numel returns the number of elements in a tensor
            # no elements means Background everywhere (negative example)
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = th.zeros(anchors_per_image.shape, dtype=th.float32, device=device)
                labels_per_image = th.zeros((anchors_per_image.shape[0],), dtype=th.float32, device=device)
            else:
                # compute the IoU between the anchors and the GT boxes
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix) # the indices of the matched GT boxes
                # get the labels for each anchor (Upove the threshold)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                labels_per_image = matched_idxs >= 0 # positive examples have a label of 1
                labels_per_image = labels_per_image.to(dtype=th.float32)
                # assign (0 for negative "background")
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0
                # assign (-1 for inbetween "ignored")
                ignore_indices = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[ignore_indices] = -1 # ignored examples have a label of -1
            # add the labels and the matched GT boxes for the current image
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_topn_idx(
        self,
        objectness: Tensor,
        num_anchors_perlevel: List[int],
    ) -> Tensor:
        """
        Get the indices of the top N anchors for each feature map.

        Args:
            objectness (Tensor): objectness logits for each anchor
            num_anchors_per_level (List[int]): number of anchors for each feature map

        Returns:
            Tensor: indices of the top N anchors for each feature map
        """
        result = []
        offset = 0
        for ob in objectness.split(num_anchors_perlevel, 1):
            num_anchors = ob.shape[1]
            pre_nms_topn = _topk_min(ob, self.pre_nms_topn(), 1)
            _, top_n_idx = ob.topk(pre_nms_topn, dim=1)
            result.append(top_n_idx + offset)
            offset += num_anchors
        return th.cat(result, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_perlevel: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Filter the proposals that are too small and apply NMS to the remaining ones.

        Args:
            proposals (Tensor): proposals
            objectness (Tensor): objectness logits for each anchor
            image_shapes (List[Tuple[int, int]]): shapes of the images
            num_anchors_perlevel (List[int]): number of anchors for each feature map

        Returns:
            Tuple[List[Tensor], List[Tensor]]: the filtered proposals and the corresponding
                scores for each proposal
        """
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach() # no backpropagation through objectness
        objectness = objectness.reshape(num_images, -1)
        # levels are the indices of the feature maps that the anchors belong to
        levels = [th.full((n,), idx, dtype=th.int64, device=device) for idx, n in enumerate(num_anchors_perlevel)]
        levels = th.cat(levels, dim=0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        # select topn boxes per level before nms
        topn_idx = self._get_topn_idx(objectness, num_anchors_perlevel)
        image_range = th.arange(num_images, device=device) # the indices of the images
        batch_idx = image_range[:, None] # the indices of the images for each anchor
        objectness = objectness[batch_idx, topn_idx] # the objectness logits for the top N anchors
        levels = levels[batch_idx, topn_idx] # the indices of the feature maps for the top N anchors
        proposals = proposals[batch_idx, topn_idx] # the top N anchors
        objectness_prob = th.sigmoid(objectness) # the objectness probabilities for the top N anchors
        # perform final cleanup for the proposals
        # small -> low scores -> NMS -> keep post_nms_topk
        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # make sure that the proposals are inside the image
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # remove low scoring boxes
            keep = th.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.post_nms_topn()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(
        self,
        objectness: Tensor,
        pred_bbox_deltas: Tensor,
        labels: List[Tensor],
        regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the loss for the RPN.

        Args:
            objectness (Tensor): objectness logits for each anchor
            pred_bbox_deltas (Tensor): predicted bbox deltas for each anchor
            labels (List[Tensor]): labels for each anchor
            regression_targets (List[Tensor]): regression targets for each anchor

        Returns:
            Tuple[Tensor, Tensor]: the objectness loss and the regression loss
        """
        # sample labels into +ve and -ve ones
        pos_inds, neg_inds = self.sampler(labels)
        pos_inds = th.where(th.cat(pos_inds, dim=0))[0]
        neg_inds = th.where(th.cat(neg_inds, dim=0))[0]
        inds = th.cat([pos_inds, neg_inds], dim=0)
        # prepare the labels and regression targets for the loss function
        objectness = objectness.flatten()
        thlabels = th.cat(labels, dim=0)
        thregression_targets = th.cat(regression_targets, dim=0)
        # smooth_l1_loss(x, y) = 0.5 * (x - y) ** 2 if abs(x - y) < 1 else abs(x - y) - 0.5
        # beta controls the width of the smooth_l1_loss function
        # reduction="sum" loss is summed over all the elements
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[pos_inds],
            thregression_targets[pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (inds.numel()) # as if it's averaged over all the elements
        # binary_cross_entropy_with_logits(x, y) = -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[inds], thlabels[inds])
        return objectness_loss, box_loss

    def forward(
        self,
        images: ImageList,
        _features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(_features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchorgeni(images, features)
        # 
        num_images = len(anchors)
        num_anchors_perlevel = [reduce(lambda x, y: x * y, ob[0].shape) for ob in objectness]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors) # no gradients in fast-rcnn proposals
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_perlevel)
        # if training, compute the losses
        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets) # assign gt boxes to anchors
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors) # encode the gt boxes
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

