# this is the region of interest module (ROI) for the faster rcnn model
# gt: ground truth

import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection._utils import (
    BoxCoder,
    Matcher,
    BalancedPositiveNegativeSampler,
)
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from torch import Tensor
from typing import Dict, List, Optional, Tuple

from ...config.config import reader
from ..reldn.reldn import RelDN
from .twomlp import TwoMLP
from .predictor import Predictor
from ._utils import union_boxes
from .loss import fastrcnn_loss, reldn_loss

cfg = reader()
cfg_device = cfg["device"]
test_threshold = cfg["test"]["threshold"]

# box parameters
box_num_classes = cfg["box"]["num_classes"]
box_score_thresh = cfg["box"]["score_thresh"]
box_nms_thresh = cfg["box"]["nms_thresh"]
box_detections_perimg = cfg["box"]["detections_perimg"]
box_fg_thresh = cfg["box"]["fg_thresh"]
box_bg_thresh = cfg["box"]["bg_thresh"]
box_batch_size_perimage = cfg["box"]["batch_size_perimage"]
box_positive_fraction = cfg["box"]["positive_fraction"]

# model parameters
model_batchsize_perimage_so = cfg["model"]["batch_size_perimage_so"]
model_positive_fraction_so = cfg["model"]["positive_fraction_so"]
model_batchsize_perimage_rel = cfg["model"]["batch_size_perimage_rel"]
model_positive_fraction_rel = cfg["model"]["positive_fraction_rel"]
model_norm_scale = cfg["model"]["norm_scale"]


class RoIHead(nn.Module):
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher,
        'sampler': BalancedPositiveNegativeSampler,
    }
    
    def __init__(self):
        super(RoIHead, self).__init__()
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'], # what featuremap names to use
            output_size=7, # 7x7 output
            sampling_ratio=2
        )
        resolution = self.box_roi_pool.output_size[0] # 7
        representation_size = 1024 # specifies the size hidden & output layers of the two mlp heads
        # 256 learned features multiplied by the spatial extent of the output feature map (7x7)
        self.box_head = TwoMLP(256 * resolution ** 2, representation_size)
        self.rlp_head = copy.deepcopy(self.box_head)
        # predict the object class and bounding box regression (for each feature map)
        representation_size = 1024
        self.box_predictor = Predictor(representation_size, box_num_classes)
        # for the visual & semantic embeddings
        self.RelDN = RelDN(self.box_head.fc7.out_features * 3)
        self.box_similarity = box_ops.box_iou # IoU similarity
        # match proposals to ground truth boxes (returns the indices of the matched gt box)
        self.proposal_matcher = Matcher(box_fg_thresh, box_bg_thresh, allow_low_quality_matches=False)
        # samplers to ensure a fixed number of positive and negative samples for
        self.sampler = BalancedPositiveNegativeSampler(box_batch_size_perimage, box_positive_fraction) # boxes
        self.sampler_so = BalancedPositiveNegativeSampler(model_batchsize_perimage_so, model_positive_fraction_so) # subject-object pairs
        self.sampler_rlp = BalancedPositiveNegativeSampler(model_batchsize_perimage_rel, model_positive_fraction_rel) # relation pairs
        self.box_coder = BoxCoder((10., 10., 5., 5.)) # encode/decode bounding boxes during training/evaluation

    def assign_pred_to_rlp_proposals(
        self,
        sbj_proposals: List[Tensor],
        obj_proposals: List[Tensor],
        gt_boxes: List[Tensor],
        gt_labels: List[Tensor],
        gt_preds: List[Tensor],
    ) -> List[Tensor]:
        """
        Assign predict to (subject, object) relation proposals

        Args:
            sbj_proposals (List[Tensor]): subject proposals
            obj_proposals (List[Tensor]): object proposals
            gt_boxes (List[Tensor]): ground truth boxes
            gt_labels (List[Tensor]): ground truth labels
            gt_preds (List[Tensor]): ground truth predicates

        Returns:
            List[Tensor]: the matched indices of the proposals to the ground truth boxes for each image
        """
        labels = []
        # this loop is for each image in the batch
        for sbj, obj, gtbx, _, gtprd in zip(sbj_proposals, obj_proposals, gt_boxes, gt_labels, gt_preds):
            # Remove dulplicates for sbj and obj gths
            gtsbj = th.unique(gtbx[:, 0, :], dim=0)
            gtobj = th.unique(gtbx[:, 1, :], dim=0)
            # Compute similarity matrix for each of sbj and obj proposals and the gths
            sbj_sim_mat = box_ops.box_iou(gtsbj, sbj)
            obj_sim_mat = box_ops.box_iou(gtobj, obj)
            # get the matched indices
            sbj_matched_idxs = self.proposal_matcher(sbj_sim_mat)
            obj_matched_idxs = self.proposal_matcher(obj_sim_mat)
            # the matched boxes themselves
            sbj_boxes = gtsbj[sbj_matched_idxs]
            obj_boxes = gtobj[obj_matched_idxs]
            cur_labels = th.zeros(sbj_boxes.shape[0])
            for i in range(len(sbj_boxes)):
                # get the indices of the matched boxes for sbj and obj from the gths
                sbj_indices = th.where(th.all(gtbx[:, 0, :] == sbj_boxes[i], dim=1))[0]
                obj_indices = th.where(th.all(gtbx[:, 1, :] == obj_boxes[i], dim=1))[0]
                # intersect the indices to get the index of the matched gtbox for the sbj-obj pair
                matched_idx = np.intersect1d(sbj_indices.cpu().numpy(), obj_indices.cpu().numpy())
                if matched_idx.any(): # if there is a match then assign the label
                    cur_labels[i] = gtprd[matched_idx[0]]
            cur_labels = cur_labels.to(dtype=th.int64, device=th.device(cfg_device))
            labels.append(cur_labels)
        return labels

    def assign_targets_to_proposals(
        self,
        proposals: List[Tensor],
        gt_boxes: List[Tensor],
        gt_labels: List[Tensor],
        assign_to: str='all'
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Assign ground truth boxes and labels to proposals

        Args:
            proposals (List[Tensor]): the region proposals
            gt_boxes (List[Tensor]): ground truth boxes
            gt_labels (List[Tensor]): ground truth labels
            assign_to (str): 'all' or 'subject'

        Returns:
            Tuple[List[Tensor], List[Tensor]]: the matched indices of the proposals to the ground truth boxes for each image
            and the labels of the corresponding ground truth boxes
        """
        matched_idxs = []
        labels = []
        if assign_to == "subject":
            slice_index = 0
        elif assign_to == 'all':
            slice_index = -1
        else:
            slice_index = 1
        # this loop is for each image in the batch
        for prop, gtboxs, gtlbls in zip(proposals, gt_boxes, gt_labels):
            if slice_index >= 0:
                gtboxs = gtboxs[:, slice_index, :]
                gtlbls = gtlbls[:, slice_index]
            else:
                gtboxs = gtboxs.view(-1, 4)
                gtlbls = gtlbls.view(-1)
            if gtboxs.numel() == 0: # Background image
                device = prop.device
                clamped_matched_idxs = th.zeros((prop.shape[0],), dtype=th.int64, device=device)
                cur_lables = th.zeros((prop.shape[0],), dtype=th.int64, device=device)
            else:
                # Compute similarity matrix between gt boxes and proposals and match them
                sim_mat = box_ops.box_iou(gtboxs, prop)
                cur_matched_idxs = self.proposal_matcher(sim_mat)
                clamped_matched_idxs = cur_matched_idxs.clamp(min=0)
                cur_lables = gtlbls[clamped_matched_idxs] # foreground (+ve ones)
                cur_lables = cur_lables.to(dtype=th.int64)
                bg_inds = cur_matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD # background
                cur_lables[bg_inds] = 0
                ignore_inds = cur_matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS # ignore
                cur_lables[ignore_inds] = -1  # -1 is ignored by sampler
            matched_idxs.append(clamped_matched_idxs)
            labels.append(cur_lables)
        return matched_idxs, labels

    def subsample(self, labels:List[Tensor], sample_for:str="all") -> List[Tensor]:
        """
        Prepare the subsample indices for the proposals

        Args:
            labels (List[Tensor]): the labels of the ground truth boxes
            sample_for (str): 'all' or 'subject'

        Returns:
            List[Tensor]: the subsample indices for the proposals
        """
        if sample_for == "all":
            pos_inds, neg_inds = self.sampler(labels)
        elif sample_for == "rel":
            pos_inds, neg_inds = self.sampler_rlp(labels)
        else:
            pos_inds, neg_inds = self.sampler_so(labels)
        sampled_inds = []
        # this loop is for each image in the batch
        for pos, neg in zip(pos_inds, neg_inds):
            idxs = th.nonzero(pos | neg).squeeze(1)
            sampled_inds.append(idxs)
        return sampled_inds

    def add_gt_proposals(self, proposals:List[Tensor], gt_boxes:List[Tensor]) -> List[Tensor]:
        """
        Add ground truth boxes to the set of proposals

        Args:
            proposals (List[Tensor]): the region proposals
            gt_boxes (List[Tensor]): ground truth boxes

        Returns:
            List[Tensor]: the proposals with the ground truth boxes added
        """
        proposals = [
            th.cat((proposal, gt_box.view(-1, 4)))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def remove_self_pairs(self, sbj_inds:np.ndarray, obj_inds:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove the self pairs from the subject and object indices

        Args:
            sbj_inds (np.ndarray): the subject indices
            obj_inds (np.ndarray): the object indices

        Returns:
            Tuple[np.ndarray, np.ndarray]: the subject and object indices without the self pairs
        """
        mask = sbj_inds != obj_inds
        return sbj_inds[mask], obj_inds[mask]

    def extract_positive_proposals(self, labels, proposals):
        """
        Extract the positive proposals from the proposals (with labels > 0) 

        Args:
            labels (List[Tensor]): the labels of the ground truth boxes
            proposals (List[Tensor]): the region proposals

        Returns:
            Tuple[List[Tensor], List[Tensor]]: the positive labels and their corresponding proposals
        """
        props, lbls = [], []
        for proposal, label in zip(proposals, labels):
            mask = label > 0
            lbls.append(label[mask])
            props.append(proposal[mask])
        return lbls, props

    def _check_targets(self, targets:Optional[List[Dict[str, Tensor]]]) -> None:
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def _check_target_types(self, targets:List[Dict[str, Tensor]]) -> None:
        assert targets is not None
        for t in targets:
            floating_point_types = (th.float, th.double, th.half)
            assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
            assert t["labels"].dtype == th.int64, 'target labels must of int64 type'

    def select_training_samples(
        self,
        proposals: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]],
    ):
        """
        Select the training samples from the proposals and the ground truth boxes

        Args:
            - proposals (List[Tensor]): the region proposals
            - targets (Optional[List[Dict[str, Tensor]]]): the ground truth boxes

        Returns:
            Tuple of:
            - **all_proposals** (List[Tensor]): the region proposals
            - **matched_idxs** (List[Tensor]): the matched indices for the proposals
            - **labels** (List[Tensor]): the labels for the proposals
            - **regression_targets** (List[Tensor]): the regression targets for the proposals
            - **data_sbj** (List[Tensor]): the subject indices for the proposals
            - **data_obj** (List[Tensor]): the object indices for the proposals
            - **data_rlp** (List[Tensor]): the relation labels for the proposals
        """
        self._check_targets(targets)
        assert targets is not None
        dtype, device = proposals[0].dtype, proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_preds = [t["preds"] for t in targets]
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, assign_to="all")
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels, sample_for="all")  # size 512
        # prepare the subsampled proposals
        all_proposals = proposals.copy()
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            # get gt boxes (from all_proposals copy), labels and the corresponding matched indices
            # for each image found in the subsampled proposals (sampled_inds)
            img_sampled_inds = sampled_inds[img_id]
            all_proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            # get the gt boxes corresponding to the matched indices if they exist
            gt_boxes_in_image = gt_boxes[img_id].view(-1, 4)
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = th.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        # encode the gt boxes to be the regression targets used in training/evaluation
        regression_targets = self.box_coder.encode(matched_gt_boxes, all_proposals)
        # get matching gt indices for subject proposals
        _, sbj_labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, assign_to="subject")
        sampled_inds = self.subsample(sbj_labels, sample_for="subject")
        sbj_proposals = proposals.copy()
        for img_id in range(num_images): # same as upove
            img_sampled_inds = sampled_inds[img_id]
            sbj_proposals[img_id] = proposals[img_id][img_sampled_inds]
            sbj_labels[img_id] = sbj_labels[img_id][img_sampled_inds]
        pos_sbj_labels, pos_sbj_proposals = self.extract_positive_proposals(sbj_labels, sbj_proposals)
        # get matching gt indices for object proposals
        _, obj_labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, assign_to="objects")
        sampled_inds = self.subsample(obj_labels, sample_for="object")
        obj_proposals = proposals.copy()
        for img_id in range(num_images): # same as upove
            img_sampled_inds = sampled_inds[img_id]
            obj_proposals[img_id] = proposals[img_id][img_sampled_inds]
            obj_labels[img_id] = obj_labels[img_id][img_sampled_inds]
        pos_obj_labels, pos_obj_proposals = self.extract_positive_proposals(obj_labels, obj_proposals)
        # prepare relation proposals
        rlp_proposals = []
        # effectively what this loop do: 
        # is to create a grid of all possible pairs of subject and object proposals
        # and append a correct box that union both subject and object proposals to rlp_proposals
        for img_id in range(num_images):
            sbj_shape = pos_sbj_labels[img_id].shape[0]
            obj_shape = pos_obj_labels[img_id].shape[0]
            sbj_inds = np.repeat(np.arange(sbj_shape), obj_shape) # repeat each element
            obj_inds = np.tile(np.arange(obj_shape), sbj_shape) # repeat the whole array
            pos_sbj_labels[img_id] = pos_sbj_labels[img_id][sbj_inds]
            pos_obj_labels[img_id] = pos_obj_labels[img_id][obj_inds]
            pos_sbj_proposals[img_id] = pos_sbj_proposals[img_id][sbj_inds]
            pos_obj_proposals[img_id] = pos_obj_proposals[img_id][obj_inds]
            rlp_proposals.append(union_boxes(pos_obj_proposals[img_id], pos_sbj_proposals[img_id]))
        # assign gt_predicate to relation proposals
        rlp_labels = self.assign_pred_to_rlp_proposals(pos_sbj_proposals, pos_obj_proposals, gt_boxes, gt_labels, gt_preds)
        # now we subsample the relations to have a fixed number of positive relations we can train on
        sampled_inds = self.subsample(rlp_labels, sample_for="rel")
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            pos_sbj_proposals[img_id] = pos_sbj_proposals[img_id][img_sampled_inds]
            pos_obj_proposals[img_id] = pos_obj_proposals[img_id][img_sampled_inds]
            rlp_proposals[img_id] = rlp_proposals[img_id][img_sampled_inds]
            pos_sbj_labels[img_id] = pos_sbj_labels[img_id][img_sampled_inds]-1
            pos_obj_labels[img_id] = pos_obj_labels[img_id][img_sampled_inds]-1
            rlp_labels[img_id] = rlp_labels[img_id][img_sampled_inds]
        data_sbj = {'proposals': pos_sbj_proposals, 'labels': pos_sbj_labels}
        data_obj = {'proposals': pos_obj_proposals, 'labels': pos_obj_labels}
        data_rlp = {'proposals': rlp_proposals, 'labels': rlp_labels}
        return all_proposals, matched_idxs, labels, regression_targets, data_sbj, data_obj, data_rlp

    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ):
        """
        Postprocesses the output of an object detection model to produce final detections.

        Args:
            class_logits (Tensor): the classification scores for all anchors. Shape: (N, A * K)
            box_regression (Tensor): predicted boxes regression deltas. Shape: (N, A * 4)
            proposals (list[Tensor]): list of proposals used to perform the computation, one per image.
            image_shapes (list[tuple]): the sizes of each image in the batch

        Returns:
            Tuple of:
            - **all_boxes**: list of tensors containing proposals boxes for each image. shape (R, 4),
            - **all_scores**: list of tensors containing scores for each image. shape (R, K),
            - **all_labels**: list of tensors containing labels for each image. shape (R, K),
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals) # decode the predicted boxes
        pred_scores = F.softmax(class_logits, -1) # give a score to each class
        # split_size in this case is a list containing the number of predicted boxes per image
        # so we split the predicted boxes and scores per image (since dim=0 is the batch dim)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        # so effectively, pred_boxes_list and pred_scores_list are lists of tensors
        # where each tensor is the predicted boxes or scores for each image
        all_boxes = []
        all_scores = []
        all_labels = []
        # for each image
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # create labels for each prediction
            labels = th.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # remove low scoring boxes
            inds = th.nonzero(scores > box_score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, box_nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:box_detections_perimg]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
        if targets: # if training
            proposals, matched_idxs, labels, regression_targets, data_sbj, data_obj, data_rlp = \
                    self.select_training_samples( proposals, targets)

            # faster_rcnn branch
            # apply the RoI pooling operation to get fixed size features for all RoIs
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            # pass those features to TwoMLP head to further prepare them for classification and regression
            box_features = self.box_head(box_features)
            # obtain the predicted class scores and bounding box regression scores
            class_logits, box_regression = self.box_predictor(box_features)

            # predicate branch
            # apply RoI pooling above (subject, object, predicate) proposals to get fixed size features
            sbj_feat = self.box_roi_pool(features, data_sbj["proposals"], image_shapes)
            obj_feat = self.box_roi_pool(features, data_obj["proposals"], image_shapes)
            rel_feat = self.box_roi_pool(features, data_rlp["proposals"], image_shapes)
            # pass the result features vectors to the TwoMLP head to get the classification scores
            sbj_feat = self.box_head(sbj_feat)
            obj_feat = self.box_head(obj_feat)
            rel_feat = self.rlp_head(rel_feat) # rlp_head is the same as box_head
            # pass spo(sbj+rel+obj), sbj, obj, rlp features to RelDN and get the scores
            concat_feat = th.cat((sbj_feat, rel_feat, obj_feat), dim=1)
            sbj_cls_scores, obj_cls_scores, rlp_cls_scores = \
                self.RelDN(concat_feat, sbj_feat, obj_feat, targets)
            #
            result = th.jit.annotate(List[Dict[str, th.Tensor]], [])
            losses = {}
            assert labels is not None and regression_targets is not None
            # compute the losses from RelDN and from faster_rcnn branch
            loss_cls_sbj, accuracy_cls_sbj = reldn_loss(sbj_cls_scores, data_sbj["labels"])
            loss_cls_obj, accuracy_cls_obj = reldn_loss(obj_cls_scores, data_obj['labels'])
            loss_cls_rlp, accuracy_cls_rlp = reldn_loss(rlp_cls_scores, data_rlp['labels'])
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_sbj": loss_cls_sbj,
                "acc_sbj"	: accuracy_cls_sbj.item(),
                "loss_obj": loss_cls_obj,
                "acc_obj"	: accuracy_cls_obj.item(),
                "loss_rlp": loss_cls_rlp,
                "acc_rlp"	: accuracy_cls_rlp.item()
            }

        else: # inference
            result = []
            # faster_rcnn branch
            # apply the RoI pooling operation to get fixed size features for all RoIs
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            # pass those features to TwoMLP head to further prepare them for classification and regression
            box_features = self.box_head(box_features)
            # obtain the predicted class scores and bounding box regression scores
            class_logits, box_regression = self.box_predictor(box_features)
            # postprocesses the result by applying NMS and removing boxes that are too small
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)

            # applay same trick as above in select_training_samples
            # i.e get all possible pairs of boxes for sbj, obj and compute union boxes for rlp
            all_sbj_boxes = []
            all_obj_boxes = []
            all_rlp_boxes = []
            all_shapes = []
            for img_id in range(num_images):
                sbj_inds = np.repeat(np.arange(boxes[img_id].shape[0]), boxes[img_id].shape[0])
                obj_inds = np.tile(np.arange(boxes[img_id].shape[0]), boxes[img_id].shape[0])
                sbj_inds, obj_inds = self.remove_self_pairs(sbj_inds, obj_inds)
                sbj_boxes = boxes[img_id][sbj_inds]
                obj_boxes = boxes[img_id][obj_inds]
                rlp_boxes = union_boxes(sbj_boxes, obj_boxes)
                all_sbj_boxes.append(sbj_boxes)
                all_obj_boxes.append(obj_boxes)
                all_rlp_boxes.append(rlp_boxes)
                all_shapes.append(rlp_boxes.shape[0])

            # predicate branch (similar to training)
            # apply RoI pooling above (subject, object, predicate) proposals to get fixed size features
            sbj_feat = self.box_roi_pool(features, all_sbj_boxes, image_shapes)
            obj_feat = self.box_roi_pool(features, all_obj_boxes, image_shapes)
            rel_feat = self.box_roi_pool(features, all_rlp_boxes, image_shapes)
            # pass the result features vectors to the TwoMLP head to get the classification scores
            sbj_feat = self.box_head(sbj_feat)
            obj_feat = self.box_head(obj_feat)
            rel_feat = self.rlp_head(rel_feat)
            # pass spo(sbj+rel+obj), sbj, obj, rlp features to RelDN and get the scores
            concat_feat = th.cat((sbj_feat, rel_feat, obj_feat), dim=1)
            sbj_cls_scores, obj_cls_scores, rlp_cls_scores = \
                self.RelDN(concat_feat, sbj_feat, obj_feat)
            # convert the scores to lists
            sbj_cls_scores_list, obj_cls_scores_list, rlp_cls_scores_list = \
                sbj_cls_scores.split(all_shapes), obj_cls_scores.split(all_shapes), rlp_cls_scores.split(all_shapes)

            # for each image, get the top k predictions
            for i, _ in enumerate(sbj_cls_scores_list):
                _, sbj_indices = th.max(sbj_cls_scores_list[i], dim=1)
                _, obj_indices = th.max(obj_cls_scores_list[i], dim=1)
                rel_scores, rel_indices = th.max(rlp_cls_scores_list[i], dim=1)
                # filter "unknown"
                mask = rel_indices > 0
                rel_scores = rel_scores[mask]
                predicates = rel_indices[mask]
                subjects = sbj_indices[mask]
                objects = obj_indices[mask]
                # before rejecting based on threshold, get the boxes
                sbj_boxes = all_sbj_boxes[i][mask]
                obj_boxes = all_obj_boxes[i][mask]
                rlp_boxes = all_rlp_boxes[i][mask]
                # apply some external/hyper threshold
                score_mask = rel_scores > test_threshold
                result = [{
                   "sbj_boxes": sbj_boxes[score_mask],
                   "obj_boxes": obj_boxes[score_mask],
                   "sbj_labels": subjects[score_mask],
                   "obj_labels": objects[score_mask],
                   "predicates": predicates[score_mask],
                }]
            losses = {}
        return result, losses




