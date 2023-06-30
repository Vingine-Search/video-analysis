# this script is to define the relation detection network head (RelDN) used
# in the paper https://arxiv.org/abs/1804.10660 "Large-Scale Visual Relationship Understanding"
# to detect relation between two objects in an image using
# one as the subject, one as the object and a predicate "verb/relation" between them

# it's used to detect the visual embidding of the subject and object 
# and the predicate in relation to the regions
# in the image and also used to detect the semantic embiddings as well

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import reader
from .wordvec import get_obj_prd_vecs

cfg = reader()
norm_scale = cfg["model"]["norm_scale"]

class RelDN(nn.Module):
    """
    Relation Detection Network (RelDN) head used in the paper

    Args:
        dim_in(int): input dimension
    """
    def __init__(self, dim_in: int):
        super(RelDN, self).__init__()
        self.obj_vecs, self.prd_vecs = get_obj_prd_vecs()
        # visual embaddings of (subject, object) and predicate
        self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)
        self.prd_vis_embeddings = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024)
        )
        # semantic embaddings of (subject, object)
        self.so_sem_embeddings = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024)
        )
        # semantic embaddings of predicate has 3 sequential layers
        self.prd_sem_hidden = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024)
        )
        self.prd_sem_embeddings = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024)
        )
        # extract features from predicates
        self.prd_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1)
        )
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights and bias of all sub-models in RelDN
        """
        for m in self.modules():
            # incase of conv2d or linear layer, fill the weight with xavier uniform
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: # zero the bias if it's not none
                    nn.init.constant_(m.bias, 0)
            # incase of batchnorm2d, fill the weight with 1 and bias with 0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, spo_feat=None, sbj_feat=None, obj_feat=None, targets=None):
        """
        Get the embaddings of the subject and object and the predicate class scores
            
        Args:
            spo_feat(Tensor): the features of the subject, predicate and object
            sbj_feat(Tensor): the features of the subject
            obj_feat(Tensor): the features of the object

        Returns:
            Tuple of:
            - sbj_cls_scores(Tensor): the scores of the subject
            - obj_cls_scores(Tensor): the scores of the object
            - prd_cls_scores(Tensor): the scores of the predicate
        """
        device = sbj_feat.device
        # visual features to visual embaddings
        sbj_vemb = self.so_vis_embeddings(sbj_feat)
        obj_vemb = self.so_vis_embeddings(obj_feat)
        prd_hidden = self.prd_feats(spo_feat)
        prd_features = th.cat((sbj_vemb.detach(), prd_hidden, obj_vemb.detach()), dim=1)
        prd_vemb = self.prd_vis_embeddings(prd_features)

        # objects vectors to Variable to be used as computational node in the graph
        ds_obj_vecs = self.obj_vecs
        ds_obj_vecs = Variable(th.from_numpy(ds_obj_vecs.astype('float32'))).to(device)
        # subject_object semantic embaddings
        so_semb = self.so_sem_embeddings(ds_obj_vecs)
        so_semb = F.normalize(so_semb, p=2, dim=1) # L(p=2) normalization
        so_semb.t_() # to tensor
        # matrix of [subject visual embeddings] * [subject_object semantic embeddings]
        sbj_vemb = F.normalize(sbj_vemb, p=2, dim=1)
        sbj_sim_matrix = th.mm(sbj_vemb, so_semb)
        sbj_cls_scores = norm_scale * sbj_sim_matrix
        # matrix of [object visual embeddings] * [subject_object semantic embeddings]
        obj_vemb = F.normalize(obj_vemb, p=2, dim=1)
        obj_sim_matrix = th.mm(obj_vemb, so_semb)
        obj_cls_scores = norm_scale * obj_sim_matrix

        # predicate vectors to Variable to be used as computational node in the graph
        # predicate semantic embaddings
        # and matrix of [predicate visual embeddings] * [predicate semantic embeddings]
        ds_prd_vecs = self.prd_vecs
        ds_prd_vecs = Variable(th.from_numpy(ds_prd_vecs.astype('float32'))).to(device)
        prd_semb = self.prd_sem_embeddings(ds_prd_vecs)
        prd_semb = F.normalize(prd_semb, p=2, dim=1)
        prd_vemb = F.normalize(prd_vemb, p=2, dim=1)
        prd_sim_matrix = th.mm(prd_vemb, prd_semb.t_())
        prd_cls_scores = norm_scale * prd_sim_matrix

        if not targets:
            sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
            obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)

        return sbj_cls_scores, obj_cls_scores, prd_cls_scores

