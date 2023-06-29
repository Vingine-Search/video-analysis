# this class helps in loading data from VRD dataset

from copy import deepcopy
from ..config import reader
from PIL import Image

import torch as th
from torchvision import transforms
from torch.utils.data import Dataset

from ._utils import (
    make_image_list,
    load_data,
    image_path,
    change_coords_order
)

from typing import List, Tuple

cfg = reader()
cfg_device = cfg["device"]


class VRD(Dataset):
    """ This class helps in loading data from VRD dataset """

    def __init__(self, dataset_path:str, data_type:str) -> None:
        self.dataset_path = dataset_path
        self.data_type = data_type
        self.annotations, self.all_objects, self.predicates, self.sg_images = load_data(dataset_path, data_type)
        self.classes, self.preds = deepcopy(self.all_objects), deepcopy(self.predicates)
        self.classes.insert(0, '__background__')
        self.preds.insert(0, 'unknown')
        self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self._preds_to_ind = dict(zip(self.preds, range(len(self.preds))))
        self.imgs_list = make_image_list(dataset_path, data_type)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_annotation(self, img_name:str) -> Tuple[List, List, List]:
        """
        Load annotation for a given image

        Args:
            img_name (str): name of the image

        Returns:
            Tuple of:
            - **boxes**: list of bounding boxes for subject and object
            - **labels**: list of labels for subject and object
            - **preds**: list of predicates
        """
        boxes = []
        labels = []
        preds = []
        annotation = self.annotations[img_name]
        for spo in annotation: # spo: subject-predicate-object
            # collect ground truth (subject, object, predicate) labels & bounding boxes
            gtsbj_label = spo['subject']['category']
            gtsbj_bbox = spo['subject']['bbox']
            gtobj_label = spo['object']['category']
            gtobj_bbox = spo['object']['bbox']
            predicate = spo['predicate']
            # prepare bboxes for subject and object
            gtsbj_bbox = change_coords_order(gtsbj_bbox)
            gtobj_bbox = change_coords_order(gtobj_bbox)
            boxes.append([gtsbj_bbox, gtobj_bbox])
            # prepare labels for subject and object i.e map to word
            gtsbj_label = self.all_objects[gtsbj_label]
            gtobj_label = self.all_objects[gtobj_label]
            predicate = self.predicates[predicate]
            # map to new index
            labels.append([self._class_to_ind[gtsbj_label], self._class_to_ind[gtobj_label]])
            preds.append(self._preds_to_ind[predicate])
        return boxes, labels, preds

    def __len__(self) -> int:
        return len(self.imgs_list)

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        boxes, labels, preds = self.load_annotation(img_name)
        img_path = image_path(self.dataset_path, self.data_type, img_name)
        img = Image.open(img_path)
        img = self.transform(img)
        assert len(boxes) == len(labels), "boxes and labels should be of equal length"
        return {
            'boxes': th.tensor(boxes, dtype=th.float32),
            'labels': th.tensor(labels, dtype=th.int64),
            'preds': th.tensor(preds, dtype=th.int64),
            'img': img
        }

