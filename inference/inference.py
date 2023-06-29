# this is a script to do inference on a single image

import os
import cv2
import json
import numpy as np
from PIL import Image
from copy import deepcopy
import argparse

import torch as th
from torch.hub import load_state_dict_from_url
from torchvision import transforms

from ..config import reader
from ..models.faster_rcnn.faster_rcnn import FasterRCNN
from ._utils import (
    draw_boxes,
    set_text,
    check_file_exists,
    check_dir_exists
)

cfg = reader()
dataset_dir = cfg["dataset_dir"]
model_url = cfg["model_url"]
results_dir = cfg["results_dir"]
device = "cpu" #cfg["device"]

def draw_relation(img, sbj_box, obj_box, sbj, obj, pred):
    color = list(np.random.random(size=3) * 256)
    # write sbj and obj
    centr_sub = (int((sbj_box[0].item() + sbj_box[2].item())/2), int((sbj_box[1].item() + sbj_box[3].item())/2))
    centr_obj = (int((obj_box[0].item() + obj_box[2].item())/2), int((obj_box[1].item() + obj_box[3].item())/2))
    set_text(img, sbj,centr_sub)
    set_text(img, obj,centr_obj)
    # draw line conencting sbj and obj
    cv2.line(img, centr_sub, centr_obj, color, thickness=2)
    predicate_point = (
        int((centr_sub[0] + centr_obj[0])/2), int((centr_sub[1] + centr_obj[1])/2))
    set_text(img, pred, predicate_point)
    # save the drawn image
    path = f"{results_dir}/rel-{opt.image_path.split('/')[-1]}"
    cv2.imwrite(path, img)


def draw_bbox(img, boxes, labels, _ind_to_class):
    for bbox, label in zip(boxes, labels):
        img = draw_boxes(img, bbox, label, _ind_to_class)
    path = f"./{results_dir}/objs-{opt.image_path.split('/')[-1]}"
    cv2.imwrite(path, img)


def infer(image_path):
    faster_rcnn = FasterRCNN().to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    # load dataset
    with open(os.path.join(dataset_dir, 'json_dataset', 'objects.json'), 'r') as f:
        objects = json.load(f)
    with open(os.path.join(dataset_dir, 'json_dataset', 'predicates.json'), 'r') as f:
        predicates = json.load(f)

    classes = deepcopy(objects)
    predicates.insert(0, 'unknown')
    classes.insert(0, '__background__')

    # load pretrained weights
    checkpoint = load_state_dict_from_url(model_url, map_location='cpu')
    faster_rcnn.load_state_dict(checkpoint['state_dict'])
    print("Model Restored")
    faster_rcnn.eval()

    # load image
    im = Image.open(image_path)
    img = np.array(im)
    draw = img.copy()
    draw_rlp = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    draw_objects = draw_rlp.copy()
    im = transform(im)

    with th.no_grad():
        detections, losses = faster_rcnn([im])

    sbj_boxes = detections[0]['sbj_boxes']
    obj_boxes = detections[0]['obj_boxes']
    sbj_labels = detections[0]['sbj_labels']
    obj_labels = detections[0]['obj_labels']
    pred_labels = detections[0]['predicates']
    for sbj_box, obj_box, sbj_label, obj_label, pred  in zip(sbj_boxes, obj_boxes, sbj_labels, obj_labels, pred_labels):
        sbj, obj, pred = objects[sbj_label], objects[obj_label], predicates[pred]
        print(sbj, pred, obj)
        draw_relation(draw_rlp, sbj_box, obj_box, sbj, obj, pred)

    # boxes = detections[0]['boxes']
    # labels = detections[0]['labels']
    # scores = detections[0]['scores']
    # draw_bbox(draw_objects, boxes, labels, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help='path to one image')
    group.add_argument('--image_dir', type=str, help='path to a directory of images')

    opt = parser.parse_args()

    # incase of single image
    if opt.image_path:
        if check_file_exists(opt.image_path):
            infer(opt.image_path)
        else:
            print("File does not exist")

    # incase of directory of images
    elif opt.image_dir:
        if check_dir_exists(opt.image_dir):
            for image_path in os.listdir(opt.image_dir):
                print(f"Image: {image_path}")
                infer(os.path.join(opt.image_dir, image_path))
        else:
            print("Directory does not exist")

    # require an argument
    else:
        print("Please provide either --image_path or --image_dir")



