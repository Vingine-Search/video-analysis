import torch as th
import numpy as np
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import argparse

from config import reader
from ._utils import check_file_exists, check_dir_exists

cfg = reader()
device = cfg["device"]
coco_names_filepath = cfg["coco_names"]


def read_coco_names():
    if not check_file_exists(coco_names_filepath):
        raise FileNotFoundError("Coco names file not found.")
    with open(coco_names_filepath, "r") as f:
        coco_names = f.read().split("\n")
    return coco_names


def prepare_obj_infer():
    obj_faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    obj_faster_rcnn.eval().to(device)
    return obj_faster_rcnn


def obj_batch_infer(images_dir, obj_faster_rcnn, coco_names):
    obj_transform = transforms.Compose([
        # transforms.Resize((800, 800)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(images_dir, transform=obj_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    boxes_predictions = []
    labels_predictions = []
    for images, _ in dataloader:
        images = images.to(device)
        with th.no_grad():
            outputs = obj_faster_rcnn(images)
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        # Get all the predicted bounding boxes.
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        # Get boxes above the threshold score.
        boxes = pred_bboxes[pred_scores >= 0.5].astype(np.int32)
        labels = outputs[0]['labels'][:len(boxes)]
        # Get all the predicited class names.
        pred_classes = [coco_names[i] for i in labels.cpu().numpy()]
        boxes_predictions.append(boxes)
        labels_predictions.append(pred_classes)
    return boxes_predictions, labels_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir", type=str, 
        help="Path to images directory to infer (images should be in subdirectory called unknown)",
        required=True
    )
    args = parser.parse_args()

    if not args.images_dir:
        raise ValueError("Please provide path to images directory to infer.")

    if not check_file_exists(coco_names_filepath):
        raise FileNotFoundError("Coco names file not found. Adjust it in config.json")

    if not check_dir_exists(args.images_dir):
        raise FileNotFoundError("Images directory not found.")

    unknow_dir_check = os.path.join(args.images_dir, "unknown")
    if not check_dir_exists(args.images_dir + "/unknown"):
        raise FileNotFoundError("Unknown images directory not found.")

    coco_names = read_coco_names()
    obj_faster_rcnn = prepare_obj_infer()
    boxes_predictions, labels_predictions = obj_infer(unknow_dir_check, obj_faster_rcnn, coco_names)
    print(labels_predictions)



