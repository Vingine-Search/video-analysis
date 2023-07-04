import torch as th
import numpy as np
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import argparse
import cv2
import easyocr

from config import reader
from ._utils import check_file_exists, check_dir_exists
from ._utils import ( 
    check_file_exists,
    check_dir_exists,
    sort_helper
)

cfg = reader()
device = cfg["device"]


def prepare_easyocr():
    use_gpu = device == "cuda"
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    return reader


def easyocr_infer(images_paths_list, reader):
    images_paths_list = sorted(images_paths_list, key=lambda x: sort_helper(x))
    images = [cv2.imread(img) for img in images_paths_list]
    images_grayscaled = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    npimages = np.array(images_grayscaled)
    return [reader.readtext(i, paragraph=True, detail=0, batch_size=4) for i in npimage]


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

    if not check_dir_exists(args.images_dir):
        raise FileNotFoundError("Images directory not found.")

    unknow_dir_check = os.path.join(args.images_dir, "unknown")
    if not check_dir_exists(args.images_dir + "/unknown"):
        raise FileNotFoundError("Unknown images directory not found.")

    reader = prepare_easyocr()
    labels = easyocr_infer(unknow_dir_check, reader)



