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

from ..config.config import reader
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
    res = []
    for i, img in enumerate(images_paths_list):
        print(i + 1)
        img = np.array(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY))
        res.append(reader.readtext(img, paragraph=True, detail=0))
    return res

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



