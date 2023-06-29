# some utility functions used by the datasets modules

import os
import json
from copy import deepcopy

from typing import List


def make_image_list(dataset_path:str, data_type:str) -> List:
    """
    Make a list of images from the dataset

    Args:
        dataset_path (str): path to the dataset
        data_type (str): type of the dataset (train, test)

    Returns:
        list: list of images for train or test
    """
    annotations_file = os.path.join(dataset_path, 'json_dataset', f'annotations_{data_type}.json')
    sg_images_file = os.listdir(os.path.join(dataset_path, 'sg_dataset', f'sg_{data_type}_images'))
    with open(os.path.join(annotations_file), 'r') as f:
        annotations = json.load(f)
    annotations_copy = deepcopy(annotations)
    for ant in annotations.items():
        if(not annotations[ant[0]] or ant[0] not in sg_images_file):
            annotations_copy.pop(ant[0])
    return [ant[0] for ant in annotations_copy.items()]


def load_data(dataset_path:str, data_type:str):
    """
    helper function to load data from the dataset

    Args:
        dataset_path (str): path to the dataset
        data_type (str): type of the dataset (train, test)

    Returns:
        tuple: annotations, all_objects, predicates, sg_images
    """
    with open(os.path.join(dataset_path, 'json_dataset', f'annotations_{data_type}.json'), 'r') as f:
        annotations = json.load(f)
    with open(os.path.join(dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
        all_objects = json.load(f)
    with open(os.path.join(dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
        predicates = json.load(f)
    sg_images = os.path.join(dataset_path, 'sg_dataset', f'sg_{data_type}_images')
    return annotations, all_objects, predicates, sg_images

def image_path(dataset_path:str, data_type:str, img_name:str) -> str:
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(dataset_path, 'sg_dataset', f'sg_{data_type}_images', img_name)
    assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
    return image_path

def change_coords_order(coords):
    """
    Change the order of the coordinates from [x1, y1, x2, y2] to [y1, x1, y2, x2]

    Args:
        coords (list): list of coordinates

    Returns:
        list: list of coordinates with the order changed
    """
    x1 = coords[2]
    y1 = coords[0]
    x2 = coords[3]
    y2 = coords[1]
    return [x1, y1, x2, y2]

