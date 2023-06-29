from vrd import VRD
from config import reader

cfg = reader()
cfg_device = cfg["device"]
cfg_dataset = cfg["dataset"]
cfg_dataset_path = cfg["dataset_dir"]


def get_training_data():
    if cfg_dataset == 'VRD':
        return VRD(cfg.dataset_dir, 'train')
    else:
        raise NotImplementedError


def get_validation_data():
    if cfg_dataset == 'VRD':
        return VRD(cfg.dataset_dir, 'test')
    else:
        raise NotImplementedError

def collater(data):
    """ 
    collater for training data 

    Args:
        data (list): list of dict with keys 'img', 'boxes', 'labels', 'preds'

    Returns:
        imgs (list): list of images
        annotations (list): list of dict with keys 'boxes', 'labels', 'preds'
    """
    imgs = [s['img'] for s in data]
    annotations = [{"boxes": s['boxes'].to(cfg_device)} for s in data]
    for i, s in enumerate(data):
        annotations[i]['labels'] = s['labels'].to(cfg_device)
    for i, s in enumerate(data):
            annotations[i]['preds'] = s['preds'].to(cfg_device)
    return imgs, annotations
