from vrd import VRD
from config import reader

cfg = reader()
cfg_dataset = cfg["dataset"]
cfg_dataset_path = cfg["dataset_dir"]


def get_training_data(cfg):
    if cfg.DATASET == 'VRD':
        return VRD(cfg.DATASET_DIR, 'train')
    else:
        raise NotImplementedError


def get_validation_data(cfg):
    if cfg.DATASET == 'VRD':
        return VRD(cfg.DATASET_DIR, 'test')
    else:
        raise NotImplementedError

