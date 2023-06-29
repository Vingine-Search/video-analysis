# these are some utiltity functions for training and testing scripts.

import os
import torch as th
from config import reader

cfg = reader()
snapshots_dir = cfg['snapshots_dir']


def load_from_ckpt(weight_path, model):
    """ Loading model from checkpoint. """
    checkpoint = th.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("** Loaded model **")


def load_train_utils(weight_path, optimizer, scheduler) -> int:
    """
    loading optmizer, scheduler.

    Args:
        weight_path: path to the checkpoint.
        optimizer: optimizer object.
        scheduler: scheduler object.

    Returns:
        step: current step of the training.
    """
    checkpoint = th.load(weight_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step = checkpoint['step'] + 1
    print(" ** Loaded optmizer and scheduler ** ")
    return step


def save_model(model, optimizer, scheduler, step):
    """ Saving model and train_utils. """
    state = {
        'step': step,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    th.save(state, os.path.join(snapshots_dir, f'large_scale_vrd_iter-{step}.pth'))


