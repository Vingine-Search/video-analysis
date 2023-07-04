# this file is used to train the model

import random
import numpy as np
import torch as th
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config.config import reader
from datasets.loader import (
    get_training_data,
    get_validation_data,
    collater
)

from models.faster_rcnn.faster_rcnn import FasterRCNN

from running_avg import RunningAvg
from progress_bar import ProgressBar
from metrics_logger import MetricsLogger
from _utils import (
    save_model,
    load_from_ckpt,
    load_train_utils,
)

cfg = reader()
cfg_device = cfg["device"]

# train configurations
train_lr = cfg["train"]["learning_rate"]
train_backbone_lr_scalar = cfg["train"]["backbone_lr_scalar"]
train_lr_policy = cfg["train"]["lr_policy"]
train_gamma = cfg["train"]["gamma"]
train_step_size = cfg["train"]["step_size"]
train_steps = cfg["train"]["steps"]
train_lrs = cfg["train"]["lrs"]
train_momentum = cfg["train"]["momentum"]
train_weight_decay = cfg["train"]["weight_decay"]
train_double_bias = cfg["train"]["double_bias"]
train_bias_decay = cfg["train"]["bias_decay"]
train_warm_up_iters = cfg["train"]["warm_up_iters"]
train_warm_up_factor = cfg["train"]["warm_up_factor"]
train_warm_up_method = cfg["train"]["warm_up_method"]
train_scale_momentum = cfg["train"]["scale_momentum"]
train_scale_momentum_threshold = cfg["train"]["scale_momentum_threshold"]
train_log_lr_change_threshold = cfg["train"]["log_lr_change_threshold"]

# other train options
optimizer_type = cfg["opt"]["optimizer_type"]
scheduler_type = cfg["opt"]["scheduler_type"]
weight_path = cfg["opt"]["weight_path"]
num_workers = cfg["opt"]["num_workers"]
batch_size = cfg["opt"]["batch_size"]
begin_iter = cfg["opt"]["begin_iter"]
max_iter = cfg["opt"]["max_iter"]


def val_epoch(model, dataloader):
    """
    Evaluate the model on the given dataloader

    Args:
        model (nn.Module): the model to be evaluated
        dataloader (DataLoader): the dataloader to be evaluated on

    Returns:
        dict: the evaluation/losses results
    """
    model.eval()
    sbj_losses = RunningAvg('Loss', ':.4e')
    obj_losses = RunningAvg('Loss', ':.4e')
    rel_losses = RunningAvg('Loss', ':.4e')
    total_losses = RunningAvg('Loss', ':.4e')
    for _, data in enumerate(dataloader):
        images, targets = data
        with th.no_grad():
            _, metrics = model(images, targets)
        final_loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
            metrics["loss_classifier"] + metrics["loss_box_reg"] + \
            metrics["loss_sbj"] + metrics["loss_obj"] + metrics["loss_rlp"]
        sbj_losses.update(metrics["loss_sbj"].item(), len(images))
        obj_losses.update(metrics["loss_obj"].item(), len(images))
        rel_losses.update(metrics["loss_rlp"].item(), len(images))
        total_losses.update(final_loss.item(), len(images))
    return {
        'sbj_loss': sbj_losses.avg,
        'obj_loss': obj_losses.avg,
        'rel_loss': rel_losses.avg,
        'total_loss': total_losses.avg,
    }

def setup_optimizer(model):
    """
    Setup the optimizer for the given model

    Args:
        model (nn.Module): the model to be optimized

    Returns:
        Optimizer: the optimizer for the given model (SGD or Adam)
    """
    # backbone bias & nonbias params
    backbone_bias_params = []
    backbone_bias_param_names = []
    backbone_nonbias_params = []
    backbone_nonbias_param_names = []
    # prediction branch bias & nonbias params
    prd_branch_bias_params = []
    prd_branch_bias_param_names = []
    prd_branch_nonbias_params = []
    prd_branch_nonbias_param_names = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'fpn' in key or 'box_head' in key or 'box_predictor' in key or 'rpn' in key:
                if 'bias' in key:
                    backbone_bias_params.append(value)
                    backbone_bias_param_names.append(key)
                else:
                    backbone_nonbias_params.append(value)
                    backbone_nonbias_param_names.append(key)
            else:
                if 'bias' in key:
                    prd_branch_bias_params.append(value)
                    prd_branch_bias_param_names.append(key)
                else:
                    prd_branch_nonbias_params.append(value)
                    prd_branch_nonbias_param_names.append(key)
    params = [
        { 'params': backbone_nonbias_params, 'lr': train_lr, 'weight_decay': train_weight_decay },
        {
            'params': backbone_bias_params, 'lr': train_lr * (train_double_bias + 1),
            'weight_decay': train_weight_decay if train_bias_decay else 0
        },
        { 'params': prd_branch_nonbias_params, 'lr': train_lr, 'weight_decay': train_weight_decay },
        {
            'params': prd_branch_bias_params, 'lr': train_lr * (train_double_bias + 1),
            'weight_decay': train_weight_decay if train_bias_decay else 0
        },
    ]
    optimizer = th.optim.Adam(params)
    if optimizer_type == "SGD":
        optimizer = th.optim.SGD(params, lr=train_lr, momentum=train_momentum)
    return optimizer

def setup_scheduler(optimizer):
    """
    Setup the scheduler for the given optimizer

    Args:
        optimizer (Optimizer): the optimizer to be scheduled

    Returns:
        _LRScheduler: the scheduler for the given optimizer (MultiStepLR or StepLR)
    """
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[83631, 111508])
    if scheduler_type == "step_lr":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
    return scheduler

def worker():
    """ The main worker of the training process """
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # load train & val data
    train_data, val_data = get_training_data(), get_validation_data()
    train_loader = DataLoader(
        train_data, 
        num_workers=num_workers,
        collate_fn=collater,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        num_workers=num_workers,
        collate_fn=collater,
        batch_size=batch_size,
        shuffle=True
    )
    print(f"Training dataset size : {len(train_loader)}")
    print(f"Validation dataset size : {len(val_loader)}")

    # set up model params
    dataiterator = iter(train_loader)
    faster_rcnn = FasterRCNN()
    # loading model from a ckpt
    if weight_path:
        load_from_ckpt(weight_path, faster_rcnn)
    faster_rcnn.to(cfg_device)
    print(f"Learning rate : {train_lr}")
    print(f"Weight Decay : {train_weight_decay}")
    
    # set up optimizer & scheduler
    optimizer = setup_optimizer(faster_rcnn)
    scheduler = setup_scheduler(optimizer)
    if weight_path:
        begin_iter = load_train_utils(weight_path, optimizer, scheduler)

    lr = optimizer.param_groups[0]['lr']
    summary_writer = MetricsLogger()
    losses_sbj = RunningAvg('Sbj loss: ', ':.2f')
    losses_obj = RunningAvg('Obj loss: ', ':.2f')
    losses_rel = RunningAvg('Rel loss: ', ':.2f')
    losses_total = RunningAvg('Total loss: ', ':.2f')
    progress = ProgressBar([losses_sbj, losses_obj, losses_rel, losses_total], prefix='Train: ')

    faster_rcnn.train()
    for step in range(begin_iter, max_iter):
        try:
            input_data = next(dataiterator)
        except StopIteration:
            dataiterator = iter(train_loader)
            input_data = next(dataiterator)

        images, targets = input_data
        _, metrics = faster_rcnn(images, targets)
        final_loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
            metrics["loss_classifier"] + metrics["loss_box_reg"] + \
            metrics["loss_sbj"] + metrics["loss_obj"] + metrics["loss_rlp"]

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        losses_sbj.update(metrics["sbj_loss"].item(), len(images))
        losses_obj.update(metrics["obj_loss"].item(), len(images))
        losses_rel.update(metrics["rlp_loss"].item(), len(images))
        losses_total.update(final_loss.item(), len(images))

        if (step) % 10 == 0:
            progress.display(step)

        if step % 2500 == 0:
            train_losses = {}
            train_losses['total_loss'] = losses_total.avg
            train_losses['sbj_loss'] = losses_sbj.avg
            train_losses['obj_loss'] = losses_obj.avg
            train_losses['rel_loss'] = losses_rel.avg
            val_losses = val_epoch(faster_rcnn, val_loader)

            lr = optimizer.param_groups[0]['lr']
            save_model(faster_rcnn, optimizer, scheduler, step)
            # write summary
            summary_writer.log_metrics(train_losses, val_losses, step, lr)
            print(f"* Average training loss : {train_losses['total_loss']:.3f}")
            print(f"* Average validation loss : {val_losses['total_loss']:.3f}")

        losses_sbj.reset()
        losses_obj.reset()
        losses_rel.reset()
        losses_total.reset()
        faster_rcnn.train()


# why having multiple lists for params in setup_optimizer?
# this seperation is needed since object detection tasks
# have 2 branches: backbone and prediction branch
# and different branches may make use of different set of hyperparams
# to better fine-tune the model

# what is the need of optimizer and scheduler?
# optimizer is used to update the weights of the model to minimize the loss functions
# scheduler is used to update the hyperparams of the optimizer over time in a way that
# balances between exploration and exploitation, example of an important hyperparam is
# the learning rate as it controls the step size of the optimizer and 
# can have a significant impact on the convergence of the model.

