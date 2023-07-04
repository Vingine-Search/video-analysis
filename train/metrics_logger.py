# this class is used to log the metrics to tensorboard, which
# is used to visualize the training process

import tensorboardX
from config.config import reader

cfg = reader()
log_dir = cfg["train"]["log_dir"]

class MetricsLogger():
    """ Log the metrics to tensorboardX """

    def __init__(self):
        self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    def log_metrics(self, train_losses, val_losses, epoch, lr):
        # write train summary
        self. summary_writer.add_scalar('train_losses/train_total_loss', train_losses['total_loss'], global_step=epoch)
        self.summary_writer.add_scalar('train_losses/train_subject_loss', train_losses['sbj_loss'], global_step=epoch)
        self.summary_writer.add_scalar('train_losses/train_object_loss', train_losses['obj_loss'], global_step=epoch)
        self.summary_writer.add_scalar('train_losses/train_relation_loss', train_losses['rel_loss'], global_step=epoch)
        # write validation summary
        self. summary_writer.add_scalar('val_losses/val_total_loss', val_losses['total_loss'], global_step=epoch)
        self.summary_writer.add_scalar('val_losses/val_subject_loss', val_losses['sbj_loss'], global_step=epoch)
        self.summary_writer.add_scalar('val_losses/val_object_loss', val_losses['obj_loss'], global_step=epoch)
        self.summary_writer.add_scalar('val_losses/val_relation_loss', val_losses['rel_loss'], global_step=epoch)
        # write learning rate
        self.summary_writer.add_scalar('lr_rate', lr, global_step=epoch)

