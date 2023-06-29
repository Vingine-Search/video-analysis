# this class is used to display a progress bar in the terminal can be used during training

class ProgressBar(object):
    """ Display a progress bar in the terminal can be used during training """

    def __init__(self, meters, prefix=""):
        self.batch_fmtstr = '[{}]'
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

