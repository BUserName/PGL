import datetime
import time
import torch
import os
import numpy as np
try:
    import tensorboardX as tb
    from tensorboardX import SummaryWriter
except ImportError:
    print("tensorboardX is not installed")
    tb = None



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class

        self.avg = np.zeros(num_class)
        self.sum = np.zeros(num_class)
        self.count = np.zeros(num_class)

    def reset(self):

        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, class_idx, val, n=1):
        for i, value in enumerate(class_idx):
            self.sum[value] += val[i] * n
            self.count[value] += n
            self.avg[value] = self.sum[value] / self.count[value]


class Logger():
    def __init__(self, args):
        # tensorboard writer
        self.args = args
        self._writer = None
        self._writer = self.get_writer()
        self._stats_scalar = {}
        # time stamp
        self._last_logged = time.time()
        self.global_step = 0


    def get_writer(self):
        if self._writer is None:
            # subfolder exp + time
            sub_folder_name = datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')
            if self.args.experiment:
                sub_folder_name = self.args.experiment + sub_folder_name
            log_dir = os.path.join(self.args.logs_dir, sub_folder_name)
            _writer = SummaryWriter(log_dir)

        return _writer


    def nvar(self, x):
        # convert torch tensor to numpy
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            x = x.item() if x.dim() == 0 else x.numpy()
        return x

    def log_scalar(self, tag, value, global_step=None):
        self._writer.add_scalar(tag, self.nvar(value), global_step)

