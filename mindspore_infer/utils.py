import sys
import datetime
import glob
import os
from shutil import copy
import numpy as np
import random
import mindspore


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return self.avg


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    mindspore.set_seed(seed)


def get_format_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")