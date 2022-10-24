import sys
import datetime
import glob
import os
from shutil import copy
import numpy as np
import random
import torch


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
    torch.manual_seed(seed)


def make_log_dir(log_dir: str, name: str):
    new_log_dir = os.path.join(
        log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + name)
    if os.path.isdir(new_log_dir):
        raise Exception(f"{new_log_dir} already exist ! abort ...")
    os.makedirs(new_log_dir)
    return new_log_dir


def copy_codes(log_dir):
    files = glob.glob("*.py")
    print(f"Copying {files} to {log_dir} for further reference.")
    code_dir = os.path.join(log_dir, "code")
    if os.path.isdir(code_dir):
        raise Exception(f"{code_dir} already exist ! abort ...")
    os.makedirs(code_dir)
    for f in files:
        copy(f, code_dir)


class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
