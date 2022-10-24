import sys
import torch.optim as optim
from datetime import datetime
from torchvision.models import AlexNet, resnet18
import torchvision
from models import U_Net
from dataset import data_loader
from metrics import ConfusionMatrix
from model_run import train_epoch, validate
import torch
import torch.nn as nn
import numpy as np
import config
from shutil import copy, copytree
import pathlib
import os
import glob

from utils import Tee, copy_codes, make_log_dir

current_file_path = pathlib.Path(__file__)
log_dir = make_log_dir(config.LOG_DIR, config.METHOD_NAME)
copy_codes(log_dir)
sys.stdout = Tee(os.path.join(log_dir, "print.log"))

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_S

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {DEVICE}")

N_CLASSES = 3

train_loader, valid_loader = data_loader(
    train_cases=list(np.arange(2)),
    valid_cases=list(np.arange(161, 163)),
    batch_size=config.BATCH_PER_GPU, is_normalize=True, is_augment=True)


model = U_Net(N_CLASSES).to(DEVICE)
if config.BASE_MODEL_PATH is not None:
    model.load_state_dict(
        torch.load(
            os.path.join(config.BASE_MODEL_PATH, f"{type(model).__name__}_valid_best.model")))


optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterions = {
    "X_Entropy": (nn.CrossEntropyLoss(), 1.0)
}

conf_matrix = ConfusionMatrix(N_CLASSES, DEVICE)
metrics = {
    "acc": lambda x, y: conf_matrix.get_accuracy(x.argmax(dim=1), y).cpu().numpy()
}

epochs = 3
for epoch in range(0, epochs):
    print(
        f'{datetime.now().time().replace(microsecond=0)} --- '
        f'Epoch: {epoch}\t')

    train_epoch(
        train_loader, model, criterions, metrics, DEVICE, optimizer, print_every=1)

    validate(
        valid_loader, model, criterions, metrics, DEVICE, print_every=1)
