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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {DEVICE}")

BATCH_SIZE = 32
N_CLASSES = 3
model_dir = './model/'
IS_LOAD_MODEL = False


train_loader, valid_loader = data_loader(
    train_cases=list(np.arange(50)),
    valid_cases=list(np.arange(161, 180)),
    batch_size=BATCH_SIZE, is_normalize=True, is_augment=True)


model = U_Net(N_CLASSES).to(DEVICE)
MODEL_PATH = f'{model_dir}{type(model).__name__}.model'
if IS_LOAD_MODEL:
    model.load_state_dict(torch.load(MODEL_PATH))


LEARNING_RATE = 1e-4
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

conf_matrix = ConfusionMatrix(N_CLASSES, DEVICE)
metrics = {"acc": lambda x, y: conf_matrix.getacc(x.argmax(dim=1), y)}

epochs = 3
is_train = True
is_validate = True
for epoch in range(0, epochs):
    print(
        f'{datetime.now().time().replace(microsecond=0)} --- '
        f'Epoch: {epoch}\t')

    if is_train:
        train_epoch(
            train_loader, model, criterion, metrics, DEVICE, optimizer, print_every=1)

    if is_validate:
        validate(
            valid_loader, model, criterion, metrics, DEVICE, print_every=1)
