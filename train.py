import sys
import torch.optim as optim
from arguments import parse_args
from lovasz_loss import Lovasz_softmax
from models import Res_U_Net, U_Net
from dataset import data_loader, split_cases
from metrics import ConfusionMatrix, dices
from model_run import train_epoch, validate
import torch
import torch.nn as nn
import numpy as np
import config
import pathlib
import os
from utils import Tee, copy_codes, get_format_time, load_model, make_log_dir, save_model, set_random_seed

args, _ = parse_args()

set_random_seed(123)
current_file_path = pathlib.Path(__file__)
log_dir = make_log_dir(args.log, config.METHOD_NAME)
copy_codes(log_dir)
sys.stdout = Tee(os.path.join(log_dir, "print.log"))
np.set_printoptions(precision=4)

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_S

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"""
using {DEVICE} | cuda num {torch.cuda.device_count()}
""")

N_CLASSES = 3

valid_cases, train_cases = split_cases(config.TRAINING_CASES, 30)
train_loader, valid_loader = data_loader(
    train_cases=train_cases,
    valid_cases=valid_cases,
    train_batch_size=config.TRAIN_BATCH_PER_GPU * torch.cuda.device_count(),
    valid_batch_size=config.VALID_BATCH_PER_GPU * torch.cuda.device_count(),
    is_normalize=True, is_augment=True)


model = Res_U_Net(3, N_CLASSES).to(DEVICE)
# model = nn.DataParallel(model)
if config.BASE_MODEL_PATH is not None:
    load_model(
        model=model,
        load_dir=config.BASE_MODEL_PATH,
        suffice="valid_best",
    )


optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterions = {
    "X_Entropy": (nn.CrossEntropyLoss(weight=torch.Tensor([1, 128, 700])).to(DEVICE), 1.5),
    "Lovasz": (Lovasz_softmax().to(DEVICE), 0.7),
}

conf_matrix = ConfusionMatrix(N_CLASSES, DEVICE)
metrics = {
    "acc": lambda pred, proba, y: conf_matrix.get_accuracy(proba, y).cpu().numpy(),
    "Dice": lambda pred, proba, y: dices(proba, y).cpu().numpy()
}

best_train_dice = 0
best_valid_dice = 0
for epoch in range(0, config.MAX_EPOCH):
    print(f"""

Epoch: {epoch} --- {get_format_time()}
""")

    train_loss, train_metrics = train_epoch(
        train_loader, model, criterions, metrics, DEVICE, optimizer, print_every=1)
    dice_now = train_metrics["Dice"].mean()
    if dice_now > best_train_dice:
        print(f"Best Dice for train until now !")
        save_model(model, log_dir, "train_best")
        best_train_dice = dice_now

    valid_loss, valid_metrics = validate(
        valid_loader, model, criterions, metrics, DEVICE, print_every=1)
    dice_now = valid_metrics["Dice"].mean()
    if dice_now > best_valid_dice:
        print(f"Best Dice for validate until now !")
        save_model(model, log_dir, "valid_best")
        best_valid_dice = dice_now
