import sys
from models import Res_U_Net, U_Net
from dataset import data_loader, split_cases
from metrics import ConfusionMatrix, dices
from model_run import validate
import numpy as np
import config
import pathlib
import os
from utils import get_format_time, set_random_seed
from mindspore import context
import mindspore


set_random_seed(123)
current_file_path = pathlib.Path(__file__)
np.set_printoptions(precision=4)

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_S
context.set_context(device_target="GPU")

N_CLASSES = 3

valid_cases, train_cases = split_cases(config.TRAINING_CASES, 30)
valid_loader = data_loader(
    train_cases=train_cases,
    valid_cases=valid_cases,
    train_batch_size=config.TRAIN_BATCH_PER_GPU,
    valid_batch_size=config.VALID_BATCH_PER_GPU,
    is_normalize=True, is_augment=True)


model = Res_U_Net(3, N_CLASSES)
print(f"loading model from {type(model).__name__}.ckpt")
mindspore.load_checkpoint(
    f"{type(model).__name__}.ckpt",
    net=model,
    strict_load=True,
)


criterions = {
}

conf_matrix = ConfusionMatrix(N_CLASSES)
metrics = {
    "acc": lambda pred, proba, y: conf_matrix.get_accuracy(proba, y).asnumpy(),
    "Dice": lambda pred, proba, y: dices(proba, y).asnumpy()
}

for epoch in range(0, config.MAX_EPOCH):
    print(f"""

Epoch: {epoch} --- {get_format_time()}
""")

    valid_loss, valid_metrics = validate(
        valid_loader, model, criterions, metrics, print_every=1)

