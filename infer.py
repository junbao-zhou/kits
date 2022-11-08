from models import Res_U_Net, U_Net
from dataset import data_loader, split_cases
from metrics import ConfusionMatrix, dices
from model_run import test, test_from_file
import torch
import numpy as np
import config
import os
from utils import load_model, set_random_seed


set_random_seed(123)
np.set_printoptions(precision=4)

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_S

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"""
using {DEVICE} | cuda num {torch.cuda.device_count()}
""")

N_CLASSES = 3

valid_cases, train_cases = split_cases(config.TRAINING_CASES, 30)


model = Res_U_Net(3, N_CLASSES).to(DEVICE)
# model = nn.DataParallel(model)
if config.BASE_MODEL_PATH is not None:
    load_model(
        model=model,
        load_dir=config.BASE_MODEL_PATH,
        suffice="valid_best",
    )

conf_matrix = ConfusionMatrix(N_CLASSES, DEVICE)
metrics = {
    "acc": lambda pred, proba, y: conf_matrix.get_accuracy(proba, y).cpu().numpy(),
    "Dice": lambda pred, proba, y: dices(proba, y).cpu().numpy()
}

test_cases = valid_cases[:10]
test(
    cases=test_cases,
    model=model,
    batch_size=config.VALID_BATCH_PER_GPU,
    device=DEVICE,
    metrics=metrics,
    output_dir="output_data",
)

test_from_file(
    cases=test_cases,
    batch_size=config.VALID_BATCH_PER_GPU,
    metrics=metrics,
    output_dir="mindspore_infer/ascend_data/output_data",
)
