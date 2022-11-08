
LOG_DIR = "./logs"
METHOD_NAME = "base"

MAX_EPOCH = 150
TRAIN_BATCH_PER_GPU = 44
VALID_BATCH_PER_GPU = 50
GPU_S = "5"
LEARNING_RATE = 5e-4
BASE_MODEL_PATH = './logs/2022-11-08-05:24:20base'

ALL_CASES = list(range(160)) + list(range(161, 300))
TRAINING_CASES = list(range(160)) + list(range(161, 210))
