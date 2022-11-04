
LOG_DIR = "./logs"
METHOD_NAME = "base"

MAX_EPOCH = 150
TRAIN_BATCH_PER_GPU = 34
VALID_BATCH_PER_GPU = 36
GPU_S = "0"
LEARNING_RATE = 1e-4
BASE_MODEL_PATH = None

ALL_CASES = list(range(160)) + list(range(161, 300))
TRAINING_CASES = list(range(160)) + list(range(161, 210))
