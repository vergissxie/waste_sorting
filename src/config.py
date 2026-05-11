from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TEST_PATH_FILE = DATA_DIR / "testpath.txt"
GARBAGE_DICT_FILE = DATA_DIR / "garbage_dict.json"

OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
RESULT_FILE = ROOT_DIR / "result.txt"

NUM_CLASSES = 40
FOLDS = 5
SEED = 42

MODEL_NAME = "convnext_tiny"
IMG_SIZE = 256
EVAL_RESIZE_RATIO = 1.14
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
USE_WEIGHTED_CE = True
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIX_PROB = 0.5
EMA = True
EMA_DECAY = 0.999
SAVE_METRIC = "blend"
AMP = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
