import os

from utils import str_to_bool

LR = 0.001
EPOCHS = os.getenv("EPOCHS", 5)
_BATCHSIZE = 64
R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
BUFFER = 256
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001
NUM_IMAGES = {"train": 1_281_167, "validation": 50000}
NUM_TRAIN_FILES = 1024
SHUFFLE_BUFFER = 100

DATA_LENGTH = int(
    os.getenv("FAKE_DATA_LENGTH", 1_281_167)
)  # How much fake data to simulate, default to size of imagenet dataset

DATASET_NAME = "ImageNet"

DISTRIBUTED = str_to_bool(os.getenv("DISTRIBUTED", "False"))