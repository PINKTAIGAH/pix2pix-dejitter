from skimage.filters import gaussian
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms as transform


def normalise(x):
    if np.sum(x) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero tensor")

    return x/np.sum(x**2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TRAIN_DIR = "/home/giorgio/Desktop/cell_dataset/train/"
TRAIN_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/train/"
#VAL_DIR = "/home/giorgio/Desktop/cell_dataset/val/"
VAL_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/val/"
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
SCHEDULAR_DECAY = 0.5
SCHEDULAR_PATIENCE = 20
NUM_WORKERS = 2
PADDING_WIDTH = 30
IMAGE_SIZE = 256 
NOISE_SIZE = IMAGE_SIZE - PADDING_WIDTH*2
SIGMA = 20
CHANNELS_IMG = 1 
L1_LAMBDA = 100
LAMBDA_GP = 10
CORRELATION_LENGTH = 10
NUM_EPOCHS =  1000
LOAD_MODEL = False 
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
# WRITER_REAL = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/real")
# WRITER_FAKE = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/fake")

kernal = np.zeros((NOISE_SIZE, NOISE_SIZE))
kernal[NOISE_SIZE//2, NOISE_SIZE//2] = 1
PSF = torch.from_numpy(normalise(gaussian(kernal, SIGMA)))

transforms = transform.Compose([
    transform.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])

transformsCell = transform.Compose([
    transform.ToTensor(),
    transform.Grayscale(),
])
