from skimage.filters import gaussian
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms as transform


def _normalise(x):
    """
    Normalise a ndArray

    Parameters
    ----------
    x: ndArray
        Input array containing dataset to be normalised

    Returns
    -------
    output: ndArray
        Normalised array
    """
    if np.sum(x) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero tensor")

    return x/np.sum(x**2)

def generatePointFunction():
    """
    Generate a image array of a point function with size of defined in config 
    file (excluding padding)
    
    Returns
    -------
    pointFunction: ndArray
        2D image array containing a point function
    """

    pointFunction = np.zeros((NOISE_SIZE, NOISE_SIZE))
    pointFunction[NOISE_SIZE//2, NOISE_SIZE//2] = 1
    return pointFunction

"""
Hyper Parameters
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Directory of files containing image datasets
# TRAIN_DIR = "/home/giorgio/Desktop/p06_images/train/"
TRAIN_DIR = "/home/brunicam/myscratch/p3_scratch/p06_images/train/"
# VAL_DIR = "/home/giorgio/Desktop/p06_images/val/"
VAL_DIR = "/home/brunicam/myscratch/p3_scratch/p06_images/val/"
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
SCHEDULAR_DECAY = 0.5
SCHEDULAR_STEP = 20                         # Step size of learning rate schedular
OPTIMISER_WEIGHTS = (0.5, 0.999)            # Beta parameters of Adam optimiser
NUM_WORKERS = 2
MAX_JITTER = 3
PADDING_WIDTH = 30
IMAGE_SIZE = 256 
NOISE_SIZE = IMAGE_SIZE - PADDING_WIDTH*2
SIGMA = 20                                  # Standard deviation of gaussian kernal for PSF
CHANNELS_IMG = 1                            # Colour channels of input image tensors 
L1_LAMBDA = 100
LAMBDA_GP = 10
CORRELATION_LENGTH = 10
NUM_EPOCHS =  300
LOAD_MODEL = True 
SAVE_MODEL = True

CHECKPOINT_DISC_LOAD = "../models/disc.pth.tar"
CHECKPOINT_GEN_LOAD = "../models/gen.pth.tar"

CHECKPOINT_DISC_SAVE = "../models/disc.pth.tar"
CHECKPOINT_GEN_SAVE = "../models/gen.pth.tar"

MODEL_LOSSES_FILE = "../raw_data/model_losses.txt"
MODEL_LOSSES_TITLES = ["epoch", "disc_loss", "gen_loss"]
EVALUATION_IMAGE_FILE= "../evaluation/default"

CRITIC_SCORE_FILE = "../raw_data/critic_score.txt"
CRITIC_SCORE_TITLES = ["epoch", "disc_real", "disc_fake"]
# WRITER_REAL = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/real")
# WRITER_FAKE = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/fake")

# Evaluation hyperparameters
EVALUATION_EPOCHS = 50
EVALUATION_METRIC_FILE = "../raw_data/sigma.txt"

"""
Tensor Transformations
"""

transforms = transform.Compose([
    transform.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])

transformsFile = transform.Compose([
    transform.ToTensor(),
    transform.RandomCrop(IMAGE_SIZE),
    transform.Grayscale(),
])

"""
Hyperparameter overwriting for automatic bash scripts 
"""
