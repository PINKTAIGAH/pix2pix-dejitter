import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
TRAIN_DIR = "/media/giorgio/HDD/GAN/pix2pix/datasets/maps/maps/train/"
VAL_DIR = "/media/giorgio/HDD/GAN/pix2pix/datasets/maps/maps/val/"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
N_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMAGE = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
N_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True 
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
SOBEL_KERNAL = torch.tensor(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32
)

bothTransform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transformOnlyInput = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transformOnlyMask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
