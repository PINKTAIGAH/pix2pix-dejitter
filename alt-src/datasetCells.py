
import sys
from skimage import transform
import torch
import utils
import config
from torchvision.datasets import MNIST, PCAM
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from torchvision.utils import save_image

class dataclass(Dataset):

    def __init__(self, imageSize, length, maxJitter, transform, ):
        self.N = imageSize 
        self.length = length
        self.maxJitter = maxJitter
        self.filter = ImageGenerator(config.PSF, config.MAX_JITTER, 28)
        self.transform = transform
        self.dataset = PCAM("/home/brunicam/myscratch/p3_scratch", download=True,
                       transform=self.transform)

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        
        groundTruth, _ = self.dataset[idx]
        groundTruth = torch.squeeze(groundTruth, 0)

        shifts = self.filter.generateShifts()
        shiftedImage = self.filter.shiftImage(groundTruth, shifts,)


        shiftedImage = torch.unsqueeze(shiftedImage, 0)
        groundTruth = torch.unsqueeze(groundTruth, 0)

        shiftedImage = utils.normaliseTensor(shiftedImage)
        groundTruth = utils.normaliseTensor(groundTruth)
        
        shiftedImage = config.transforms(shiftedImage)
        groundTruth = config.transforms(groundTruth)
        
        print(shiftedImage.shape, groundTruth.shape)

        return shiftedImage, groundTruth 



def test():
    filter = dataclass(96, 10, 1.0, config.transformsPcam)
    x, y = filter[3]

    save_image(x, "images/x.png")
    save_image(y-x, "images/y.png")
    
if __name__ == "__main__":
    test()
