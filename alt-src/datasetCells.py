
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

    def __init__(self, imageSize, maxJitter, transform, ):
        self.N = imageSize 
        self.maxJitter = maxJitter
        self.filter = ImageGenerator(config.PSF, config.MAX_JITTER, 28)
        self.transform = transform
        self.dataset = PCAM("/home/giorgio/Desktop/", download=True,
                       transform=self.transform)

    def __len__(self):
        return len(self.dataset) 

    def __getitem__(self, idx):
        
        groundTruth, _ = self.dataset[idx]
        groundTruth = torch.squeeze(groundTruth, 0)

        shifts = self.filter.generateShifts()
        shiftedImage = self.filter.shiftImage(groundTruth, shifts,)


        shiftedImage = torch.unsqueeze(shiftedImage, 0)
        shiftedImage = utils.normaliseTensor(shiftedImage)
        shiftedImage = config.transforms(shiftedImage)

        return shiftedImage, groundTruth 



def test():
    filter = dataclass(None, None, config.transformsPcam)
    x, y = filter[1]

    save_image(x, "images/x.png")
    save_image(y, "images/y.png")
    
if __name__ == "__main__":
    test()
