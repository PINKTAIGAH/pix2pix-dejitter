import sys
import torch
import torchvision
import utils 
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from JitterFilter import JitterFilter
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class JitteredDataset(Dataset):
    def __init__(self, N, maxJitter, psfSigma=3, length=100, concatImages=False):
        self.N = N
        self.length = length
        self.maxJitter = maxJitter
        self.Generator = ImageGenerator(self.N)
        self.Filter = JitterFilter()
        self.concatImages = concatImages
        self.psfSigma = psfSigma

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        groundTruthNumpy = self.Generator.genericNoise(sigma=self.psfSigma)
        jitteredTruthNumpy = self.Filter.rowJitter(groundTruthNumpy, self.N,
                                                   self.maxJitter)

        # Image.fromarray(groundTruthNumpy*255).convert("RGB").save("x_np.png")
        # Image.fromarray(jitteredTruthNumpy*255).convert("RGB").save("y_np.png")

        groundTruthTorch = torch.tensor(groundTruthNumpy, dtype=torch.float32) 
        jitteredTruthTorch = torch.tensor(jitteredTruthNumpy, dtype=torch.float32) 

        groundTruthTorch = torch.unsqueeze(groundTruthTorch, 0)
        jitteredTruthTorch = torch.unsqueeze(jitteredTruthTorch, 0)

        if self.concatImages:
            return utils.tensorConcatinate(jitteredTruthTorch, groundTruthTorch)


        return jitteredTruthTorch, groundTruthTorch

if __name__ == "__main__":
    """
    Tensorboard setup
    """
    writerScaled = SummaryWriter("runs/jittered")
    writerUnscaled = SummaryWriter("runs/unjittered")

    N=1000 
    dataset = JitteredDataset(N, 20)
    loader = DataLoader(dataset, batch_size=64)
    sys.exit()
    for x, y in loader:
        
        x = utils.normaliseTensor(x)
        y = utils.normaliseTensor(y)

        save_image(x, "images/Jittered.pdf")
        save_image(y, "images/Unjittered.pdf")

        xGrid = torchvision.utils.make_grid(x, normalize=True)
        yGrid = torchvision.utils.make_grid(y, normalize=True)
        
        writerScaled.add_image("Example of jittered", xGrid)
        writerUnscaled.add_image("Example of unjittered", yGrid)
        
        writerUnscaled.close()
        writerScaled.close()
        sys.exit()
