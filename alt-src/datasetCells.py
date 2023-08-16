from skimage import transform
import numpy as np
from PIL import Image
import os
import torch
import utils
import config
from torchvision.datasets import MNIST, PCAM
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class CellDataset(Dataset):
    def __init__(self, rootDirectory, imageSize, maxJitter, transform):
        self.rootDirectory = rootDirectory
        self.listFiles = os.listdir(self.rootDirectory)
        self.N = imageSize 
        self.maxJitter = maxJitter
        self.filter = ImageGenerator(config.PSF, config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                                     config.PADDING_WIDTH, config.MAX_JITTER)
        self.transform = transform

    def __len__(self):
        return len(self.listFiles)

    def getImage(self, index):

        imageFile = self.listFiles[index]
        imagePath = os.path.join(self.rootDirectory, imageFile)
        image = Image.open(imagePath)
        image = self.transform(image)
        return image

    def __getitem__(self, index):

        groundTruth = self.getImage(index)
        groundTruth = torch.squeeze(groundTruth, 0)

        flowMapShift, flowMapUnshift , _ = self.filter.generateFlowMap()
        shifted = self.filter.shift(groundTruth, flowMapShift)

        groundTruth = torch.unsqueeze(groundTruth, 0)
        shifted = torch.squeeze(shifted, 0)

        groundTruth = utils.normaliseTensor(groundTruth)
        shifted = utils.normaliseTensor(shifted)

        groundTruth = config.transforms(groundTruth)
        shifted = config.transforms(shifted)

        return shifted, groundTruth

def test():
    dataset = CellDataset(config.VAL_DIR,
                          config.IMAGE_SIZE, config.MAX_JITTER, config.transformsCell)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(len(dataset)):
        x, y, = dataset[i]
        if i == 0:
            ax1.imshow(x[0], animated=True)
            ax2.imshow(y[0], animated=True)
        else:
            plt.cla()
            ax1.imshow(x[0], animated=True)
            ax2.imshow(y[0], animated=True)
            plt.draw()
            plt.pause(0.5)
        print(i)
    

if __name__ == "__main__":
    test()
