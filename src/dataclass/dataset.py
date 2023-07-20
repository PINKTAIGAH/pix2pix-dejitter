import sys
import torch
import utils 
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from JitterFilter import JitterFilter
from torchvision.utils import save_image

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

        groundTruthTorch = torch.tensor(groundTruthNumpy, dtype=torch.float32) 
        jitteredTruthTorch = torch.tensor(jitteredTruthNumpy, dtype=torch.float32) 

        groundTruthTorch.view(-1, 1, self.N, self.N)
        jitteredTruthTorch.view(-1, 1, self.N, self.N)

        if self.concatImages:
            return utils.tensorConcatinate(jitteredTruthTorch, groundTruthTorch)


        return jitteredTruthTorch, groundTruthTorch

if __name__ == "__main__":
    # dataset = JitteredDataset(10, 2)
    # jittered, truth = dataset[0] 
    # print(jittered.shape, truth.shape)

    # """
    dataset = JitteredDataset(256, 2)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(loader.__len__())
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")

        sys.exit()
    # """
