import sys
import torch
import utils 
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
    dataset = JitteredDataset(10, 2)
    jittered, truth = dataset[0] 


    # """
    N=256 
    dataset = JitteredDataset(N, 2)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        # x = x - torch.min(x, 0)
        min_vals, _ = x.view(-1, N*N).min(axis=1)
        min_x = torch.ones_like(x)
        for i in range(min_vals.shape[0]):
            min_x[i] = min_x[i]*min_vals[i]
        
        x = x - min_x

        max_vals, _ = x.view(-1, N*N).max(axis=1)
        max_x = torch.ones_like(x)
        for i in range(max_vals.shape[0]):
            max_x[i] = max_x[i]*max_vals[i]

        x = x/max_x*255 
        print(x)
        save_image(x, "scaled_x.png")
        sys.exit()
    # """
