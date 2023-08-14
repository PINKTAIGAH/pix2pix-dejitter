import torch
import config
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset import JitteredDataset

def shift(input, shiftMap):
    return F.grid_sample(input, shiftMap, mode="bicubic", padding_mode="zeros",
                         align_corners=False)

def get_theta(shift_x, shift_y):
    return torch.tensor([[[1.0, 0, shift_x], [0, 1.0, shift_y]]])

dataset = JitteredDataset(config.IMAGE_SIZE, 1)
_, groundTruth = dataset[0]
groundTruth = torch.unsqueeze(groundTruth, 0)


line = torch.arange(0, config.IMAGE_SIZE-1)
grid_x, grid_y = torch.meshgrid(line, line)
gridmap = torch.unsqueeze(torch.dstack([grid_x, grid_y]), 0)

identity = F.affine_grid(get_theta(0.0, 0.0), groundTruth.size()) 
gap = identity[0, 0, 1, 0] - identity[0, 0, 0, 0]
gridmap = torch.clone(identity)
shiftmap = torch.rand((config.IMAGE_SIZE, config.IMAGE_SIZE))*20
gridmap[:, :, :, 0] += shiftmap*gap

output = shift(groundTruth, gridmap)
 
print(gridmap.shape, shiftmap.shape, output.shape)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(groundTruth[0, 0], cmap="gray")
ax2.imshow(output[0, 0], cmap="gray")
plt.show()

