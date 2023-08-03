import torch
import sys
import config
from time import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from kornia.geometry.transform import translate
from scipy.datasets import face
from datasetCells import CellDataset

def shift(image, shifts):
    y = torch.zeros_like(image)
    for k in range(H):
        shift = shifts[:,k, :] 
        print(y[:, :, k, :].shape , shift.shape, x.shape)
        y[:, :, k, :] = translate(x, shift, padding_mode="zeros",
                                  align_corners=True)[:, :, k, :]
    return y 

def alt_shift(image, vectors):
    output = torch.zeros_like(image)
    shifts = torch.clone(vectors)
    for i in range(image.shape[0]):
        single_image = torch.unsqueeze(torch.clone(image[i]), 0)
        single_shift = torch.clone(shifts[i])
        for j in range(x.shape[-2]):
            print(single_shift[j])
            output[i, :, j, :] = translate(single_image[:, :, j, :],
                                      torch.unsqueeze(single_shift[j], 0),
                                      padding_mode="reflection", align_corners=False)
    return output 

dataset = CellDataset("/home/giorgio/Desktop/cell_dataset/val/",
                      config.IMAGE_SIZE, config.MAX_JITTER, config.transformsCell)

"""
_, x = dataset[0]
y = translate(x, torch.tensor([[20.0, 0.0]]), padding_mode="zeros")
z = translate(y, -torch.tensor([[20.0, 0.0]]), padding_mode="zeros")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(torch.squeeze(x, 0), cmap="gray")
ax2.imshow(torch.squeeze(y, 0), cmap="gray")
ax3.imshow(torch.squeeze(z, 0), cmap="gray")
plt.show()
"""





loader = DataLoader(dataset, batch_size=3, )  

_, x = next(iter(loader))
B, C, H, W = x.shape
shifts = torch.randn((B, H, 1))
shifts = torch.cat([shifts, torch.zeros_like(shifts)], 2) * config.MAX_JITTER

shifts2 = torch.rand((B, H, 1))
shifts2 = torch.cat([shifts2, torch.zeros_like(shifts2)], 2) * config.MAX_JITTER

t1=time()
# y = shift(x, shifts)
y = alt_shift(x, shifts)
print(f"Time to jitter : {time()-t1} s\n\n")
shifted_image = torch.clone(y)
unshifted = alt_shift(y, -shifts)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
ax1.imshow(torch.squeeze(x[0], 0), cmap="gray")
ax2.imshow(torch.squeeze(y[0], 0), cmap="gray")
ax3.imshow(torch.squeeze(unshifted[0], 0), cmap="gray")
ax4.imshow(torch.squeeze(x[1], 0), cmap="gray")
ax5.imshow(torch.squeeze(y[1], 0), cmap="gray")
ax6.imshow(torch.squeeze(unshifted[1], 0), cmap="gray")
ax7.imshow(torch.squeeze(x[2], 0), cmap="gray")
ax8.imshow(torch.squeeze(y[2], 0), cmap="gray")
ax9.imshow(torch.squeeze(unshifted[2], 0), cmap="gray")
plt.axis("off")
plt.show()

sys.exit()

x = torch.tensor(face(gray=True)).type(torch.float32)
x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
y = torch.zeros_like(x)

B, C, H, W = x.shape
shifts = torch.randn((B, 1, H))

shifts = torch.cat([shifts, torch.zeros_like(shifts)], 1)

t1=time()
for _ in range(16):    
    for idx in range(H):
        shift = shifts[:,:, idx] * 10 
        y[:, :, idx, :] = translate(x[:, :, idx, :], shift,
                                padding_mode="reflection", align_corners=False)
print(f"Time to jitter : {time()-t1} s\n\n")

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(torch.squeeze(torch.squeeze(x, 0), 0), cmap="gray")
ax2.imshow(torch.squeeze(torch.squeeze(y, 0), 0), cmap="gray")
plt.axis("off")
plt.show()

