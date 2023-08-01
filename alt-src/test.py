import torch
from time import time
import numpy as np
import matplotlib.pyplot as plt
from kornia.geometry.transform import translate
from scipy.datasets import face

x = torch.tensor(face(gray=True)).type(torch.float32)
x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
y = torch.zeros_like(x)

B, C, H, W = x.shape
shifts = torch.randn((B, 1, H))

shifts = torch.cat([shifts, torch.zeros_like(shifts)], 1)

t1=time()
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
