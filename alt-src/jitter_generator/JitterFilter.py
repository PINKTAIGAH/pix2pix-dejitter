from torch.utils.data import Dataset
import scipy.ndimage as ndimg
import torch
import config
import utils
import matplotlib.pyplot as plt

def generate_ground_truth(ft_psf,):
    white_noise = torch.randn(*ft_psf.shape)
    ground_truth = torch.fft.ifft2(ft_psf * torch.fft.fft2(white_noise))  
    return torch.real(ground_truth), white_noise

def generate_shifts(max_jitter, image_hight):
    return torch.randn(image_hight-1)*max_jitter

"""
This is subject to change
"""

def shift_image(image, shifts):
    if len(image.shape) < 2:
        raise Exception("Can only tensors with a minimum of 2 dimentions") 

    total_shifts = torch.hstack([torch.tensor([0]), torch.cumsum(shifts, dim=0)])
    image_hight, image_width = image.shape[-2:]
    shifted_image = torch.zeros_like(image)

    image, shifted_image, total_shifts = image.numpy(), shifted_image.numpy(), total_shifts.numpy() 

    for i, shift in enumerate(shifts):
        shifted_image[i,:] = ndimg.shift(image[i,:], shift, output=None, order=3,
                                   cval=0.0, mode="wrap", prefilter=True)
    return torch.from_numpy(shifted_image)
        
def unshift_image(image, shifts):
    if len(image.shape) < 2:
        raise Exception("Can only tensors with a minimum of 2 dimentions") 

    total_shifts = torch.hstack([torch.tensor([0]), torch.cumsum(shifts, dim=0)])
    image_hight, image_width = image.shape[-2:]
    shifted_image = torch.zeros_like(image)

    image, shifted_image, total_shifts = image.numpy(), shifted_image.numpy(), total_shifts.numpy() 

    for i, shift in enumerate(shifts):
        shifted_image[i,:] = ndimg.shift(image[i,:], -shift, output=None, order=3,
                                   cval=0.0, mode="wrap", prefilter=True)

    return torch.from_numpy(shifted_image)
