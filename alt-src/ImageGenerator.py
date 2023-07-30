from torch.utils.data import Dataset
import scipy.ndimage as ndimg
import torch
import config
import utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class ImageGenerator(object):

    def __init__(self, psf, maxJitter, imageHeight):
        
        self.psf = psf
        self.ftPsf = torch.fft.fft2(self.psf)
        self.maxJitter = maxJitter
        self.imageHight = imageHeight

    def generateGroundTruth(self):

        whiteNoise = torch.randn(*self.ftPsf.shape)
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        return torch.real(groundTruth), whiteNoise

    def generateShifts(self):
        return torch.randn(self.imageHight-1)*self.maxJitter

    
    def shiftImage(self, image, shifts):
        if len(image.shape) < 2:
            raise Exception("Can only tensors with a minimum of 2 dimentions") 

        totalShifts = torch.hstack([torch.tensor([0]), torch.cumsum(shifts, dim=0)])
        imageHight, imageWidth = image.shape[-2:]
        shiftedImage = torch.zeros_like(image)

        image = image.numpy() 
        shiftedImage = shiftedImage.numpy()
        totalShifts = totalShifts.numpy()

        for i, shift in enumerate(totalShifts):
            shiftedImage[i,:] = ndimg.shift(image[i,:], shift, output=None, order=3,
                                   cval=0.0, mode="wrap", prefilter=True)
        return torch.from_numpy(shiftedImage)

    def unshiftImage(self, image, shifts):
        if len(image.shape) < 2:
            raise Exception("Can only tensors with a minimum of 2 dimentions") 

        totalShifts = torch.hstack([torch.tensor([0]), torch.cumsum(shifts, dim=0)])
        imageHight, imageWidth = image.shape[-2:]
        shiftedImage = torch.zeros_like(image)

        image = image.numpy() 
        shiftedImage = shiftedImage.numpy()
        totalShifts = totalShifts.numpy()

        for i, shift in enumerate(totalShifts):
            shiftedImage[i,:] = ndimg.shift(image[i,:], -shift, output=None, order=3,
                                   cval=0.0, mode="wrap", prefilter=True)
        return torch.from_numpy(shiftedImage)

def test():

    filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE,)

    groundTruth, whiteNoise = filter.generateGroundTruth()
    shifts = filter.generateShifts()
    shiftedImage = filter.shiftImage(groundTruth, shifts)

    groundTruth = torch.unsqueeze(groundTruth, 0)
    whiteNoise = torch.unsqueeze(whiteNoise, 0)
    shiftedImage = torch.unsqueeze(shiftedImage, 0)

    whiteNoise = utils.normaliseTensor(whiteNoise)
    shiftedImage = utils.normaliseTensor(shiftedImage)
    groundTruth = utils.normaliseTensor(groundTruth)

    save_image(shiftedImage, "test.png", )

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(groundTruth[0])
    ax2.imshow(shiftedImage[0])
    print(whiteNoise.min().item(), whiteNoise.max().item())
    print(groundTruth.min().item(), groundTruth.max().item())
    print(shifts)
    plt.show()

    

if __name__ == "__main__":
    test()
