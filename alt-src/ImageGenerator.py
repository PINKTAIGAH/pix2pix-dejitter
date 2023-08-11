from torch.utils.data import Dataset
from scipy.ndimage import shift
import torch
import config
import utils
import wavelets
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import Pad

class ImageGenerator(Dataset):

    def __init__(self, psf, imageHeight, correlationLength, paddingWidth):
        
        self.psf = psf
        self.ftPsf = torch.fft.fft2(self.psf)
        self.imageHight = imageHeight
        self.correlationLength = correlationLength
        self.pad = Pad(paddingWidth)

    def generateGroundTruth(self):

        whiteNoise = torch.randn(*self.ftPsf.shape)
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        return self.pad(torch.real(groundTruth).type(torch.float32))

    def generateShifts(self):

        # maxJitter = np.random.uniform(0.5, 10)
        # shiftsMap = wavelets.generateShiftMatrix(self.imageHight,
                                    # self.correlationLength,
                                    # maxJitter)
        
        maxJitter = np.random.uniform(1.5, 20)
        return np.random.uniform(-maxJitter, maxJitter, size=self.imageHight)

    def shiftImage(self, input, shiftMatrix, outputTensor=True):
        if not isinstance(input, np.ndarray):
            input.numpy()

        output = np.copy(input)
        for i in range(self.imageHight):
            shifts = shiftMatrix[i]
            output[i, :] = shift(input[i, :], shifts, output=None, 
                                    order=3, mode="constant", cval=0, prefilter=True)
        if outputTensor:
            return torch.from_numpy(output).type(torch.float32)

        return output
    
    """
    def shiftImage(self, input, shiftMatrix, outputTensor=True):
        if not isinstance(input, np.ndarray):
            input.numpy()

        output = np.copy(input)
        for i in range(self.imageHight):
            for j in range(self.imageHight):
                shifts = np.cumsum(shiftMatrix[i])
                output[i, :j] = shift(input[i, :j], shifts[j], output=None, 
                                    order=3, mode="constant", cval=0, prefilter=True)
        if outputTensor:
            return torch.from_numpy(output).type(torch.float32)

        return output
    """

def test():

    filter = ImageGenerator(config.PSF, config.IMAGE_SIZE,
                            config.CORRELATION_LENGTH, config.PADDING_WIDTH)

    groundTruth = filter.generateGroundTruth()
    shiftsMap = filter.generateShifts()
    shifted = filter.shiftImage(groundTruth, shiftsMap)
    unshifted = filter.shiftImage(shifted, -shiftsMap)

    groundTruth = torch.unsqueeze(groundTruth, 0)
    shifted = torch.unsqueeze(shifted, 0)
    unshifted = torch.unsqueeze(unshifted, 0)

    groundTruth = utils.normaliseTensor(groundTruth)
    shifted = utils.normaliseTensor(shifted)
    unshifted = utils.normaliseTensor(unshifted)

    save_image(shifted, "images/test.png", )
    print(groundTruth.shape, shiftsMap.shape)

    fig, (ax1,ax2, ax3) = plt.subplots(1,3)
    ax3.imshow(unshifted[0])
    ax2.imshow(shifted[0])
    ax1.imshow(groundTruth[0])
    plt.show()

    

if __name__ == "__main__":
    test()
