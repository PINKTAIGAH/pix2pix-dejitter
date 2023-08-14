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

    def wavelet(self, x, x_0=0.0, std=1.0):
        return np.exp(-(x-x_0)**2/(2*std**2))
    
    def generateSignal(self, x):
        frequency, phase = np.random.uniform(), np.random.uniform(0, 2*np.pi)
        return np.sin(2*np.pi*frequency*x + phase)

    def generateShiftMap(self):
        
        shiftMap = np.empty((self.imageHight, self.imageHight))
        waveletCenters = np.arange(0, self.imageHight, self.correlationLength*3)

        for i in range(self.imageHight):
            x = np.arange(self.imageHight)
            yFinal = np.zeros_like(x, dtype=np.float64)
            for _, val in enumerate(waveletCenters):
                y = self.generateSignal(x)
                yWavelet = self.wavelet(x, val, self.correlationLength)
                yFinal += utils.adjustArray(y * yWavelet)
            shiftMap[i] = yFinal
        return torch.from_numpy(shiftMap)

    """
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
    shiftMap = filter.generateShiftMap()

    """
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
    """
    x = np.arange(config.IMAGE_SIZE)
    fig, (ax1,ax2, ax3) = plt.subplots(3,1)
    ax3.scatter(x, shiftMap[0])
    ax2.scatter(x, shiftMap[2])
    ax1.scatter(x, shiftMap[4])
    plt.show()
    
    

if __name__ == "__main__":
    test()
