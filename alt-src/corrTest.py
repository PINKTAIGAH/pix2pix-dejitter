import config
import utils
from dataset import JitteredDataset
from ImageGenerator import ImageGenerator
import matplotlib.pyplot as plt
import numpy as np

def generateShiftedImage(input, filter, maxJitter):
    shifts = generateShifts(maxJitter)
    return filter.shiftImage(input[0], shifts)

def generateShifts(maxJitter):

    return np.random.uniform(-maxJitter, maxJitter, size=config.IMAGE_SIZE)

def test():
    dataset = JitteredDataset(config.IMAGE_SIZE, 1000)
    filter = ImageGenerator(config.PSF, config.IMAGE_SIZE, 
                            config.CORRELATION_LENGTH, config.PADDING_WIDTH)

    
    _, img = dataset[0]

    maxJitterArray = np.arange(0, 9, 0.25) 
    shiftedImg = [generateShiftedImage(img, filter, maxJitterArray[i]) for i in range(maxJitterArray.shape[0])]
    imgSimilarity = []

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) 
    
    for i in range(len(shiftedImg)):
        if i <= 8:    
            axes[i].imshow(shiftedImg[i], cmap="gray")
        imgSimilarity.append(utils.corrImage(img[0], shiftedImg[i]))
    plt.show()
    plt.plot(imgSimilarity)
    plt.show()


if __name__ == "__main__":
    test()
