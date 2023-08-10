import config
import utils
from dataset import JitteredDataset
from ImageGenerator import ImageGenerator
import matplotlib.pyplot as plt

def generateShiftedImage(input, filter):
    shifts = filter.generateShifts()
    return filter.shiftImage(input[0], shifts)

def test():
    dataset = JitteredDataset(config.IMAGE_SIZE, 1000)
    filter = ImageGenerator(config.PSF, config.IMAGE_SIZE, 
                            config.CORRELATION_LENGTH, config.PADDING_WIDTH)

    
    _, img = dataset[0]

    shiftedImg = [generateShiftedImage(img, filter) for _ in range(9)]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) 
    
    for i in range(len(shiftedImg)):
        axes[i].imshow(shiftedImg[i], cmap="gray")
    plt.show()

if __name__ == "__main__":
    test()
