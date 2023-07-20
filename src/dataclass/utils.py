import torch
import torch.nn.functional as F

def tensorConcatinate(tensorLeft, tensorRight):
    tensorRight = tensorRight.view(-1, tensorRight.shape[-1])
    tensorLeft = tensorLeft.view(-1, tensorLeft.shape[-1])
    return torch.cat((tensorLeft, tensorRight), dim=1) 

def verticalEdges(inputTensor):
    print(inputTensor.shape)
    print(config.SOBEL_KERNAL.shape)
    return F.conv2d(inputTensor.reshape(1, 1, *inputTensor.shape),
                    config.SOBEL_KERNAL.reshape(1, 1, *config.SOBEL_KERNAL.shape),
                    padding="same")
