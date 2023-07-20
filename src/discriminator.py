import torch
import torch.nn as nn

"""
Class containing a single convolutional block for the neural networks
"""
class CNNBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride=2):
        super().__init__()
        
        self.convolution = nn.Sequential(
            ### Padding mode is set to false to avoid artefacts
            nn.Conv2d(inChannels, outChannels, kernel_size=4,
                      stride=stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
            return self.convolution(x)


"""
Define discriminator class
"""

### We pass in a concatinated tensor with x and y
class Discriminator(nn.Module):
    def __init__(self, inChannel=3, features=[64, 128, 256, 512]):

        ### input will be a 256p image and go to 26*26 output
        super().__init__()

        ### No batchnorm in initial convolutional block
        self.initialBlock = nn.Sequential(
            ### 2* in channels as we pass the input image allong with ground truth
            nn.Conv2d(inChannel*2, features[0], kernel_size=4, stride=2,
                      padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        ### Layers is a list containing each CNNBlock of the critic model
        layers = []
        inChannels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(inChannels, feature,
                         stride=1 if feature==features[-1] else 2),
            )
            inChannels = feature    # make outchannels inchannels for next layer

        layers.append(
            ### Make the output a single channel wide that contains prob of result
            nn.Conv2d(inChannels, 1, kernel_size=4,
                      stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers) # unpack all elements of layers list

    
    def forward(self, x, y):
        
        x = torch.cat([x,y], dim=1) # Concatinate fake and real input
        
        x = self.initialBlock(x)    # initial convolutional blovk

        return self.model(x)        # remaining convolutional blocks
        


"""
Unit test for discriminator model
"""
def test():
    x = torch.randn((1, 3, 256, 256))       # 1 * 3 channels * 256 * 256 pixels
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    predictions = model(x, y)
    print(predictions.shape)

if __name__ == "__main__":
    test()

