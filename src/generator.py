import torch
import torch.nn as nn

"""
Write generic convolution block used for the upscaling and downscaling of the 
encoder/decoder sections
"""

class Block(nn.Module):
    
    def __init__(self, inChannel, outChannels, down=True,
                 activation="relu", useDropout=False):
    ### Down refers to if we are in the encoding or decoding phase os the u-net
        super().__init__()
        self.convolution = nn.Sequential(
            
            nn.Conv2d(inChannel, outChannels, kernel_size=4, stride=2,
                      padding=1, bias=False , padding_mode="reflect")\
            if down ### Use conv2d if in encoder section
            else nn.ConvTranspose2d(inChannel, outChannels, kernel_size=4,
                                    stride=2, padding=1, bias=False),
            ### Use convtransose2d if in decoder section 
            
            nn.BatchNorm2d(outChannels),
            nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2),
        )
        self.useDropout = useDropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.convolution(x)
        return self.dropout(x) if self.useDropout else x

"""
Define the generator class
"""

class Generator(nn.Module):
    def __init__(self, inChannels, features=64):
        super().__init__()
        self.initialDown = nn.Sequential(
            nn.Conv2d(inChannels, features, kernel_size=4, stride=2, 
                      padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
        )   # 128

        self.down1 = Block(features, features*2, down=True,
                           activation="leaky", useDropout=False)    # 64
        self.down2 = Block(features*2, features*4, down=True,
                           activation="leaky", useDropout=False)    # 32
        self.down3 = Block(features*4, features*8, down=True,
                           activation="leaky", useDropout=False)    # 16
        self.down4 = Block(features*8, features*8, down=True,
                           activation="leaky", useDropout=False)    # 8
        self.down5 = Block(features*8, features*8, down=True,
                           activation="leaky", useDropout=False)    # 5
        self.down6 = Block(features*8, features*8, down=True,
                           activation="leaky", useDropout=False)    # 2

        # Final layer of the encoder
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, 
                      padding=1, padding_mode="reflect"),
            nn.ReLU(),
        )   # 1*1

        self.up1 = Block(features*8, features*8, down=False,
                         activation="relu", useDropout=True)
        ### Input features *2 due to linked therefore gets concatinated w/ down
        self.up2 = Block(features*8*2, features*8, down=False,
                         activation="relu", useDropout=True)
        self.up3 = Block(features*8*2, features*8, down=False,
                         activation="relu", useDropout=True)
        self.up4 = Block(features*8*2, features*8, down=False,
                         activation="relu", useDropout=False)
        self.up5 = Block(features*8*2, features*4, down=False,
                         activation="relu", useDropout=False)
        self.up6 = Block(features*4*2, features*2, down=False,
                         activation="relu", useDropout=False)
        self.up7 = Block(features*2*2, features, down=False,
                         activation="relu", useDropout=False)
        
        self.finalUp = nn.Sequential(
            nn.ConvTranspose2d(features*2, inChannels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh(),  # Output of generator is in range [-1, 1]
        )

    def forward(self, x):
        d1 = self.initialDown(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        bottleneck = self.bottleneck(d7)

        ### The concatinations are due to the skip connections in the u-net
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        
        return self.finalUp(torch.cat([up7, d1], 1))
        
def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(inChannels=3, features=64)
    predictions = model(x)
    print(predictions.shape)

if __name__ == "__main__":
    test()
