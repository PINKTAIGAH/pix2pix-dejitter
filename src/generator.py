import torch
import torch.nn as nn

class Block(nn.Module):
    """
    A torch.nn.Module instance containing a template for a convolutional block 
    used in the UNET network used in the pix2pix Generator

    Atributes
    ---------
    block: torch.nn.Sequential
        Object that will return the output of a convolutional block of the
        a PatchGAN discriminator (kernal size 4)
        
        Object is capable of being a convolutional or a transpose convolutional
        block depending on input parameters
        
        Activation function used will either be LeakyReLU or ReLU depending on 
        input parameters

    Parameters
    ----------
    inChannels: int
        Number of image channels in input image

    outChannels: int
        Number of image channels that output image will have

    down: bool, optional
        If True, block will preform a 2D convolution, else will preform a 
        transpose convolution

    act: string, optional
        If "relu", activation function applied to output will be ReLU.
        If "leaky", activation function applied to output will be LeakyReLU

    useDropout: bool, optional
        If True, Dropot will be applied to the output. This will zero random 
        elements of the output image.
        This has been proben to improve regularsation and is required by the 
        pix2pix paper
    """
    def __init__(self, inChannels, outChannels, down=True, act="relu", useDropout=False):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            # Operation to preformed on input if down parameter is True
            nn.Conv2d(
                inChannels, outChannels, kernel_size=4, stride=2, padding=1,
                bias=False, padding_mode="reflect",
            )
            if down
            # Operation to preformed on input if down parameter is False
            else nn.ConvTranspose2d(inChannels, outChannels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outChannels),
            # Apply activation function depending on the act parameter
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.useDropout = useDropout
        # Probability of element to be zerod in Dropout is 0.5, as described 
        # in original paper
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        """
        Returns output of convolutional blocks of a UNET block when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through convolutional block

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output image of the convolutional block
        """
        x = self.block(x)
        return self.dropout(x) if self.useDropout else x


class Generator(nn.Module):
    """
    A torch.nn.Module instance containing a UNET neural network used by the 
    pix2pix generator. 
    """
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU(),
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test():
    x = torch.randn((1, 1, 512, 512))
    model = Generator(in_channels=1, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
