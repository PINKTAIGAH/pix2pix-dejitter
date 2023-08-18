import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    A torch.nn.Module instance containing a template for a convolutional block 
    used in a PatchGAN discriminator

    Atributes
    ---------
    block: torch.nn.Sequential
        Object that will return the output of a convolutional block of the
        a PatchGAN discriminator (kernal size 4)
        Output is then passed through batch normalisation and a leaky ReLU 
        activiation function

    Parameters
    ----------
    inChannels: int
        Number of image channels in input image

    outChannels: int
        Number of image channels that output image will have

    stride: int
        Stride of convolution
    """
    def __init__(self, inChannels, outChannels, stride):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            # These bias and padding_mode parameters are requiered for model to work 
            nn.Conv2d(
                inChannels, outChannels, kernel_size=4, stride=stride, 
                padding=1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Returns output of convolutional block when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through convolutional block 

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output image 
        """
        return self.block(x)

class Discriminator(nn.Module):
    """
    A torch.nn.Module instance containing a PatchGAN discriminator neural network
    designed to discriminate images of size 128*128, 256*256 and 512*512.

    Atributes
    ---------
    initial: torch.nn.Sequential instance
        Object that will return the output of the initial convolutional block of 
        a PatchGAN discriminator (does not contain a batch normalisation)

    model: torch.nn.Sequential instance
        Object that will return the outut of the PatchGAN discriminator

    Parameters
    ----------
    inChannels: int
        Number of colour channels in discriminator input

    features: list of ints
        List containing the number of output features for each convolutional
        block in the PatchGAN discriminator

    Notes
    -----
    PatchGAN discriminator outputs optimal results with an input image of size
    256*256 pixels

    * If input image is 128*128, output of PatchGAN will have shape ==> 14*14
    * If input image is 256*256, output of PatchGAN will have shape ==> 30*30
    * If input image is 512*512, output of PatchGAN will have shape ==> 62*62
    """
    def __init__(self, inChannels=1, features=[64, 128, 256, 512]):
        super().__init__()
        # Initial convolutional block of PatchGAN, does not contain BatchNorm
        self.initial = nn.Sequential(
            nn.Conv2d(
                # inChannels is multiplied by 2 due to both discriminator inputs
                # being concatinated together
                inChannels * 2, features[0], kernel_size=4, stride=2,
                padding=1, padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # Goal is to create a list containing objects for each block in PatchGAN
        # in a sequential order
        layers = []
        inChannels = features[0]
        # Iterate over numbers of output features
        for feature in features[1:]:
            # Append the following convolutionl block in the sequence to the list
            # Assing a stride of 2 to each block, except for the last, with a strid eof 1
            layers.append(
                CNNBlock(inChannels, feature, stride=1 if feature == features[-1] else 2),
            )
            # Set output features of this block as input channels for the next block
            inChannels = feature

        # Append the final block to the list, without a batch notmalisation and a
        # stride of 1
        layers.append(
            nn.Conv2d(
                inChannels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Returns output of PatchGAN discriminator when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through discriminator

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing discriminator score of the inputted image
            Output size is dependant on size of input (see Notes)
        """
        # Concatinate both fake and real input to feed into PatchGAN, as 
        # instructed in the original paper
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    N = 256
    x = torch.randn((1, 1, N, N))
    y = torch.randn((1, 1, N, N))
    model = Discriminator(inChannels=1)
    preds = model(x, y)
    print(preds.shape)

if __name__ == "__main__":
    test()
