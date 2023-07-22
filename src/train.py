import torch
from utils import saveCheckpoint, loadCheckpoint, saveSomeExamples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import JitteredDataset 
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision

# torch.backends.cudnn.benchmark = True


def train(discriminator, generator, loader, optimiserDiscriminator,
          optimiserGenerator, l1Loss, bce, gScaler, dScaler):

    loop = tqdm(loader, leave=True)
    step = 0

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            yFake = generator(x)
            discriminatorReal = discriminator(x, y)
            discriminatorRealLoss = bce(discriminatorReal,
                                        torch.ones_like(discriminatorReal))
            discriminatorFake = discriminator(x, yFake.detach())
            discriminatorFakeLoss = bce(discriminatorFake,
                              torch.zeros_like(discriminatorFake))
            discriminatorLoss = (discriminatorRealLoss + discriminatorFakeLoss) / 2

        discriminator.zero_grad()
        dScaler.scale(discriminatorLoss).backward()
        dScaler.step(optimiserDiscriminator)
        dScaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            discriminatorFake = discriminator(x, yFake)
            generatorFakeLoss = bce(discriminatorFake,
                              torch.ones_like(discriminatorFake))
            L1 = l1Loss(yFake, y) * config.L1_LAMBDA
            generatorLoss = generatorFakeLoss + L1

        optimiserGenerator.zero_grad()
        gScaler.scale(generatorLoss).backward()
        gScaler.step(optimiserGenerator)
        gScaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                discriminatorReal=torch.sigmoid(discriminatorReal).mean().item(),
                discriminatorFake=torch.sigmoid(discriminatorFake).mean().item(),
            )
	
    
        with torch.no_grad():
            fakeSample = generator(x) 
            imageGridReal = torchvision.utils.make_grid(y[:32], normalize=True)
            imageGridFake = torchvision.utils.make_grid(fakeSample[:32], normalize=True)

            config.WRITER_REAL.add_image("real", imageGridReal, global_step=step)
            config.WRITER_FAKE.add_image("fake", imageGridFake, global_step=step)

            step +=1

   

def main():
    discriminator = Discriminator(inChannel=config.CHANNELS_IMAGE).to(config.DEVICE)
    generator = Generator(inChannels=config.CHANNELS_IMAGE, features=64).to(config.DEVICE)

    optimiserDiscriminator = optim.Adam(discriminator.parameters(),
                                        lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    optimiserGenerator = optim.Adam(generator.parameters(),
                                    lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        loadCheckpoint(
            config.CHECKPOINT_GEN, generator,
            optimiserGenerator, config.LEARNING_RATE,
        )
        loadCheckpoint(
            config.CHECKPOINT_DISC, discriminator,
            optimiserDiscriminator, config.LEARNING_RATE,
        )

    trainDataset = JitteredDataset(config.IMAGE_SIZE, config.IMAGE_JITTER, length=4096)
    trainLoader = DataLoader(
        trainDataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.N_WORKERS,
    )
    gScaler = torch.cuda.amp.GradScaler()
    dScaler = torch.cuda.amp.GradScaler()

    validationDataset = JitteredDataset(config.IMAGE_SIZE, config.IMAGE_JITTER, length=128)
    validationLoader = DataLoader(validationDataset, batch_size=1, shuffle=False)

    for epoch in range(config.N_EPOCHS):
        train(
            discriminator, generator, trainLoader, optimiserDiscriminator,
            optimiserGenerator, L1_LOSS, BCE, gScaler, dScaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            saveCheckpoint(generator, optimiserGenerator,
                           filename=config.CHECKPOINT_GEN)
            saveCheckpoint(discriminator, optimiserDiscriminator,
                           filename=config.CHECKPOINT_DISC)

        if epoch % 50 == 0:
            saveSomeExamples(generator, validationLoader, epoch, folder="evaluation")

    config.WRITER_REAL.close()
    config.WRITER_FAKE.close()

if __name__ == "__main__":
    main()
