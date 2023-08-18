import torch
from utils import  save_checkpoint, load_checkpoint, save_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import JitteredDataset  
from datasetFile import FileDataset 
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def _trainFunction(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
    ):
    # Initialise tqdm object to visualise training in command line
    loop = tqdm(loader, leave=True)

    # Iterate over images in batch of data loader
    for idx, (x, y, _) in enumerate(loop):
        # Send tensors from dataloader to device
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            # Generate unshifted image 
            y_fake = gen(x)

            # Calculate discriminator score of true & fake image & gradient penalty 
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            # Calculate adverserial loss of original GAN for discriminator
            D_loss = (D_real_loss + D_fake_loss) / 2

        # Zero gradients of discriminator to avoid old gradients affecting backwards
        # pass
        disc.zero_grad()
        # Backwards pass 
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            # Compute pix2pix loss function of generator 
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        # Zero gradients of discriminator to avoid old gradients affecting backwards
        # pass
        opt_gen.zero_grad()
        # Backwards pass 
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Output loss function of generator and discriminator to command line
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    """
    with torch.no_grad():
        config.WRITER_REAL.add_scalar("discriminator real", torch.sigmoid(D_real).mean().item(), epoch)
        config.WRITER_FAKE.add_scalar("discriminator fake", torch.sigmoid(D_fake).mean().item(), epoch)
        config.WRITER_REAL.add_scalar("discriminator loss", D_loss.item(), epoch)
        config.WRITER_REAL.add_scalar("generator loss", G_loss.item(), epoch)
        config.WRITER_REAL.add_scalar("correlation", findCorrelation(gen, val_loader), epoch)
    """

def main():
    # Define discriminator and generator objects 
    disc = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)
    # Define optimiser for both discriminator and generator
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
    )
    opt_gen = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
    )

    # Define GAN adverserial loss (BCE) and L1 loss of pix2pix
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # Load previously saved models and optimisers if True
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    # Initialise training dataset and dataloader
    #train_dataset = JitteredDataset(config.IMAGE_SIZE, 1000) 
    train_dataset = FileDataset(config.TRAIN_DIR, config.IMAGE_SIZE,) 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # Initialise Gradscaler to allow for automatic mixed precission during training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Initialise validation dataset and dataloader
    #val_dataset = JitteredDataset(config.IMAGE_SIZE, 500,) 
    val_dataset = FileDataset(config.VAL_DIR, config.IMAGE_SIZE,) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    """
    Training loop
    """
    # Iterte over epochs
    for epoch in range(config.NUM_EPOCHS):
        # Train one iteration of generator and discriminator
        _trainFunction(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,
            g_scaler, d_scaler,)

        # Save images of ground truth, jittered and generated unjittered images 
        # using models of current epoch
        save_examples(gen, val_loader, epoch, folder="evaluation")

        # Save models and optimisers every 5 epochs
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

if __name__ == "__main__":
    main()
