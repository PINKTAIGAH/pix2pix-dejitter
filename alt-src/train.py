import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import JitteredDataset  
from datasetCells import CellDataset 
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
    ):
    loop = tqdm(loader, leave=True)
    step = 0

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

        with torch.no_grad():
            fakeSample = gen(x) 
            imageGridReal = make_grid(y[:6], normalize=True)
            imageGridFake = make_grid(fakeSample[:6], normalize=True)

            config.WRITER_REAL.add_image("real", imageGridReal, global_step=step)
            config.WRITER_FAKE.add_image("fake", imageGridFake, global_step=step)

            step +=1
    
    with torch.no_grad():
        config.WRITER_REAL.add_scalar("discriminator real", torch.sigmoid(D_real).mean().item())
        config.WRITER_FAKE.add_scalar("discriminator fake", torch.sigmoid(D_fake).mean().item())
        config.WRITER_REAL.add_scalar("discriminator loss", D_loss.item())
        config.WRITER_REAL.add_scalar("generator loss", G_loss.item())


def main():
    disc = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = JitteredDataset(config.IMAGE_SIZE, 1000) 
    # train_dataset = CellDataset(config.TRAIN_DIR, config.IMAGE_SIZE,
                                # config.MAX_JITTER, config.transformsCell) 
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = JitteredDataset(config.IMAGE_SIZE, 500,) 
    # val_dataset = CellDataset(config.VAL_DIR, config.IMAGE_SIZE,
                                # config.MAX_JITTER, config.transformsCell) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    """
    schedular_disc = optim.lr_scheduler.ReduceLROnPlateau(opt_disc, mode="min",
                                                          factor=config.SCHEDULAR_DECAY,
                                                          patience=config.SCHEDULAR_PATIENCE,
                                                          verbose=True)
    schedular_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, mode="min",
                                                         factor=config.SCHEDULAR_DECAY,
                                                         patience=config.SCHEDULAR_PATIENCE,
                                                         verbose=True)
    """

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,
            g_scaler, d_scaler,)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")

if __name__ == "__main__":
    main()
