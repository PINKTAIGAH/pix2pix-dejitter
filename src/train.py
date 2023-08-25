import torch
import utils 
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
    """
    Iterate though one epoch of training for the pix2pix generator and discriminator.

    Parameters
    ----------
    disc: torch.nn.Module instance
        Object to return the output of the PatchGAN discriminator of pix2pix.

    gen: torch.nn.Module instance
        Object to return the output of the UNET generator of pix2pix.

    loader: torch.utils.Data.DataLoader instance
        Iterable object containing the training dataset divided into batches.

    opt_disc: torch.optim instance
        Instance of the optimiser unsed to train the discriminator. The optimiser
        currently being used is Adam.

    opt_gen: torch.optim instance
        Instance of the optimiser unsed to train the generator. The optimiser
        currently being used is Adam.

    l1_loss: torch.nn.L1Loss instance
        Object that retuns the output of the L1 distance between input parameters.

    bce: torch.nn.BCEWithLogitsLoss instance
        Object that returns the output of the GAN adverserial loss function.

    d_scaler: torch.cuda.amp.Gradscaler instance
        Object that will scale the type size appropiatly to allow for automatic
        mixed precision for discriminator forward and backward pass.

    g_scaler: torch.cuda.amp.Gradscaler instance
        Object that will scale the type size appropiatly to allow for automatic
        mixed precision for generator forward and backward pass.

    Returns
    -------
    output: tuple of floats
        Tuple containing the mean generator and discriminator losses trhoughout
        the entire epoch
    """
    # Initialise tqdm object to visualise training in command line
    loop = tqdm(loader, leave=True)
    # Initialise running loss that will contaain cumulative sum of losses
    running_loss_disc = 0.0
    running_loss_gen = 0.0
    running_disc_real = 0.0
    running_disc_fake = 0.0

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

        with torch.no_grad():
            # Add current loss to the running loss
            running_loss_gen += G_loss.mean().item()
            running_loss_disc += D_loss.mean().item()

            # Add current loss to the running loss
            running_disc_real += torch.sigmoid(D_real).mean().item()
            running_disc_fake += torch.sigmoid(D_fake).mean().item()

    # Create tuple with output values
    with torch.no_grad():
        output = (
            running_loss_disc/config.BATCH_SIZE,
            running_loss_gen/config.BATCH_SIZE,
            running_disc_real/config.BATCH_SIZE,
            running_disc_fake/config.BATCH_SIZE,
        ) 
    return output 

def main():
    # Define discriminator and generator objects 
    disc = Discriminator(inChannels=config.CHANNELS_IMG).to(config.DEVICE)
    gen = Generator(inChannels=1, features=64).to(config.DEVICE)
    # Define optimiser for both discriminator and generator
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
    )
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
    )

    # Define GAN adverserial loss (BCE) and L1 loss of pix2pix
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # Load previously saved models and optimisers if True
    if config.LOAD_MODEL:
        utils.load_checkpoint(
            config.CHECKPOINT_GEN_LOAD, gen, opt_gen, config.LEARNING_RATE,
        )
        utils.load_checkpoint(
            config.CHECKPOINT_DISC_LOAD, disc, opt_disc, config.LEARNING_RATE,
        )

    # Initialise training dataset and dataloader
    # train_dataset = JitteredDataset(1000) 
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
    # val_dataset = JitteredDataset(500,) 

    val_dataset = FileDataset(config.VAL_DIR, config.IMAGE_SIZE,) 
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    ### Temporary ###
    # val_dataset = FileDataset(config.TRAIN_DIR, config.IMAGE_SIZE,) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    ### Temporary ###
    """
    Training loop
    """
    # Iterte over epochs
    for epoch in range(config.NUM_EPOCHS):
        # Train one iteration of generator and discriminator
        # Model losses has two elements, disc loss and gen loss respectivly
        model_losses = _trainFunction(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,
            g_scaler, d_scaler,)

        if epoch == 0:
            utils.write_out_titles(config.MODEL_LOSSES_TITLES, config.MODEL_LOSSES_FILE)
            
        # Write out epoch and then loss per epoch. Start new line once compleated
        utils.write_out_value(epoch, config.MODEL_LOSSES_FILE, new_line=False)    
        utils.write_out_value(model_losses[0], config.MODEL_LOSSES_FILE, new_line=False)    
        utils.write_out_value(model_losses[1], config.MODEL_LOSSES_FILE, new_line=True)    

        # Write out epoch and then critic score per epoch. Start new line once compleated
        utils.write_out_value(epoch, config.CRITIC_SCORE_FILE, new_line=False)    
        utils.write_out_value(model_losses[2], config.CRITIC_SCORE_FILE, new_line=False)    
        utils.write_out_value(model_losses[3], config.CRITIC_SCORE_FILE, new_line=True)    
        # Save images of ground truth, jittered and generated unjittered images 
        # using models of current epoch
        utils.save_examples_concatinated(gen, val_loader, epoch,
                                         folder=config.TRAIN_IMAGE_FILE,)

        ### Temporary ###
        # Save images of ground truth, jittered and generated unjittered images 
        # using models of current epoch
        # save_examples(gen, val_loader, epoch, folder="evaluation")
        ### Temporary ###

        # Save models and optimisers every 5 epochs
        if config.SAVE_MODEL and epoch % 5 == 0:
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN_SAVE)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC_SAVE)

if __name__ == "__main__":
    main()
