import torch
import config
import utils
import torch.optim as optim
from dataset import JitteredDataset
from datasetFile import FileDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader

"""
Evaluate the putputs of a trained pair of GAN models by loading them and calculating
measures of image similarity.
"""

# Initialise generator and discriminator
disc = Discriminator(inChannels=config.CHANNELS_IMG).to(config.DEVICE)
gen = Generator(inChannels=1, features=64).to(config.DEVICE)
# Define optimiser for both discriminator and generator
opt_disc = optim.Adam(
    disc.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
)
opt_gen = optim.Adam(
    gen.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
)

# Load generator and discriminator from specified file
utils.load_checkpoint(
    config.CHECKPOINT_GEN_LOAD, gen, opt_gen, config.LEARNING_RATE,
)
utils.load_checkpoint(
    config.CHECKPOINT_DISC_LOAD, disc, opt_disc, config.LEARNING_RATE,
)

# Load validation dataset and dataloader
val_dataset = JitteredDataset(1,)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=True)

# Set generator mode to evaluation
gen.eval()

# Initialise function that will calculate L1
L1 = torch.nn.L1Loss()
l1_list = []

# Iterate over epochs
with torch.no_grad():
    for epoch in range(config.EVALUATION_EPOCHS):
        # Iterate over all images in batches
        for idx, (x, y, _) in enumerate(val_loader):
            # Send x, y, and generated y to device
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            y_fake = gen(x).to(config.DEVICE)
            print(L1(y, y_fake).item())
            l1_list.append(L1(y, y_fake).item())
 
l1_output = sum(l1_list)/len(l1_list)
utils.write_out_value(config.SIGMA, config.EVALUATION_METRIC_FILE) 
utils.write_out_value(l1_output, config.EVALUATION_METRIC_FILE, new_line=True)
