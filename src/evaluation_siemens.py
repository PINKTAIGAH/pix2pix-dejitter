import torch
import config
import utils
import torch.optim as optim
from dataset import JitteredDataset
from datasetFile import FileDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
import numpy as np
from skimage import filters

"""
Evaluate the putputs of a trained pair of GAN models by loading them and calculating
measures of image similarity.
"""

#EVALUATION_IMAGE_FILE = '../evaluation/siemens_sigma_30'
#CHECKPOINT_GEN_LOAD = '../models/gen.siemens_sigma_30.tar'
#CHECKPOINT_DISC_LOAD = '../models/disc.siemens_sigma_30.tar'
#SIGMA = 30

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
# val_dataset = JitteredDataset(1,)
val_dataset = FileDataset(config.SIEMENS_VAL_DIR, config.IMAGE_SIZE)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Set generator mode to evaluation
gen.eval()

# Initialise function that will calculate L1
L1 = torch.nn.L1Loss()
l1_list = []
sobel_delta_list = []

# Iterate over epochs
with torch.no_grad():
    for epoch in range(config.EVALUATION_EPOCHS):
        # Iterate over all images in batches
        print(f"Evaluating epoch ==> {epoch}")

        # Save ecample of images
        sobel_score = utils.save_examples_concatinated(
            gen, val_loader, (epoch), config.EVALUATION_IMAGE_FILE, True
        )
        sobel_delta_list.append(sobel_score)

        utils.write_out_value(epoch, "../raw_data/sobel_trial.txt", False)
        utils.write_out_value(sobel_delta_list[-1], "../raw_data/sobel_trial.txt", True)

"""
# Write out mean sobel delta to file
sobel_delta_array = np.array(sobel_delta_list)
sobel_delta_output = sobel_delta_array.mean()
sobel_delta_error = np.abs(sobel_delta_array.std()/np.sqrt(sobel_delta_array.size))
utils.write_out_value(config.SIGMA, config.EVALUATION_METRIC_FILE) 
utils.write_out_value(sobel_delta_output, config.EVALUATION_METRIC_FILE, new_line=False)
utils.write_out_value(sobel_delta_error, config.EVALUATION_METRIC_FILE, new_line=True)
"""


