import numpy as np
import utils

OUTPUT_FILE = "../raw_data/sobel_siemens.txt"
SIGMA = 2
dataset = np.loadtxt("../raw_data/sobel_sigma_2.txt", delimiter=',', usecols=range(2)).T
sobel_delta_array = np.unique(dataset[1])
sobel_delta_output = sobel_delta_array.mean()
    
sobel_delta_error = np.abs(sobel_delta_array.std()/np.sqrt(sobel_delta_array.size))
utils.write_out_value(SIGMA,  OUTPUT_FILE) 
utils.write_out_value(sobel_delta_output, OUTPUT_FILE)
utils.write_out_value(sobel_delta_error, OUTPUT_FILE, new_line=True)
