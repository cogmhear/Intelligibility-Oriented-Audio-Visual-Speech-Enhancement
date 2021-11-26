#  Copyright (c) 2021 Mandar Gogate, All rights reserved.
from os.path import expanduser, join
from scipy import signal

## Generic Params
HOME_DIR = expanduser("~")
USE_GPU = True
SEED = 16108

# Dataset Params
GRID_ROOT = "./data/GRID/Raw/"
GRID_CHIME3_ROOT = "./data/GRID_CHIME3/utterances/"

# GRID Meta
GRID_EXCEPTIONS = [15, 8, 12]
GRID_FEMALE_SPEAKERS = [4, 7, 11, 15, 16, 18, 20, 22, 23, 24, 25, 29, 31, 33, 34]
GRID_MALE_SPEAKERS = [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19, 26, 27, 28, 30, 32]
GRID_SPEAKERS = GRID_MALE_SPEAKERS + GRID_FEMALE_SPEAKERS

# Visual Data Params
nb_channels, img_height, img_width = 1, 88, 88
GRID_IMAGES_ROOT_sq = "./data/GRID/lip_sq_bg/"

# Fourier Transform Params
stft_size = 511
window_size = 400
window_shift = 160
window_length = None
sampling_rate = 16000
fading = False
windows = signal.windows.hann
