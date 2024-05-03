import astrodash
import astropy
import numpy as np

from dataloader import spectra_redshift_type
from calibration import extract_correct_softmax


# Problem setup
n = 1000 # number of calibration points
alpha = 0.1 # 1-alpha is the desired coverage



