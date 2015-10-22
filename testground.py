import numpy as np
from glob import glob
K= np.array([[1.14377690e+03,   0.00000000e+00,   9.71017458e+02],
 [  0.00000000e+00,   1.14128154e+03,   5.18938715e+02],
 [  0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])


names = glob('calibration_samples/*')
print names