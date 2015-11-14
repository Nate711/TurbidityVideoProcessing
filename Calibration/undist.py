# source: Solems tutorial
import numpy as np
import cv2
from glob import glob

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--source', '-s',help='undistort images',required=True)
parser.add_argument('--dest','-d',help='save folder', required=True)

args = parser.parse_args()

'''
# calibration parameters from all of my calibration images
K = np.array([[1.14377690e+03,   0.00000000e+00,   9.71017458e+02],
 [  0.00000000e+00,   1.14128154e+03,   5.18938715e+02], #2nd e is 03
 [  0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])
d = np.array([-0.35194004,  0.22321843, -0.00039513, -0.00041706, -0.07914194]) # just use first two terms (no translation)
'''
# calibration parameters from subset of calibration images

K = np.array([[  1.14436060e+03,   0.00000000e+00,   9.73815117e+02],
 [  0.00000000e+00,   1.14177584e+03,   5.18330528e+02],
 [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
d = np.array([-0.35000645,  0.22143909,0,0,0 ]) # just use first two terms (no translation)


# calibration parameters from sample images
'''
K = np.array([[ 611.18390818,    0.,          515.31102633],
 [   0.,          611.06735263,  402.07540928],
 [   0.,            0.,            1.        ]])

d = np.array([-0.36824155,  0.28485465,0,0,0])#,  0.00079123,  0.00064925, -0.1634569 ])
'''


# read one of your images
# works well with 

# for sample images
#imgNames = glob('../calibration_samples_ex/GO*[0-9].JPG')

# for my own images
relativePath = 'calibration_samples2/'
imgNames = glob(relativePath + 'Cam1_Frame[0-9]*.jpg')
saveFolder = 'Dist_Undist_Images/'

# for undistorting gopro real images
relativePath = 'Cam1/'
imgNames = glob(relativePath + 'Cam*.jpg')
saveFolder = 'Cam1_Undist/'


relativePath = args.source
imgNames = glob(relativePath + '*.jpg')
saveFolder = args.dest

print 'Image Names: ' + str(imgNames)


for name in imgNames:
	img = cv2.imread(name)
	h, w = img.shape[:2]
	# undistort
	newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0) 
	newimg = cv2.undistort(img, K, d, None, newcamera)
	
	# for samples
	#path = name[name.index('GOPR00'):-4] + '_UNDIST' + name[-4:] # ex GOPR0015_UNDIST.JPG
	# for my own
	#path = name[name.index('Calibration'):-4] + '_UNDIST' + name[-4:] # ex Calibration101_UNDIST.jpg

	path = name[len(relativePath):-4] + '_Undist' + name[-4:]


	print saveFolder + path
	cv2.imwrite(saveFolder + path,newimg)


'''img = cv2.imread("Cam1/Cam1_Frame157.jpg")
#img = cv2.imread("../calibration_samples2/Calibration_Frame100.jpg")
h, w = img.shape[:2]

# undistort
newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0) 
newimg = cv2.undistort(img, K, d, None, newcamera)

cv2.imwrite("_original.jpg", img)
cv2.imwrite("_undistorted.jpg", newimg)'''