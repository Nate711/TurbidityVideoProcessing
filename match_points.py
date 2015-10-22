import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

img1 = cv2.imread('Cam1/Cam1_Frame250.jpg',0)
img2 = cv2.imread('Cam2/Cam2_Frame250.jpg',0)
#plt.imshow(img1),plt.show()
sift = cv2.SIFT()

# find keypoints and descriptors with SIFT
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test from Lowe's paper (idk what that is)
for i,(m,n) in enumerate(matches):
	if m.distance < 0.8*n.distance:
		good.append(m)
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)

# convert to int32
pts1 = np.float32(pts1) #int32 wasn't working????
pts2 = np.float32(pts2) 

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

print(F)

# use only good points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

#print(pts1)

def drawLines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
	lines - corresponding epilines '''
	r,c = img1.shape
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	
	#print(img1) # actual image
	
	img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	
	#print(zip(lines,pts1,pts2)) # works
	
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

		# in place transformatino in CV2
		cv2.line(img1, (x0,y0), (x1,y1), color,1) # previously img1 = blah
		
		cv2.circle(img1,tuple(pt1),5,color,-1)
		#print(img1)
		cv2.circle(img2,tuple(pt2),5,color,-1)
	return img1,img2

# Find epilines through the points in the right image and first image. Draw the lienes on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F) # what does this do?
lines1 = lines1.reshape(-1,3)
img5,img6 = drawLines(img1,img2,lines1,pts1,pts2)



lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawLines(img2,img1,lines2,pts2,pts1)

cv2.imshow('hello',img5)
cv2.imshow('hello',img3)
#plt.subplot(121),plt.imshow(img5)
#plt.subplot(122),plt.imshow(img3)
#plt.show()
