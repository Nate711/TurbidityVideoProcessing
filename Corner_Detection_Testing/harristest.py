import cv2
import numpy as np

def nothing(x):
	pass

filename = 'varo.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('hello')
cv2.createTrackbar('dst', 'hello', 50, 255,nothing)

while(1):
	K = cv2.getTrackbarPos('dst','hello')
	print K
	t,gray2 = cv2.threshold( gray, K, 255,cv2.THRESH_BINARY);
	cv2.imshow('hello',gray2)

	if cv2.waitKey(1) & 0xff == 27:
		break

cv2.destroyAllWindows()

gray = np.float32(gray)


dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()