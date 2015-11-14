import numpy as np
import csv
from scipy import linalg
import cv2

def factor(P):
	K,R = linalg.rq(P[:,:3])
	T = np.diag(np.sign(np.diag(K)))
	if linalg.det(T) < 0:
		T[1,1] *= -1

	K = np.dot(K,T)
	R = np.dot(T,R)
	t = np.dot(linalg.inv(K),P[:,3]) # this gives t not C
	return K,R,t

def rq(A):
	Q,R = np.linalg.qr(np.flipud(A).T)
	R = np.flipud(R.T)
	Q=Q.T
	return R[:,::-1],Q[::-1,:]
def homo2Rect(V):
	return V[:-1]/V[-1]
def rect2Homo(V):
	return np.append(V,1)

def constructCameraMatrix(K,R,t):
	Rt = np.concatenate((R,t.reshape(3,1)),1)
	camMat = np.dot(K,Rt)
	return camMat

def readCameraMatrix(filename):
	with open(filename,'rU') as f:
		reader = csv.reader(f,delimiter=' ')

		camMat = [np.array(row) for row in reader]

		camMat = np.array([map(float,
					row[np.where(row!='')]) for row in camMat])

		return camMat

def transformationMatrixRt(R,t):
	trans = np.concatenate((R,t.reshape(3,1)),1)
	#print trans
	trans = np.concatenate((trans,np.array([[0,0,0,1]])),axis=0)

	return trans

def groundCoordinates((xi,yi), K,R,t):
	wi = 1
	#print 'image coordinates:\n' + str((xi,yi,wi)) + '\n'
	invK = np.linalg.inv(K)

	#print 'image frame coordinates: \n' + str(xcyczc)
	C = -np.dot(R.T,t)

	#print C
	# skew is not taken into account here
	cx,cy = K[0,2],K[1,2]
	fx,fy = K[0,0],K[1,1]

	zc = -C[2] / (R[0,2]*(xi-cx)/fx + R[1,2]*(yi-cy)/fy + R[2,2])
	xcyczc = zc*np.dot(invK,np.array([xi,yi,wi]))

	#print 'xycyzc\n {} \n'.format(xcyczc)

	#print 'reverse 3d image coordinates to get pixel coordinates\n' + str(np.around(homo2Rect(np.dot(K,xcyczc)),2)) + '\n'

	xcyczcwc = rect2Homo(xcyczc)
 
	inverseTransformationMatrix= np.linalg.inv(transformationMatrixRt(R,t))
	xyzw = np.dot(inverseTransformationMatrix,xcyczcwc)
	#print 'world coordinates (x,y,z)\n' + str(np.around(homo2Rect(xyzw))) + '\n'

	return xyzw
def imageCoordinates((x,y,z,w), camMat):
	return np.dot(camMat,[x,y,z,w])
def imageCoordinates(xyzw, camMat):
	return np.dot(camMat,xyzw)

def drawGPSGrid(img,camMat):
	gmin,gmax = -500,500

	x = np.arange(gmin,gmax,10)
	for xi in x:
		y0,y1 = gmin, gmax
		(x0,y0) = np.array(homo2Rect(imageCoordinates((xi,y0,0,1),camMat)),dtype='int')
		(x1,y1) = np.array(homo2Rect(imageCoordinates((xi,y1,0,1),camMat)),dtype='int')
		cv2.line(img,(x0,y0),(x1,y1),(255,0,0),1,lineType=cv2.CV_AA)
	for yi in np.arange(gmin,gmax,10):
		x0,x1 = gmin, gmax
		(x0,y0) = np.array(homo2Rect(imageCoordinates((x0,yi,0,1),camMat)),dtype='int')
		(x1,y1) = np.array(homo2Rect(imageCoordinates((x1,yi,0,1),camMat)),dtype='int')
		cv2.line(img,(x0,y0),(x1,y1),(255,0,0),1,lineType=cv2.CV_AA)

geomNames = ['P1','P2','P3','P4','P5','P6']
imgNamesI = ['1726_p1_s.pgm','1727_p1_s.pgm','1728_p1_s.pgm','1762_p1_s.pgm','1763_p1_s.pgm','1764_p1_s.pgm']
imgNamesII = ['1726_p1_s1.pgm','1727_p1_s1.pgm','1728_p1_s1.pgm','1762_p1_s1.pgm','1763_p1_s1.pgm','1764_p1_s1.pgm']

for (index,name) in enumerate(geomNames):
	camMat = readCameraMatrix('3D_I/' + name)
	K,R,t = factor(camMat)
	#R = np.array([[0,.866,0.5],[0,-.5,.866],[1,0,0]])
	#R = np.array([[0.5,.866,0],[.866,-.5,0],[0,0,1]])

	xyzw = groundCoordinates((0,0),K,R,t)
	print 'world coordinates (x,y,z)\n' + str(np.around(homo2Rect(xyzw),3)) + '\n'

	# reverse the process to make sure everything is working
	camMat = constructCameraMatrix(K,R,t)
	xiyi = homo2Rect(imageCoordinates(xyzw,camMat))
	print 'image coordinates\n' + str(np.around(xiyi,3)) + '\n'

	img = cv2.imread('images_I/'+imgNamesI[index],cv2.IMREAD_GRAYSCALE)

	drawGPSGrid(img,camMat)

	cv2.imshow('image',img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# summary of problem:
''' 
I think everything is working??
'''