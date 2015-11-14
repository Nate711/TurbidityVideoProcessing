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

img = cv2.imread('images/1726_p1_s.pgm',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

camMat = readCameraMatrix('3D_I/P1')
#print 'cam mat\n'+str(camMat)+'\n'
K,R,t = factor(camMat)

#print 'factorization (k,r,t)\n' + str((K,R,t)) + '\n'

xyzw = groundCoordinates((0,0),K,R,t)
print 'world coordinates (x,y,z)\n' + str(np.around(homo2Rect(xyzw))) + '\n'

# reverse the process to make sure everything is working
camMat = constructCameraMatrix(K,R,t)
xiyi = homo2Rect(np.dot(camMat,xyzw))
print 'image coordinates\n' + str(np.around(xiyi,2)) + '\n'
#print 'reconstructed camMat\n' + str(camMat) # gets the wrong t value

# summary of problem:
''' 
I think everything is working??
'''