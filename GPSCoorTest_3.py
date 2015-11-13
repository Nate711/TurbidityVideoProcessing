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


img = cv2.imread('images/1726_p1_s.pgm',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

with open('3D_I/P1', 'rU') as f:  # problem with rb
	reader = csv.reader(f,delimiter=' ')

	camMat = [np.array(row) for row in reader]

	camMat = np.array([map(float,
				row[np.where(row!='')]) for row in camMat])
	
	print 'cam mat\n'+str(camMat)+'\n'
	K,R,t = factor(camMat)

	#override R
	#R = np.array([[1,0,0],[0,1,0],[0,0,1]])
	R = np.array([[1,0,0],[0,-.5,.866],[0,.866,0.5]])

	#t[0]=0;t[1]=0

	print 'factorization (k,r,t)\n' + str((K,R,t)) + '\n'

	# Test overriding camera matrix
	'''
	# ok overwrite R and t,K to see wth is going on, looks like it is working
	t = np.array([[0],[0],[4]])
	R = np.array([[.5,.866,0],[.866,-.5,0],[0,0,1]])
	#R = np.array([[1,0,0],[0,1,0],[0,0,1]])
	Rt = np.concatenate((R,t),1)
	camMat = np.dot(K,Rt)
	'''

	'''# not sure if the K from the factorization is good or not
	K = np.eye(3)
	K[0,0] = 100
	K[1,1] = 100
	'''

	(xi,yi,wi) = rect2Homo([20,150]).reshape(3,1) # is there a problem with making this a column vector?
	print 'image coordinates:\n' + str((xi,yi,wi)) + '\n'

	invK = np.linalg.inv(K)

	#print 'image frame coordinates: \n' + str(xcyczc)
	C = -np.dot(R.T,t)

	#print C
	# skew is not taken into account here
	cx,cy = K[0,2],K[1,2]
	fx,fy = K[0,0],K[1,1]

	zc = -C[2] / (R[0,2]*(xi-cx)/fx + R[1,2]*(yi-cy)/fy + R[2,2])
	print 'zc: \n {} \n'.format(zc)
	xcyczc = zc*np.dot(invK,np.array([xi,yi,wi]))
	
	print 'xycyzc\n {} \n'.format(xcyczc)

	print 'image coordinates\n' + str(np.around(homo2Rect(np.dot(K,xcyczc)),2)) + '\n'

	xcyczcwc = rect2Homo(xcyczc)
	#print 'xcyczcwc\n' + str(np.around(xcyczcwc,2)) + '\n'
	
	negRTranst = -np.dot(R.T,t)

	transformation = np.zeros((4,4))
	transformation[0:3,0:3] = R.T
	transformation[0:3,3] = negRTranst
	transformation[3,:] = [0,0,0,1]
	#print transformation

	xyzw = np.dot(transformation,xcyczcwc)

	# normalize xyzw
	xyz = homo2Rect(xyzw)
	print 'world coordinates (x,y,z)\n' + str(np.around(xyz)) + '\n'

	Rt = np.concatenate((R,t.reshape(3,1)),1)
	camMat = np.dot(K,Rt)

	xiyi = homo2Rect(np.dot(camMat,xyzw))
	print 'image coordinates\n' + str(np.around(xiyi,2)) + '\n'

	print 'reconstructed camMat\n' + str(camMat) # gets the wrong t value

	#print 'refactored camMat \n {} \n'.format(factor(camMat))

# summary of problem:
''' 
given P# files not working out, image->world-> image is broken : last image coors incorrect
K is not the problem
R is ok I think (image coordinates slighlty off in reconstruction for some reason, maybe error in camera matrix? but it's like 4%)
only thing is t left
t is totally screwed up
only works if tx and ty = 0
somehow K*Rt messes up tx and ty
'''