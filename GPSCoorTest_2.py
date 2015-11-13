import numpy as np
import csv
from scipy import linalg
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

with open('3D_I/P2', 'rU') as f:  # problem with rb
	reader = csv.reader(f,delimiter=' ')

	camMat = [np.array(row) for row in reader]

	camMat = np.array([map(float,
				row[np.where(row!='')]) for row in camMat])
	
	print 'cam mat\n'+str(camMat)+'\n'
	K,R,t = factor(camMat)

	#override R
	#R = np.array([[1,0,0],[0,1,0],[0,0,1]])
	R = np.array([[0,.866,0.5],[0,-.5,.866],[1,0,0]])

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

	xiyiwi = rect2Homo([0,-600]).reshape(3,1) # is there a problem with making this a column vector?
	print 'image coordinates:\n' + str(xiyiwi) + '\n'

	invK = np.linalg.inv(K)
	xcyczc = np.dot(invK,xiyiwi)

	#print 'image frame coordinates: \n' + str(xcyczc)

	C = -np.dot(R.T,t)

	#print C
	# this is still wrong oops
	xcyczc[2] = (-C[2] - R[0,2]*xcyczc[0]-R[1,2]*xcyczc[1]) / R[2,2]

	# MAKES SENSE BUT DOESNT SHOW UP IN THE MATH!
	xcyczc[1] *= xcyczc[2]
	xcyczc[0] *= xcyczc[2]

	print 'xycyzc\n' + str(xcyczc) + '\n'

	print 'image coordinates\n' + str(homo2Rect(np.dot(K,xcyczc))) + '\n'

	xcyczcwc = np.append(xcyczc,[1])
	print 'xcyczcwc\n' + str(xcyczcwc) + '\n'
	
	negRTranst = -np.dot(R.T,t)

	transformation = np.zeros((4,4))
	transformation[0:3,0:3] = R.T
	transformation[0:3,3] = negRTranst
	transformation[3,:] = [0,0,0,1]
	#print transformation

	xyzw = np.dot(transformation,xcyczcwc)

	# normalize xyzw
	xyz = homo2Rect(xyzw)
	print 'world coordinates (x,y,z)\n' + str(xyz) + '\n'

	# why is z not zeroooo!!!!!!

	Rt = np.concatenate((R,t.reshape(3,1)),1)
	camMat = np.dot(K,Rt)

	xiyiwi = np.dot(camMat,xyzw)
	xiyi = homo2Rect(xiyiwi)
	print 'image coordinates\n' + str(xiyi) + '\n'

	print 'reconstructed camMat\n' + str(camMat) # gets the wrong t value

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