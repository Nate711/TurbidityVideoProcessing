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
	t = np.dot(linalg.inv(K),P[:,3])
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

with open('3D_I/P3', 'rU') as f:  # problem with rb
	reader = csv.reader(f,delimiter=' ')

	camMat = [np.array(row) for row in reader]

	camMat = np.array([map(float,
				row[np.where(row!='')]) for row in camMat])
	
	print 'cam mat\n'+str(camMat)+'\n'
	K,R = rq(camMat[:,:3])
	T = np.diag(np.sign(np.diag(K)))
	K = np.dot(K,T)
	R = np.dot(T,R)

	#override R
	#R = np.array([[1,0,0],[0,1,0],[0,0,1]])
	#R = np.array([[.5,.866,0],[.866,-.5,0],[0,0,1]])

	t = np.dot(np.linalg.inv(camMat[:,:3]),camMat[:,3]).reshape(3,1)
	#print t
	#overwrite t
	#t[0]=0;t[1]=0


	print 'factorzation (k,r,t)\n' + str((K,R,t)) + '\n'

	#print t # t not quite the same as t from factor
	'''
	[x,y,1] = r1r2t^-1 DOT K^-1 TIMES l DOT xiyiwi
	'''

	# Test overriding camera matrix
	'''
	# ok overwrite R and t,K to see wth is going on, looks like it is working
	t = np.array([[0],[0],[4]])
	R = np.array([[.5,.866,0],[.866,-.5,0],[0,0,1]])
	#R = np.array([[1,0,0],[0,1,0],[0,0,1]])
	Rt = np.concatenate((R,t),1)

	K = np.eye(3)
	camMat = np.dot(K,Rt)
	'''

	r1r2t = np.array([R[0],R[1],t])
	r1r2tPrime = np.linalg.inv(r1r2t)

	invK = np.linalg.inv(K)
	#print 'invK\n' + str(invK) + '\n'

	#print 'r1r2tPrime\n' + str(r1r2tPrime) + '\n'

	xiyiwi = rect2Homo([100,200]) # is there a problem with making this a column vector?

	print 'image coordinates:\n' + str(xiyiwi) + '\n'

	# (r1r2t^-1) dot invk dot xiyiwi then add z=0
	xyw = np.dot(r1r2tPrime,np.dot(invK,xiyiwi))
	xyzw = np.insert(xyw,2,0)

	# normalize xyzw
	xyz = homo2Rect(xyzw)
	print 'world coordinates (x,y,z)\n' + str(xyz) + '\n'

	Rt = np.concatenate((R,t.reshape(3,1)),1)
	camMat = np.dot(K,Rt)

	xiyiwi = np.dot(camMat,xyzw)
	xiyi = homo2Rect(xiyiwi)
	print 'image coordinates\n' + str(xiyi)

	print camMat # gets the wrong t value

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