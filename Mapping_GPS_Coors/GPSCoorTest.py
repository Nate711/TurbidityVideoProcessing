import numpy as np
import csv
from scipy import linalg
import cv2
import math

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

    # this line does the opposite transformation, should not be used:
    # xyzw = np.dot(transformationMatrixRt(R,t),xcyczcwc)

    #print 'world coordinates (x,y,z)\n' + str(np.around(homo2Rect(xyzw))) + '\n'

    return xyzw

def imageCoordinates((x,y,z,w), camMat):
    return np.dot(camMat,[x,y,z,w])

def imageCoordinates(xyzw, camMat):
    return np.dot(camMat,xyzw)

def drawGPSGrid(img,camMat):
    #img = np.array(img,dtype='uint8')
    gmin,gmax = -500,500

    x = np.arange(gmin,gmax,10)

    color = (150,150,150)

    for xi in x:
        y0,y1 = gmin, gmax
        (x0,y0) = np.array(homo2Rect(imageCoordinates((xi,y0,0,1),camMat)),dtype='int')
        (x1,y1) = np.array(homo2Rect(imageCoordinates((xi,y1,0,1),camMat)),dtype='int')
        cv2.line(img,(x0,y0),(x1,y1),color,1,lineType=cv2.CV_AA)
    for yi in np.arange(gmin,gmax,10):
        x0,x1 = gmin, gmax
        (x0,y0) = np.array(homo2Rect(imageCoordinates((x0,yi,0,1),camMat)),dtype='int')
        (x1,y1) = np.array(homo2Rect(imageCoordinates((x1,yi,0,1),camMat)),dtype='int')
        cv2.line(img,(x0,y0),(x1,y1),color,1,lineType=cv2.CV_AA)

    #(x0,y0) = np.array(homo2Rect(imageCoordinates((25,25,0,1),camMat)),dtype='int')
    #cv2.circle(img,(x0,y0),25,(0,0,0),thickness=-1,lineType=cv2.CV_AA)

def latLong(latlong,dr):
    rho = 6371000.0 # 6371km = radius of earth
    dlat = dr[1] / rho
    dlong = dr[0] / (rho*math.cos(latlong[0]*math.pi/180))

    return (latlong[0]+dlat, latlong[1]+dlong)

def drawCrossHair(img, center, size=10, color=(0,0,0),thickness=1): # center in (x,y) format
    cv2.line(img,(center[0] - size,center[1]-size), (center[0]+size,center[1]+size),color,thickness,lineType=cv2.CV_AA)
    cv2.line(img,(center[0] + size,center[1]-size), (center[0]-size,center[1]+size),color,thickness,lineType=cv2.CV_AA)


def verifyPose(anglePerturbation = 0):
    camMat = readCameraMatrix('3D_I/P1')
    K,R,t = factor(camMat)
    #print K,R,t
    for theta in np.linspace(0,math.pi/2-.1,15):

        phi = theta*0.5

        R1 = np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
        R2 = np.array([[math.cos(phi),math.sin(phi),0],[-math.sin(phi),math.cos(phi),0],[0,0,1]])
        R = np.dot(R1,R2)

        K = np.array([[-1000,0,400],[0,-1000,300],[0,0,1]])
        t = np.array([0,0,-1000])
        #t = np.array([0,0,+700*theta-1800])

        #print R
        camMat = constructCameraMatrix(K,R,t)

        img = np.ones((600,800,3),np.uint8)*255
        drawGPSGrid(img,camMat)

        # test circle
        gpsPoint = (100,50,0,1)
        circle1 = np.array(homo2Rect(imageCoordinates(gpsPoint,camMat)),dtype='int')

        #cv2.circle(img,tuple(circle1),3,(50,0,50),lineType=cv2.CV_AA,thickness=1)
        drawCrossHair(img,circle1)

        gpsCalc =  groundCoordinates(circle1,K,R,t)

        circle2 = np.array(homo2Rect(imageCoordinates(gpsCalc,camMat)),dtype='int')

        #cv2.circle(img,tuple(circle2),5,(0,50,50),lineType=cv2.CV_AA,thickness=1)
        drawCrossHair(img,circle2)

        print 'Dist btn set point and calculated point: %s m' % str(np.around(np.linalg.norm(gpsPoint-gpsCalc),3)) 

        # draw perturbed circle and print distance
        if(anglePerturbation > 0):
            # testing angle sensitivity
            theta += anglePerturbation
            R1 = np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
            phi += anglePerturbation
            R2 = np.array([[math.cos(phi),math.sin(phi),0],[-math.sin(phi),math.cos(phi),0],[0,0,1]])
            R3 = np.dot(R1,R2)

            gpsCalcPerturb = groundCoordinates(circle1,K,R3,t)
            circle3 = np.array(homo2Rect(imageCoordinates(gpsCalcPerturb,camMat)),dtype='int')

            cv2.circle(img,tuple(circle3),10,(50,50,50),lineType=cv2.CV_AA,thickness=1)
            #drawCrossHair(img,circle3)

            print 'Dist btn set point and perturbed point with angle {} deg: {} m' \
                    .format(np.around(anglePerturbation*180/math.pi), str(np.around(np.linalg.norm(gpsPoint-gpsCalcPerturb),3)))        

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def gpsCoorImageMask():
    for (img,geom) in zip(imgNamesI,geomNames): # img names and geom names are global vars -- bad!
        camMat = readCameraMatrix('3D_I/' + geom)
        K,R,t = factor(camMat)
        #R = np.array([[0,.866,0.5],[0,-.5,.866],[1,0,0]])
        #R = np.array([[0.5,.866,0],[.866,-.5,0],[0,0,1]])

        # Check image coordinate 0,0 to make sure the transformation is reversible
        xyzw = groundCoordinates((0,0),K,R,t)
        print 'world coordinates (x,y,z)\n' + str(np.around(homo2Rect(xyzw),3)) + '\n'

        # reverse the process to make sure everything is working
        camMat = constructCameraMatrix(K,R,t)
        xiyi = homo2Rect(imageCoordinates(xyzw,camMat))
        print 'image coordinates\n' + str(np.around(xiyi,3)) + '\n'


        img = cv2.imread('images_I/'+img,cv2.IMREAD_GRAYSCALE)

        # begin mapping gps coordinates to image
        gpsCoors = np.zeros((img.shape[0],img.shape[1],2))

        # reference point of world center
        cameraGPSPosition = (37.0,-122.0) #### NOTE wait so if using GPS position then t should be 0 in cam matrix?

        # assigns GPS coordinates to image-esque array
        skipFactor = 10

        for y,x in np.ndindex((gpsCoors.shape[1]/skipFactor,gpsCoors.shape[0]/skipFactor)):
            dx = tuple(groundCoordinates((x*skipFactor,y*skipFactor),K,R,t)[0:2])

            gpsCoors[y*skipFactor,x*skipFactor] = latLong(cameraGPSPosition, dx)

        #print gpsCoors[::skipFactor,::skipFactor,0]

        #cv2.imshow('image',gpsCoors[::skipFactor,::skipFactor,0])
        drawGPSGrid(img,camMat)

        cv2.imshow('image',img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

geomNames = ['P1','P2','P3','P4','P5','P6']
imgNamesI = ['1726_p1_s.pgm','1727_p1_s.pgm','1728_p1_s.pgm','1762_p1_s.pgm','1763_p1_s.pgm','1764_p1_s.pgm']
imgNamesII = ['1726_p1_s1.pgm','1727_p1_s1.pgm','1728_p1_s1.pgm','1762_p1_s1.pgm','1763_p1_s1.pgm','1764_p1_s1.pgm']

verifyPose(5*math.pi/180)

'''
# summary of problem:
I think everything is working??
'''