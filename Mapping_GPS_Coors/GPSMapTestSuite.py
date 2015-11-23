import GPSCoorTest
import numpy as np
import csv
import cv2
import math

def testImageWorldImage():
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K = np.array([[-1000,1,400],[0,-1000,300],[0,0,1]])
    t = np.array([0,0,-1000])
    P = GPSCoorTest.constructCameraMatrix(K,R,t)

    image1 = (15,0)
    gps1 = GPSCoorTest.groundCoordinates(image1,K,R,t)
    gps2 = GPSCoorTest.groundCoordinatesNew(image1,P)

    image2 = GPSCoorTest.imageCoordinates(gps2,P)
    print 'testing image->world->image: {} -> {} -> {}'.format(image1,gps2,image2)

