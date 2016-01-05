import csv
import math
import cv2
import numpy as np


def factorCameraMatrix(P):
    """
    This method factors a given camera matrix P into its factors K (intrinsic),
    R (world axes in cam ref frame), and t (world center in camera ref frame)

    :param P:
    :return:
    """
    K, R = np.linalg.rq(P[:, :3])
    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1

    K = np.dot(K, T)
    R = np.dot(T, R)
    t = np.dot(np.linalg.inv(K), P[:, 3])  # this gives t not C
    return K, R, t


def normHomo(V):
    """
    Normalize homo coordinates

    :param V:
    :return:
    """
    return V / V[-1]


def homo2Rect(V):
    """
    Converts homogeneous coordinates into rectangular coordinates

    :param V:
    :return:
    """
    return V[:-1] / V[-1]


def rect2Homo(V):
    """
    Converts rectangular coordinates into homogeneous coordinates with w=1

    :param V:
    :return:
    """
    return np.append(V, 1)


def constructCameraMatrix(K, R, t):
    """
    Constructs a camera matrix P from K, R, and t

    :param K:
    :param R:
    :param t:
    :return:
    """
    Rt = np.concatenate((R, t.reshape(3, 1)), 1)
    camMat = np.dot(K, Rt)
    return camMat


def readCameraMatrix(filename):
    """
    Reads a camera matrix from a given file

    :param filename:
    :return:
    """
    with open(filename, 'rU') as f:
        reader = csv.reader(f, delimiter=' ')

        camMat = [np.array(row) for row in reader]

        camMat = np.array([map(float,
                               row[np.where(row != '')]) for row in camMat])

        return camMat


def transformationMatrixRt(R, t):
    """
    Returns the transformation matrix corresponding to the given R and t

    :param R:
    :param t:
    :return:
    """
    trans = np.concatenate((R, t.reshape(3, 1)), 1)
    # print trans
    trans = np.concatenate((trans, np.array([[0, 0, 0, 1]])), axis=0)

    return trans


def groundCoordinatesNew(Xi, K, R, t):
    """
    The important function! returns the world coordinates of an image pixel given the camera's KRt

    :param Xi:
    :param K:
    :param R:
    :param t:
    :return:
    """
    return groundCoordinates(Xi, constructCameraMatrix(K, R, t))


def groundCoordinatesNew(Xi, P):
    PModified = np.delete(P, 2, 1)

    PModifiedPrime = np.linalg.inv(PModified)

    groundXY = homo2Rect(np.dot(PModifiedPrime, rect2Homo(Xi)))

    return normHomo(np.append(groundXY, [0, 1]))


def groundCoordinates((xi, yi), K, R, t):
    """
    The important function! returns the world coordinates of an image pixel given the camera's KRt.
    This is the OLD algorithm because I wrote a newer more simple one. I can't believe I spent so
    much time on this one when the above one works. It's a little bit reading a 50 page wikipedia
    article to find one sentence when you could have just command-f-ed it. Except this is with
    math and not words.

    :param K:
    :param R:
    :param t:
    :return:
    """
    wi = 1
    # print 'image coordinates:\n' + str((xi,yi,wi)) + '\n'
    invK = np.linalg.inv(K)

    # print 'image frame coordinates: \n' + str(xcyczc)
    C = -np.dot(R.T, t)

    # print C
    # skew is not taken into account here
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    zc = -C[2] / (R[0, 2] * (xi - cx) / fx + R[1, 2] * (yi - cy) / fy + R[2, 2])
    xcyczc = zc * np.dot(invK, np.array([xi, yi, wi]))

    # print 'xycyzc\n {} \n'.format(xcyczc)

    # print 'reverse 3d image coordinates to get pixel coordinates\n' + str(np.around(homo2Rect(np.dot(K,xcyczc)),2)) + '\n'

    xcyczcwc = rect2Homo(xcyczc)

    inverseTransformationMatrix = np.linalg.inv(transformationMatrixRt(R, t))

    xyzw = np.dot(inverseTransformationMatrix, xcyczcwc)

    # this line does the opposite transformation, should not be used:
    # xyzw = np.dot(transformationMatrixRt(R,t),xcyczcwc)

    # print 'world coordinates (x,y,z)\n' + str(np.around(homo2Rect(xyzw))) + '\n'

    return normHomo(xyzw)


def imageToGroundHomography(P):
    return np.linalg.inv(groundToImageHomography(P))


def groundToImageHomography(P):
    return np.delete(P, 2, 1)


def imageCoordinates(Xw, P):
    """
    Returns the image coordinates of a given world coordinate. Xw is coordinates in world frame. P is camera matrix.

    :param Xw:
    :param P:
    :return:
    """
    return normHomo(np.dot(P, Xw))


def drawRectGrid(img, color=(150, 150, 150), step=10):
    """
    Draws lines parallel to world x and y onto an image given the img and camera matrix

    :param img:
    :param color:
    :param step:
    """
    xmax = img.shape[1]
    ymax = img.shape[0]
    x = np.arange(start=0, stop=xmax, step=step)
    y = np.arange(start=0, stop=ymax, step=step)

    for xi in x:
        (x0, y0) = (xi, 0)
        (x1, y1) = (xi, ymax)
        cv2.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv2.CV_AA)

    for yi in y:
        (x0, y0) = (0, yi)
        (x1, y1) = (xmax, yi)
        cv2.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv2.CV_AA)

        # (x0,y0) = np.array(homo2Rect(imageCoordinates((25,25,0,1),camMat)),dtype='int')
        # cv2.circle(img,(x0,y0),25,(0,0,0),thickness=-1,lineType=cv2.CV_AA)


def drawGPSGrid(img, camMat):
    # img = np.array(img,dtype='uint8')
    gmin, gmax = -500, 500

    x = np.arange(gmin, gmax, 10)

    color = (150, 150, 150)

    for xi in x:
        y0, y1 = gmin, gmax
        (x0, y0) = np.array(homo2Rect(imageCoordinates((xi, y0, 0, 1), camMat)), dtype='int')
        (x1, y1) = np.array(homo2Rect(imageCoordinates((xi, y1, 0, 1), camMat)), dtype='int')
        cv2.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv2.CV_AA)
    for yi in np.arange(gmin, gmax, 10):
        x0, x1 = gmin, gmax
        (x0, y0) = np.array(homo2Rect(imageCoordinates((x0, yi, 0, 1), camMat)), dtype='int')
        (x1, y1) = np.array(homo2Rect(imageCoordinates((x1, yi, 0, 1), camMat)), dtype='int')
        cv2.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv2.CV_AA)

        # (x0,y0) = np.array(homo2Rect(imageCoordinates((25,25,0,1),camMat)),dtype='int')
        # cv2.circle(img,(x0,y0),25,(0,0,0),thickness=-1,lineType=cv2.CV_AA)


def drawCrossHair(img, center, size=10, color=(0, 0, 0), thickness=1):  # center in (x,y) format
    """
    Draws a cross hair on an image

    :param img:
    :param center:
    :param size:
    :param color:
    :param thickness:
    """
    cv2.line(img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, thickness,
             lineType=cv2.CV_AA)
    cv2.line(img, (center[0] + size, center[1] - size), (center[0] - size, center[1] + size), color, thickness,
             lineType=cv2.CV_AA)


def nothing():
    pass


def verifyPose(anglePerturbation=0):
    """
    Cycles through a bunch of different camera poses to verify that the image -> world coordinate algorithm works

    :param anglePerturbation:
    """
    cv2.namedWindow('image')
    cv2.createTrackbar('bye', 'image', 30, 100, nothing)

    K = np.array([[-1000, 1, 400], [0, -1000, 300], [0, 0, 1]])
    t = np.array([0, 0, -1000])

    while (1):
        theta = float(cv2.getTrackbarPos('bye', 'image')) / 100.0 * math.pi
        phi = theta * .05
        # for theta in np.linspace(0, math.pi / 2 - .1, 15):

        phi = theta * 0.5

        R1 = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        R2 = np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])
        R = np.dot(R1, R2)
        # t = np.array([0,0,+700*theta-1800])

        camMat = constructCameraMatrix(K, R, t)

        # initialize image and draw the gps grid on it
        img = np.ones((600, 800, 3), np.uint8) * 255
        drawGPSGrid(img, camMat)

        # draw this point on the grid
        worldPoint = (150, 50, 0, 1)

        # calculate image coordinate
        circle1 = np.array(homo2Rect(imageCoordinates(worldPoint, camMat)), dtype='int')

        # cv2.circle(img,tuple(circle1),3,(50,0,50),lineType=cv2.CV_AA,thickness=1)
        drawCrossHair(img, circle1)

        # from the image point calculate the world coordinates (image->world)
        gpsCalc = groundCoordinatesNew(circle1, camMat)

        # now using those world coordinates calculate the image coordinates (world -> image)
        circle2 = np.array(homo2Rect(imageCoordinates(gpsCalc, camMat)), dtype='int')

        # cv2.circle(img,tuple(circle2),5,(0,50,50),lineType=cv2.CV_AA,thickness=1)
        drawCrossHair(img, circle2)

        print 'Dist btn set point and calculated point: %s m' % str(np.around(np.linalg.norm(worldPoint - gpsCalc), 3))

        # draw perturbed circle and print distance
        if (anglePerturbation > 0):
            # testing angle sensitivity
            theta += anglePerturbation
            R1 = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
            phi += anglePerturbation
            R2 = np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])
            R3 = np.dot(R1, R2)

            gpsCalcPerturb = groundCoordinates(circle1, K, R3, t)
            circle3 = np.array(homo2Rect(imageCoordinates(gpsCalcPerturb, camMat)), dtype='int')

            cv2.circle(img, tuple(circle3), 10, (50, 50, 50), lineType=cv2.CV_AA, thickness=1)
            # drawCrossHair(img,circle3)

            print 'Dist btn set point and perturbed point with angle {} deg: {} m' \
                .format(np.around(anglePerturbation * 180 / math.pi),
                        str(np.around(np.linalg.norm(worldPoint - gpsCalcPerturb), 3)))

        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


def gpsCoorImageMask():
    """
    Deprecated
    """
    for (img, geom) in zip(imgNamesI, geomNames):  # img names and geom names are global vars -- bad!
        camMat = readCameraMatrix('3D_I/' + geom)

        # K,R,t = factor(camMat)
        # R = np.array([[0,.866,0.5],[0,-.5,.866],[1,0,0]])
        # R = np.array([[0.5,.866,0],[.866,-.5,0],[0,0,1]])

        # Check image coordinate 0,0 to make sure the transformation is reversible
        xyzw = groundCoordinatesNew((0, 0), camMat)
        print 'world coordinates (x,y,z)\n' + str(np.around(homo2Rect(xyzw), 3)) + '\n'

        # reverse the process to make sure everything is working

        xiyi = homo2Rect(imageCoordinates(xyzw, camMat))
        print 'image coordinates\n' + str(np.around(xiyi, 3)) + '\n'

        img = cv2.imread('images_I/' + img, cv2.IMREAD_GRAYSCALE)

        # begin mapping gps coordinates to image
        gpsCoors = np.zeros((img.shape[0], img.shape[1], 2))

        # reference point of world center
        cameraGPSPosition = (37.0, -122.0)  #### NOTE wait so if using GPS position then t should be 0 in cam matrix?

        # assigns GPS coordinates to image-esque array
        skipFactor = 10

        for y, x in np.ndindex((gpsCoors.shape[1] / skipFactor, gpsCoors.shape[0] / skipFactor)):
            dx = tuple(groundCoordinates((x * skipFactor, y * skipFactor), K, R, t)[0:2])

            gpsCoors[y * skipFactor, x * skipFactor] = latLong(cameraGPSPosition, dx)

        # print gpsCoors[::skipFactor,::skipFactor,0]

        # cv2.imshow('image',gpsCoors[::skipFactor,::skipFactor,0])
        drawGPSGrid(img, camMat)

        cv2.imshow('image', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def scaleImageAboutCenterMatrix(img, scale):
    width = img.shape[1]
    height = img.shape[0]

    T = np.float32([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
    S = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    M = np.dot(S, T)
    NT = np.float32([[1, 0, width / 2], [0, 1, height / 2]])  # last matrix should be 2x3
    M = np.dot(NT, M)
    return M


def latLong(referencePt, dr):
    """
    Returns the latitude and longitude of a point in world coordinates dr using the set point latlong
    :param referencePt: GPS coordinates of reference point
    :param dr: world coordinate offset (first index is x offset in meters, second index is y offset in meters)
    :return: GPS coordinates of new point

    Remember latitude is measured up/down from the equator, not down from the pole like in polar coordinates
    """
    rho = 6371000.0  # 6371km = radius of earth

    # dr is essentially the length of an arc segment so the angle traced by dr is dr/rho
    dlat = dr[1] / rho

    # instead of dividing by rho, divide by r (top down projection of rho onto xy plane)
    dlong = dr[0] / (rho * math.cos(referencePt[0] * math.pi / 180.0))

    return (referencePt[0] + dlat, referencePt[1] + dlong)

def worldCoordinates(referenceGPSPt,GPSPt):
    """
    :param referenceGPSPt: gps pt with alt = 0 that represents 0,0,0 in the world rect frame
    :param GPSPt: the gps pt to map to world rect frame
    :return: world rect frame coordinates in x,y,z
    """
    rho = 6371000.0
    dlat = GPSPt[0] - referenceGPSPt[0]
    dlong = GPSPt[1] - referenceGPSPt[1]

    dy = dlat*rho

    # r is the projection of rho onto the equatorial plane
    r = math.cos(referenceGPSPt[0] * math.pi/180.0)
    dx = dlong*r

    z = GPSPt[2]
    return (dx,dy,z)

def transformGroundPhoto(img, P):
    """
    :param img: The image of the ground plane to orthorectify
    :param P: The camera matrix
    :return: The orthorectified image
    """
    # print 'Size of input image: {}'.format(img.shape)
    height, width = img.shape[0:2]
    # print height,width

    # get the homography that maps between the image and the world map
    H = imageToGroundHomography(P)

    # get a translation matrix that shifts the world map image so that the origin is at the center
    T = np.float32([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]])
    # T = np.eye(3)
    # print T

    # compute the new homography so the world center is at the center of the world map image
    M = np.dot(T, H)

    # warp the image
    img = cv2.warpPerspective(img, M, (width, height))  # kind of strange to do width,height and not height,width

    return img


def transformGroundPhoto(img, K, R, c):
    t = -np.dot(R, c)
    P = constructCameraMatrix(K, R, t)
    return transformGroundPhoto(img, P)


def getRotationMatrix3DYPR(yaw, pitch, roll):  # this is incorrect
    # first rotation
    yawMat = np.array([[math.cos(yaw), -math.sin(yaw), 0], \
                       [math.sin(yaw), math.cos(yaw), 0], \
                       [0, 0, 1]])

    pitchMat = np.array([[1, 0, 0], \
                         [0, math.cos(pitch), math.sin(pitch)], \
                         [0, -math.sin(pitch), math.cos(pitch)]])

    rollMat = np.array([[math.cos(roll), 0, math.sin(roll)], \
                        [0, 1, 0], \
                        [-math.sin(roll), 0, math.cos(roll)]])

    R = np.dot(pitchMat, yawMat)
    R = np.dot(rollMat, R)
    return R


def mapBackyard(writeImages=False):
    """
    :param writeImages: boolean, tells whether to write rectified images to disk or not
    :return: nothing
    Orthorectifies and displays images from my yard
    """
    # P = readCameraMatrix('3D_I/P1')
    backyardNames = np.array(['P0T0_5.jpg', 'P0T45.jpg', 'P30T0_5.jpg', 'P31T46.jpg', 'P41T0.jpg', \
                              'P45T46.jpg', 'P51T180.jpg', 'P56T47.jpg', 'P56T314.jpg'])
    pitches = np.float32([0, 0, 30, 31, 41, 45, 51, 56, 56])
    thetas = np.float32([0.5, 45, 0.5, 46, 0, 46, 180, 47, 314])

    images = zip(backyardNames, pitches, thetas)

    # intrinsic for Olympus at 4/3 at 1200 x 900
    K = np.array([[1.00137397e+03, 0.00000000e+00, 5.84613792e+02],
                  [0.00000000e+00, 9.99010269e+02, 4.37792216e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    for image in images:
        yaw = image[2] * math.pi / 180
        pitch = image[1] * math.pi / 180
        img = cv2.imread('OlympusUD/{}'.format(image[0]))

        R = getRotationMatrix3DYPR(yaw=yaw, pitch=pitch, roll=0)

        # world center in camera coordinates
        # these units set the units for everything else. each pixel in the gps image will correspond to one unit (ie mm or m)

        # hard code the camera center to 200 units up so that the ground map fills the screen
        # (the actual z height will be determined correctly for actual use)
        # C is the camera center in world frame
        C = np.array([0, 0, 200])
        t = -np.dot(R, C)
        # print 'T (world center in camera coordinates): {}'.format(t)
        P = constructCameraMatrix(K, R, t)
        # print 'Coordinates of spot in center of image: {}'.format(groundCoordinatesNew((0, 0), P))

        img2 = transformGroundPhoto(img, P)

        # resize so i can display it properly on my laptop screen
        img2 = cv2.resize(img2, (800, 600))

        # display original photo
        cv2.namedWindow('Original')
        cv2.moveWindow('Original', 800, 0)
        imgResized = cv2.resize(img, (800, 600))
        cv2.imshow('Original', imgResized)

        # show transformed photo
        cv2.namedWindow('Map')
        cv2.imshow('Map', img2)

        if writeImages:
            cv2.imwrite('backyard/' + image[0][0:-4] + '_M.jpg', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


geomNames = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
imgNamesI = ['1726_p1_s.pgm', '1727_p1_s.pgm', '1728_p1_s.pgm', '1762_p1_s.pgm', '1763_p1_s.pgm', '1764_p1_s.pgm']
imgNamesII = ['1726_p1_s1.pgm', '1727_p1_s1.pgm', '1728_p1_s1.pgm', '1762_p1_s1.pgm', '1763_p1_s1.pgm',
              '1764_p1_s1.pgm']

# intrinsic matrix for Go Pro 5MP at MED FOV

K = np.array([[1.43231702e+03, 0.00000000e+00, 1.28269633e+03],
              [0.00000000e+00, 1.43306970e+03, 9.47284290e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# verifyPose(anglePerturbation=0)

mapBackyard(writeImages=False)

'''
Use Case

Micasense:
1. Measure K for rededge
2. Record gimbal orientation
3. Turn rededge images into readable format with GPS coordinates
4. Choose a GPS coordinate as the world center
5. For each image:
    5.1 Find world rect coordinates of the camera center using the worldCoordinates() method
    5.1 Use transformGroundPhoto() method to orthorectify image using K, gimbal orientation, and world rect coordinate
6. Run through data set and find pixel neighborhood surrounding each water measurement
7. Math it out and analyze!

Dual GoPro
1. Combine images and combine channels
2. Attach GPS coordinates to images by matching time logs
3. Follow the steps for Micasense
'''
