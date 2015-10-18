import numpy as np
import cv2

def readFrames(videoName,FPS,START_TIME,END_TIME):
	capture = cv2.VideoCapture(videoName) #Cam2.mp4

	START_FRAME = FPS*START_TIME

	END_FRAME = FPS*END_TIME

	count=0
	while count < START_FRAME:
		capture.grab()
		print('Skipping Frame ' + str(count))
		count+= 1

	frames = []

	while(count < END_FRAME):
		ret,frame = capture.read()
		frames.append(frame)
		count+= 1
		print('Capturing Frame ' + str(count))

	capture.release()

	frameNumbers = np.char.mod('%d',np.arange(START_FRAME,END_FRAME))
	return frames, frameNumbers

def writeFrames(frames,colorSpace,frameNames,pathName,cameraName,displayFrames=True):
	if frameNames is None:
		frameNames = np.char.mod('%d',np.arange(1,len(frames)+1))

	filenameCount = 0
	for i in xrange(len(frames)):
		# Convert Image to GrayScale
		grayFrame = cv2.cvtColor(frames[i],colorSpace)

		# Determine the path and name for the file 
		pathFile = pathName + cameraName + '_Frame'+frameNames[i]+'.jpg'
		print('Writing ' + pathFile)
		
		# Write frame to jpg
		cv2.imwrite(pathFile,grayFrame)

		# Show frame in window
		if(displayFrames):
			cv2.imshow('Frame',grayFrame)

		filenameCount+=1

		# Quit on command q
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()


grayFrames, frameTitles = readFrames('Cam1.mp4',30,0,10)
writeFrames(grayFrames,cv2.COLOR_RGB2GRAY,frameTitles,'Cam1/','Cam1',displayFrames=False)


img1 = grayFrames[150]
img2 = grayFrames[150]
