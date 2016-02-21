
#
# Automated APGAR Scoring
# Authored by Rajath Kumar M P
# on 21 Feb 2016
#

import cv2
import numpy as np
import time
from gcm import *

def appearance(filename):
	
	img = cv2.imread(filename)

	cv2.imshow('Input-Image', img)
	time.sleep(2)

	lower = np.array([105, 90, 120], dtype=np.uint8)
	upper =np.array([195,160,170], dtype=np.uint8) 

	mask = cv2.inRange(img, lower, upper)
	output = cv2.bitwise_and(img, img, mask = mask)


	cv2.imshow('Output-Image', output)
	time.sleep(2)
	cv2.destroyAllWindows()

	shape1 = img.shape

	count = 0
	for i in range(shape1[0]):
		for j in range(shape1[1]):
			a = output[i][j][1]
			if a > 0:
				count+=1

	perc = (float(count)/(shape1[0]*shape1[1]))*100

	global app

	if perc>0 and perc<8:
		app = 2
	elif perc>8 and perc<16:
		app = 1
	else:
		app = 0

	print 'Appearance:', app

	

def grimace(filename):

	import argparse,sys

	try:
		from FeatureGen import findRatio, generateFeatures
	except ImportError:
		exit()

	try:
		import dlib
		from skimage import io
		import numpy
		import cv2
		from sklearn.externals import joblib
	except ImportError:
		exit()

	emotions={ 1:"Anger", 2:"Contempt", 3:"Disgust", 4:"Sleep", 5:"Happy", 6:"Sadness", 7:"Cry"}

	def Predict_Emotion(filename):

		try:
			img=io.imread(filename)
			cvimg=cv2.imread(filename)
		except:
			return

		win.clear_overlay()
		win.set_image(img)


		dets=detector(img,1)

		if len(dets)==0:
			print "Unable to find any face."
			return

		for k,d in enumerate(dets):

			shape=predictor(img,d)
			landmarks=[]
			for i in range(68):
				landmarks.append(shape.part(i).x)
				landmarks.append(shape.part(i).y)
		
	
			landmarks=numpy.array(landmarks)
	
			features=generateFeatures(landmarks)
			features= numpy.asarray(features)

			pca_features=pca.transform(features)

			emo_predicts=classify.predict(pca_features)

			global fin

			fin = emotions[int(emo_predicts[0])]
			print 'Grimace:', fin

			#font = cv2.FONT_HERSHEY_SIMPLEX
			#cv2.putText(cvimg,emotions[int(emo_predicts[0])],(20,20), font, 1,(0,255,255),2)

			win.add_overlay(shape)

		cv2.namedWindow("Output")
		cv2.imshow("Output",cvimg)
		time.sleep(2)
		cv2.destroyAllWindows()

		


	if __name__ == "__main__":


		landmark_path="shape_predictor_68_face_landmarks.dat"

		detector= dlib.get_frontal_face_detector()

		try:
			predictor= dlib.shape_predictor(landmark_path)
		except:
			exit()

		win=dlib.image_window()


		try:
			classify=joblib.load("traindata.pkl")
			pca=joblib.load("pcadata.pkl")
		except:
			exit()

		Predict_Emotion(filename)





def activity(filename):

	from skimage.measure import structural_similarity as ssim
	import matplotlib.pyplot as plt

	cap = cv2.VideoCapture(filename)

	count = 0
	a = []

	try:
		while(cap.isOpened()):
			ret, frame = cap.read()

			if ret == False:
				break

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			count+=1
			a.append(gray)
			cv2.imshow('frame',gray)
			
	except:
		exit()


	cap.release()
	cv2.destroyAllWindows()

	ab = []
	for i in range(count - 1):
		ab1 = cv2.subtract(a[i],a[i+1])
		ab.append(ab1)

	s = []
	for i in range(len(ab)-1):
		s1 = ssim(ab[i],ab[i+1])
		s.append(s1)



	global finstr

	finstr = ''.join(str(x)+',' for x in s)


	gcm = GCM("AIzaSyBEWD4gOE8ZIuG2KXMSvmuS7PAieYP3LYc")

	data = {'appearance': app, 'grimace': fin, 'activity': finstr }

	reg_id = 'dAd0v68RHQc:APA91bHyxUB4VhMqkcL5uVkiVAhgjXIReocK9jIrjJWtUOr4LpF8VDVBILDQT54oTMshnxj30MejTfauGEpg4ZWfdf3IJRC0nULx3S1LxWZhiZs8qN6CuASHQZjbYHTevzAWgjcjvg8w'

	gcm.plaintext_request(registration_id=reg_id, data=data)

	#fig = plt.figure(1)
	#plt.ylim((0,1))
	#plt.plot(range(len(s)),s)
	#plt.show()
	#time.sleep(2)
	#plt.close(fig)




if __name__ == "__main__":

	appearance('appearance/1.png')
	grimace('grimace/2.jpg')
	activity('activity/3.gif')




