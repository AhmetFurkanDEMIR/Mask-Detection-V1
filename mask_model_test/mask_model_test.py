import demir_API

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import wget
import os


net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel") # yüz tespiti

vs = VideoStream(src=0).start() # kamera görüntüleri
time.sleep(2.0)

model = load_model("mask_model.h5") # maske tespiti


while True:


	frame = vs.read() 
	frame = imutils.resize(frame, width=700)


	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))


	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):


		confidence = detections[0, 0, i, 2]


		if confidence < 0.5:
			continue


		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		image = cv2.resize(frame[startY:endY, startX:endX], (150, 150)) # tespit edilen yüzü modele dahil etmek için işliyoruz
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		a = model.predict(image) # yüz modele gönderilir çıkan sonuca göre maske var yada yok

		texts = "NONE"

		if a > 0.3: # ekrana yansıtmalar
			
			text = "N O   M A S K"
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		elif a<= 0.3:

			text = "M A S K"
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
			cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()