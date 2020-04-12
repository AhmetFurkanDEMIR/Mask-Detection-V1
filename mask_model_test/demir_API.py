import os

try:

	import imutils
	import cv2
	from keras.preprocessing.image import img_to_array
	from keras.models import load_model
	import wget
	import numpy

except:

	print(" GEREKLİ MODÜLLER KURULUYOR ...")

	os.system("python3 -m pip install --upgrade pip")

	os.system("pip3 install imutils")

	os.system("pip3 install tensorflow")

	os.system("pip3 install keras")

	os.system("pip3 install opencv-python")

	os.system("pip3 install wget")

	os.system("pip3 install numpy")

	os.system("pip3 install argparse")

	print("İŞLEM TAMAMLANDI")

kontrol = None

import wget

if not os.path.isfile(os.path.join("deploy.prototxt.txt")):

	print("\nEKSİK DOSYALAR İNDİRİLİYOR ....")

	kontrol = True 

	file_url1 = "https://www.dropbox.com/s/08hxbqqi5145v5o/deploy.prototxt.txt?dl=1"
	wget.download(file_url1)

if not os.path.isfile(os.path.join("res10_300x300_ssd_iter_140000.caffemodel")):

	print("\nEKSİK DOSYALAR İNDİRİLİYOR ....")

	kontrol = True

	file_url2 = "https://www.dropbox.com/s/m6v8yuymewig62n/res10_300x300_ssd_iter_140000.caffemodel?dl=1"
	wget.download(file_url2)

if not os.path.isfile(os.path.join("mask_model.h5")):
	
	print("\nEKSİK DOSYALAR İNDİRİLİYOR ....")
		
	kontrol = True

	file_url3 = "https://www.dropbox.com/s/gzxmd2uam42kp81/mask_model.h5?dl=1"

	wget.download(file_url3)

if kontrol == True:

	print("\nEKSİK DOSYALAR İNDİRİLDİ")
