#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Modelimizin görsel halini diske kaydetmek.
from keras.utils import plot_model 

# Xception Öneğitimli Evrişimli Sinir Ağı 
from keras.applications import xception

# veri çeşitlendirme
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

# eniyileme
from keras import optimizers

# model tanımlamak
from keras.models import Sequential

# çıktı boyutunu düzenleme
from keras.layers import Flatten

# tamamen bağlı katman
from keras.layers import Dense

# iletim sönümü
from keras.layers import Dropout

# kayıpları ve başarımı görselleştirme
import matplotlib.pyplot as plt

from time import sleep as sl

# matris işlemleri
import numpy as np

# klasör ve işletim sistemi
import os, shutil

# keras
import keras

# Yığın normalleştirme
from keras.layers import BatchNormalization


# --- modeli GPU üzerinden eğitiyoruz --- #


# tensrflow ve kerasın GPU versiyonu gerekli
# Nvdia GPU
# CUDA
# cuDNN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# cpu üzerinden eğitimi gerçekleştirmek isterseniz "-1" yapın
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ----------- veri ön işleme (veri klasör yollarını kodda tanımlama) ----------- #


# Test veri seti klasörü
train_dir = os.path.join("Train") 


# Doğrulama veri seti klasörü
validation_dir = os.path.join("Validation")


# Eğitim veri setinde maske takan isnan verileri
train_mask_dir = os.path.join(train_dir,"Mask") 


# Eğitim veri setinde maske takmayan insan verileri
train_no_mask_dir = os.path.join(train_dir,"No_mask")


# Doğrulama veri setinde maske takan isnan verileri
validation_mask_dir = os.path.join(validation_dir,"Mask")


# Doğrulama veri setinde maske takmayan insan verileri
validation_no_mask_dir = os.path.join(validation_dir,"No_mask")


os.system("clear")


# --- Veri seti çeşitlendirme --- #


# veri çeşitlendirme
train_datagen = ImageDataGenerator(

      # resim pixellerini 0,1 arasına sıkıştırma
      rescale=1./255,

      # derece cinsinden (0-180) resimlerin rastgele döndürülme açısı
      rotation_range=40,

      # resimlerin yatayda ve dikeyde kaydırılma oranları
      width_shift_range=0.2,

      # resimlerin yatayda ve dikeyde kaydırılma oranları
      height_shift_range=0.2,

      # burkma işlemi
      shear_range=0.2,

      # yakınlaştırma işlemi
      zoom_range=0.2,

      # dikeyde resim döndürme
      horizontal_flip=True,

      # işlemlerden sonra ortaya çıkan  fazla 
      # görüntü noktalarının nasıl doldurulacağını belirler
      fill_mode='nearest')


# test resimlerinde çeşitlendirme yapmıyoruz.
test_datagen = ImageDataGenerator(rescale=1./255)


# çeşitlendirilmiş verileri kullanmak (eğitim)
train_generator = train_datagen.flow_from_directory(

        # hedef dizin
        train_dir,

        # tüm resimler (150x150) olarak boyutlandırılacak
        target_size=(150, 150),

        # yığın boyutu
        batch_size=20,

        # binary_crossentropy kullandığımız için
        # ikili etiketler gerekiyor.
        class_mode='binary')


# verileri kullanmak (doğrulama)
validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')


sl(5)



# --- Xception Öneğitimli Evrişimli Sinir Ağı oluşturma --- #


"""

artık bağlantı Xception da dahil 2015 sonrasında birçok modelde kullanılan

bir diğer çizgisel ağ yapısıdır. 2015 sonlarında ILSVRC ImageNet yarışmasını kazanan

Microsoft 'tan He vd. tarıfından geliştirilmiştir.

Modelleri her büyük çaplı derin öğrenme modelinin başının belası olan iki yaygın problemle mücadele ediyor:

Gradyan yok olması ve gösterimsel darboğaz

"""


conv_base = xception.Xception(weights="imagenet", include_top=False, input_shape=(150,150,3))


# Öneğitimli Evrişimli Sinir Ağındaki bazı katmanlar haricinde diğer tüm katmanları donduruyoruz.

# Dondurma sebebimiz ise parametre sayısı çok fazla olunca fazla işlem kapasitesi demektir.

# Dondurma sebebimiz Dense katmanları rastgele başlatıldığından eğitim esnasnında çok büyük güncellemeler 
# alacaktır ve buda daha önce öğrenilen gösterimleri yok edecektir.

# Bu haliyle sadece block14_sepconv1 katmanın ve sonraki katmanların ağırlıkları güncellenecek


conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:

	if layer.name == "block14_sepconv1":

		set_trainable = True

	if set_trainable:

		layer.trainable = True

	else:

		layer.trainable = False


# model
model = Sequential()

# modelimize Xception Öneğitimli Evrişimli Sinir Ağını ekledik
model.add(conv_base)

# Yığın normalleştirme
# modelin daha iyi geneleştirme yapmasını sağlar.
# eğitim süresince  verinin ortalaması ve standart sapmasının değişimlerine bakarak veriyi normalize eder. 
model.add(BatchNormalization())

# 3B çıktıları 1B vektörlere düzenler
model.add(Flatten())

# tamamen bağlı katmanlar
model.add(Dense(256, activation="relu"))

# iletim sönümü : 
# modelin aşırı uydurma yapmasını engeller.
# Sinir ağlarının düzleştirilmesinde kullanılır.
# verdiğimiz orana göre elemanları sıfırlar.
model.add(Dropout(0.6))

# fonksiyonu sigmoid olarak kullanarak çıkan değeri [0,1] arasına sıkıştırdık
# çünki ikili sınıflandırma var (MASK, NO MASK)
model.add(Dense(1, activation="sigmoid"))


# modelimizi görüntülemek
print(model.summary())


"""

Layer (type)                 Output Shape              Param #   
=================================================================
xception (Model)             (None, 5, 5, 2048)        20861480  
_________________________________________________________________
batch_normalization_5 (Batch (None, 5, 5, 2048)        8192      
_________________________________________________________________
flatten_1 (Flatten)          (None, 51200)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               13107456  
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       
=================================================================
Total params: 33,977,385
Trainable params: 17,860,609
Non-trainable params: 16,116,776

"""


# --- Modeli derleme ve eğitme --- #


# modeli derleme
model.compile(loss="binary_crossentropy", # kayıp fonksiyonu

              # eniyileme:
              # ağımızın girdisi olan veri ile oluşturduğu kaybı göz önünde
              # bulundurarak kendisini güncelleme mekanizması
			  optimizer=optimizers.RMSprop(lr=2e-5),

			  # eğitim ve test süresince takip edilecek metrikler. 
			  metrics=["acc"])


# modelin görsel halini diske kaydetme
plot_model(model, show_shapes=True, to_file="model.png")


# modeli eğitme
history = model.fit_generator( # acc, loss, val_acc, val_loss değerlerini history adlı değişkenden alacağız.

	  # eğitim verileri
      train_generator,

      # döngü bitene kadar geçeceği örnek sayısı (alınacak yığın)
      steps_per_epoch=175,

      # döngü sayısı
      epochs=100,

      # doğrulama verisi
      validation_data=validation_generator,

      # doğrulama için yığın sayısı
      validation_steps=75,

      verbose=2)

# modelimizi test için kaydettik.
model.save('mask_model.h5')



# --- sonuçları görselleştirme --- #


# Eğitim başarım skoru
acc = history.history["acc"]

# doğrulama başarım skoru
val_acc = history.history["val_acc"]

# eğitim kayıp skoru
loss = history.history["loss"]

# doğrulama kayıp skoru
val_loss = history.history["val_loss"]

# epochs sayısına göre grafik çizdireceğiz.
epochs = range(1, len(acc) + 1)

# eğitim başarımını kendine özel çizdirdik.
plt.plot(epochs, acc, "bo", label="Eğitim başarımı")

# doğrulama başarımını kendine özel çizdirdik.
plt.plot(epochs, val_acc, "b", label="Doğrulama başarımı")

# çizdirmemizin başlığı
plt.title("Eğitim ve doğrulama başarımı")

plt.legend()

plt.figure()

# eğitim kaybını kendine özel çizdirdik.
plt.plot(epochs, loss, "bo", label="Eğitim kaybı")

# doğrulama kaybını kendine özel çizdirdik.
plt.plot(epochs, val_loss, "b", label="Doğrulama kaybı")


# çizdirmemizin başlığı
plt.title("Eğitim ve doğrulama kaybı")

plt.legend()

# ekrana çıkartma
plt.show()
