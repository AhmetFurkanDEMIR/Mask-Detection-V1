![licence](https://img.shields.io/badge/Keras-V2.3.1-red)
![licencee](https://img.shields.io/badge/Tensorflow-V2.0-yellow)
![licence](https://img.shields.io/badge/demir-ai-blueviolet)
![licence](https://img.shields.io/badge/Ahmet%20Furkan-DEM%C4%B0R-blue)

# Mask Detection V1

* Güvenlik amaçlı marketlerde, otobüslerde ve sokaklarda insanların maske takıp takmadıklarını kontrol etmek için yazdığım proje.
* Projenin ilk sürümüdür, eksikliklerini kapatmak ve tüm maske türleri için tespit yapması üzerinde çalışıyorum.


# Datasets

* 500 adet maske takan ve 600 adet maske takmayan insan yüzleri vardır. 
* Toplamda 1100 adet veri vardır
* Verileri internet üzerinden toplayıp kırptım.
* Verileri Kaggle üzerinde saklamaktayım
* Veri seti = www.kaggle.com/dataset/fc2133756a6f808aad3c673a4eda70ba5b94adde6c953f1d8936030b94ffd2ae
* Veri seti API = kaggle datasets download -d ahmetfurkandemr/mask-datasets-v1

![Screenshot_2020-04-12_21-00-08](https://user-images.githubusercontent.com/54184905/79076187-d7aafa00-7d00-11ea-8354-8ec26e282dd1.png)

# Mask Detection - Face Detection

* Cihazınızda Python 3.5 ve üzeri kurulu olması gerekmektedir.
* Modeli test etmek için "mask_model_test" klasörü içindeki "mask_model_test.py" adlı Python dosyasını çalıştırmanız yeterli olacaktır.
* Bu dosya ile aynı konumda bulunan "demir_API" adlı Python dosyası, otomatik olarak sizin için gerekli olan Pip leri kuracaktır.
* Bu API aynı zamanda Maske tespiti için eğittiğim modelleri internet üzerinden indirecektir.
* Tüm işlemler tamamlandıktan sonra kamera açılacaktır ve Maske tespiti işlemi yapmaya başlayacaktır.
