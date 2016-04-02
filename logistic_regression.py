# Contoh model Logistic Regression untuk task Klasifikasi
# Di dokumen akan di bahas langkah-langkah untuk membuat model menggunakan 
# algoritma atau metode logistic regression. Mulai dari training data sampai 
# visualisasi model yang telah di buat.
# 
# bisa dibuka notebooknya di sini: 
# https://pyk.github.io/ai-mentoring-surabaya/pertemuan-2-3

# Pertama-tama kita import beberapa python packages yang kita gunakan
# numpy dan scikit-learn untuk membuat traning dan test data secara acak.
import numpy as np
from sklearn import linear_model, datasets
# matplotlib kita gunakan untuk visualisasi training data, test data dan 
# model yang kita buat
import matplotlib.pyplot as plt

# Langkah 1: Data
# Sebagai contoh kita akan membuat datanya secara acak. Dalam prakteknya, 
# data bisa berasal dari mana saja. Mulai dari nilai khusus seperti harga 
# produk, ukuran dan lain-lain. Biasanya juga berbentuk raw teks atau gambar
# yang kemudian di representasikan sebagai vektor. Untuk representasi data 
# dari raw teks ke vektor akan di bahas lebih lengkap saat sesi feature 
# engineering, sesuai dengan studi kasus yang kita lakukan.
# Kita disini cuma memakai 2 feature supaya mudah di plot di bidang 2D dan 4 
# class. Dalam prakteknya jumlah feature bisa lebih dari 60 ribu dimensi, 
# sehingga sangat sulit untuk di bayangkan secara visual.
data = datasets.make_classification(n_samples=100, n_features=2,
    n_informative=2, n_redundant=0, n_classes=4, n_clusters_per_class=1)
# Untuk keperluan grafik, akan kita simpan nilai x {max,min} dan y {max,min} 
# dari data yang kita generate secara acak
x_min, x_max = data[0][:, 0].min() - .5, data[0][:, 0].max() + .5
y_min, y_max = data[0][:, 1].min() - .5, data[0][:, 1].max() + .5
# Kita split data untuk training(80%) dan test(20%). 
train_input = data[0][:80]
train_output = data[1][:80]
test_input = data[0][-20:]
test_output = data[1][-20:]
# Kita akan coba visualisasikan training data kita ke dalam bidang 2D
plt.figure(1, figsize=(16, 16))
plt.title('Logistic Regression Training Data')
plt.scatter(train_input[:, 0], train_input[:, 1], c=train_output, edgecolors='none',s=50)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()
# Titik-titk di gambar tersebut mewakili satu object yang ingin kita 
# klasifikasikan, misal sebuah teks berita, deskripsi produk dan lain-lain. 
# Setiap object masuk dalam satu kelas. Tiap kelas diwakili dengan warna 
# yang berbeda dengan yang lainnya.
# Intuisi dari pembuatan model ini adalah kita ingin mencari garis pemisah 
# atau area yang meliputi tiap kelas yang berbeda. Sehingga, ketika ada data
# baru atau titik baru yang belum diketahui kelasnya, kita mencoba memprediksi titik tersebut masuk kelas mana sesuai dengan model yang sudah kita buat. 

# visualisasi data test
plt.figure(2, figsize=(16, 16))
plt.title('Logistic Regression Training Data')
plt.scatter(test_input[:, 0], test_input[:, 1], c=test_output, edgecolors='none',s=50)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

# Nah anggap aja kita belum tau test data itu masuk class mana saja.
# Model yang kita buat nantilah yang akan membantu kita menentukan masuk
# class yang mana.

# Langkah 2: Algoritma Machine Learning
# Dalam contoh ini, kita akan menggunakan Logistic regression.
logreg = linear_model.LogisticRegression()

# Langkah 3: Model
# Model akan dibuat berdasarkan train_{input,output} data
logreg.fit(train_input, train_output)

# Langkah 3: visualisasi model
h = .01 # nilai tambah untuk tiap iterasi di meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(3, figsize=(16, 16))
plt.title('Model Logistic Regression')
plt.pcolormesh(xx, yy, Z)
plt.scatter(train_input[:, 0], train_input[:, 1], c=train_output, edgecolors='k',s=50)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()


# Gambar diatas merepresentasikan model pertama kita, cukup bagus lah ya 
# untuk 100 iterasi. Ada beberapa object yang overlap, dan inilah kelemahan 
# model linear seperti logistic regression.
# Kita bisa memperbaiki model dengan meningkatkan jumlah iterasi, 
# memperkecil toleransi dan lain-lain. Istilahnya adalah tune-in model.
# Mari kita lihat model dan test data kita.
plt.figure(3, figsize=(16, 16))
plt.title('Model Logistic Regression + Test Data')
plt.pcolormesh(xx, yy, Z)
plt.scatter(test_input[:, 0], test_input[:, 1], c=test_output, edgecolors='k',s=50)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

# hmmm, lumayan lah ya.
# Anyway, kita udah dapet nih intuisi dasar machine learning. Pertama ada 
# data, terus langkah berikutnya apa udah di ilustrasikan disini.
# Untuk evaluasi algoritmanya akan di bahas lebih dalam pada sesi Evaluasi 
# Algoritma pada program mentoring.
# Follow https://artificialintelligence.id untuk updatenya.
# Regards, editorial@artificialintelligence.id
