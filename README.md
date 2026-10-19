# **Klasifikasi Gambar Hewan dengan CNN (Sequential, Conv2D, Pooling)**

Proyek ini membangun model Convolutional Neural Network (CNN) dari nol untuk mengklasifikasikan gambar hewan ke dalam 4 kelas: **butterfly, cat, chicken, elephant**. Model dilatih di Google Colab, lalu disimpan dalam tiga format: SavedModel, TF-Lite, dan TFJS.

## **Dataset**

* Sumber: Animals10 (Kaggle), `paulbombay/animals10`  
  &nbsp;https\://www\.kaggle.com/datasets/paulbombay/animals10  
* Dataset asli berisi 10 kelas. Proyek ini memakai 4 kelas: `butterfly`, `cat`, `chicken`, `elephant`.  
* Folder `train` dan `test` asli digabung, lalu dibagi ulang dengan jumlah sama di setiap kelas, sehingga kelas tidak timpang.  
* Total gambar yang dipakai: **5600**, dengan pembagian  **train \= 4000/ val \= 800 / test \= 800**  gambar per kelas.  
* Dataset dibagi menjadi tiga set terpisah: train (melatih model), validation (memantau EarlyStopping), dan test (evaluasi akhir).

### **Pemilihan kelas**

Awalnya dicoba 5 kelas. Dari confusion matrix pada beberapa percobaan, kelas `spider` dan `squirrel` konsisten sulit dibedakan dari kelas lain (akurasi per kelas sekitar 0,5 sampai 0,7), sedangkan gambar `spider` sudah diperiksa manual dan labelnya benar. Karena itu jumlah kelas dikurangi menjadi 4 dan jumlah gambar per kelas ditambah.

## **Pra-pemrosesan**

File gambar asli tidak diubah (resolusi gambar asli tidak seragam). Pra-pemrosesan dilakukan di dalam pipeline:

1. **Pemeriksaan data**: file rusak dan gambar duplikat (hasil: 0 rusak, 0 duplikat), serta rentang ukuran gambar.  
2. **Resize** ke **224x224** piksel saat data dimuat.  
3. **Normalisasi** piksel 0-255 menjadi 0-1 (layer `Rescaling` di dalam model).  
4. **Augmentasi** (flip horizontal, rotasi, zoom) sebagai layer di awal model, hanya aktif saat training.

## **Arsitektur model**

Model `Sequential` dengan empat blok. Tiap blok berisi dua `Conv2D` \+ `BatchNormalization` \+ ReLU, lalu `MaxPooling2D` (filter 32, 64, 128, 256). Setelahnya `GlobalAveragePooling2D`, `Dense(128, relu)`, `Dropout(0.3)`, dan `Dense(4, softmax)`.

* Loss: `categorical_crossentropy`, optimizer: `Adam`.  
* Callback: `EarlyStopping` (memantau `val_loss`, `restore_best_weights=True`) dan `ReduceLROnPlateau`.  
* Seed acak: 42\.

## **Hasil**

| Set | Akurasi |
| :---- | :---- |
| Train | 0,9323 |
| Validation | 0,9150 |
| Test | 0,8850 |

Akurasi train dan test sama-sama di atas 85%. Akurasi per kelas pada test set:

| Kelas | Akurasi |
| :---- | :---- |
| butterfly | 0,95 |
| cat | 0,89 |
| chicken | 0,84 |
| elephant | 0,85 |

## **Inference**

Notebook menyertakan sel inference yang memakai `model.tflite` dan `label.txt`: gambar diunggah, model menebak kelasnya, lalu gambar dan persentase prediksi ditampilkan. Bukti inference ada pada output sel tersebut di `notebook.ipynb`.

## **Struktur folder**

submission/

├── tfjs\_model/

│   ├── group1-shard1of1.bin

│   └── model.json

├── tflite/

│   ├── model.tflite

│   └── label.txt

├── saved\_model/

│   ├── saved\_model.pb

│   └── variables/

├── notebook.ipynb

├── README.md

└── requirements.txt

&nbsp;

## **Cara menjalankan**

Notebook dijalankan di Google Colab dengan GPU T4 (Python 3.13). Dataset diunduh otomatis dengan `kagglehub` (tanpa token, dataset publik).

Instalasi paket tambahan:

pip install tf-keras kagglehub importlib\_resources

pip install tensorflowjs==4.20.0 \--no-deps

pip install tensorflow-hub==0.16.1 \--no-deps

&nbsp;

Catatan teknis:

* `tensorflowjs` dipasang dengan `--no-deps` karena dependensinya (`tensorflow-decision-forests`) mensyaratkan versi TensorFlow yang bentrok dengan TensorFlow di Colab. Paket itu tidak dipakai oleh model ini, sehingga di awal notebook dibuat modul kosong pengganti `tensorflow_decision_forests` agar `import tensorflowjs` berhasil.  
* `TF_USE_LEGACY_KERAS=1` diatur sebelum `import tensorflow` supaya konversi ke TFJS berjalan dengan Keras versi lama.  
* Layer augmentasi tidak disertakan pada model ekspor (hanya dipakai saat training). Prediksi model latih dan model ekspor sudah diperiksa identik.

&nbsp;