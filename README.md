# Deteksi Penyakit Daun Padi Menggunakan MobileNetV3 + Classifier Konvensional

**Kelas**: 2024-C  
**Ketua**: Sandi Ardi Prayitno (24031554037)  
**Anggota**: Anggoro Daru Prasetyo (24031554070)  
**Topik**: Pangan (Rumusan Masalah 1.3 dan 1.2)  
**GitHub**: [https://github.com/sandi-ardi/Machine-Learning](https://github.com/sandi-ardi/Machine-Learning)

---

## 1. Pendahuluan

### 1.1 Latar Belakang
Padi adalah komoditas utama di Indonesia. Penyakit seperti Brown Spot, Hispa, Leaf Blast, dan kondisi Healthy (daun sehat) perlu dibedakan untuk mendukung deteksi dini. Petani masih mendeteksi secara manual yang lambat dan tidak efisien. Penelitian ini mengimplementasikan pembelajaran mesin untuk deteksi otomatis kondisi daun padi.

### 1.2 Tujuan
- Eksplorasi dan preprocessing dataset citra daun padi.
- Ekstraksi fitur visual menggunakan MobileNetV3Large pre-trained.
- Perbandingan performa MLP, Random Forest, SVM, dan XGBoost.
- Evaluasi dengan akurasi, presisi, recall, F1-score, dan confusion matrix.
- Uji coba model terbaik pada data uji eksternal.

---

## 2. Akuisisi Data

**Sumber**: [Rice Diseases Image Dataset - Kaggle](https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset/data)  
**Format**: Citra JPEG, RGB  
**Struktur**:
- `LabelledRice/Labelled` – 3355 gambar, digunakan sebagai data uji eksternal
- `RiceDiseaseDataset/train` – data latih
- `RiceDiseaseDataset/validation` – data validasi

Tidak dilakukan scraping sendiri (dataset publik sudah memadai).

---

## 3. Preprocessing dan Eksplorasi Data

### 3.1 Import Library
Kode menggunakan `os`, `PIL`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, dan `sklearn`.

### 3.2 Distribusi Kelas

| Kelas       | Train | Validation | LabelledRice |
|-------------|-------|------------|--------------|
| BrownSpot   | 400   | 123        | 523          |
| Healthy     | 400   | 123        | 1488         |
| Hispa       | 400   | 123        | 565          |
| LeafBlast   | 400   | 123        | 779          |
| **TOTAL**   | **1600** | **492** | **3355**  |

Dataset train dan validasi seimbang antar kelas. LabelledRice memiliki distribusi tidak seimbang dengan kelas Healthy mendominasi.

### 3.3 Contoh Gambar per Kelas
Visualisasi sampel gambar dari setiap kelas ditampilkan untuk dataset train, validation, dan LabelledRice menggunakan fungsi `show_samples()`.

### 3.4 Distribusi Ukuran Gambar Sebelum Resize
Gambar memiliki variasi resolusi. Histogram disimpan di `eda_ukuran_train.png`, `eda_ukuran_validation.png`, `eda_ukuran_labelled.png`.

### 3.5 Resize ke 224×224
Semua gambar diubah ukuran menjadi 224×224 menggunakan `Image.BILINEAR`, disimpan ke `dataset_subset/train_resized`, `validation_resized`, `labelled_resized`.

### 3.6 Augmentasi Data (hanya untuk training)
Menggunakan `ImageDataGenerator` dengan `rotation_range=20`, `horizontal_flip=True`, `zoom_range=0.15`, `fill_mode='nearest'`. Data validasi dan uji tidak diaugmentasi.

---

## 4. Ekstraksi Fitur dengan MobileNetV3Large

- **Model**: `MobileNetV3Large` dengan bobot ImageNet, `include_top=False`, `pooling='avg'`.
- **Output fitur**: 960 dimensi.
- **Proses**: Semua gambar (train, validation, external test) dilewatkan ke model tanpa pelatihan ulang (`trainable=False`).
- **Hasil**: `X_train` (1600, 960), `X_val` (492, 960), `X_ext` (3355, 960). Label disimpan sebagai integer.
- File `.npy` disimpan untuk efisiensi.

---

## 5. Pelatihan Classifier Konvensional

Classifier dilatih pada fitur `X_train` dan label `y_train`.

| Model | Konfigurasi |
|-------|-------------|
| **MLP** | 3 hidden layer (512,256,128), ReLU, Adam, early stopping, StandardScaler |
| **Random Forest** | 300 trees, max_depth=None, n_jobs=-1 |
| **SVM** | Kernel RBF, C=10, gamma='scale', StandardScaler |
| **XGBoost** | 300 estimators, max_depth=6, lr=0.1, subsample=0.8, colsample=0.8 |

**Hasil training:**

| Classifier    | Train Acc | Val Acc | Waktu (s) |
|---------------|-----------|---------|-----------|
| XGBoost       | 1.0000    | 0.4329  | 44.8      |
| Random Forest | 1.0000    | 0.4289  | 1.3       |
| SVM           | 0.8519    | 0.4106  | 7.5       |
| MLP           | 0.4788    | 0.3841  | 2.7       |

---

## 6. Evaluasi pada Data Validasi

### 6.1 Classification Report per Model

**MLP**
| Kelas      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| BrownSpot  | 0.4800    | 0.4878 | 0.4839   | 123     |
| Healthy    | 0.3164    | 0.4553 | 0.3733   | 123     |
| Hispa      | 0.3492    | 0.3577 | 0.3534   | 123     |
| LeafBlast  | 0.4531    | 0.2358 | 0.3102   | 123     |
| **Macro avg** | 0.3997 | 0.3841 | 0.3802   | 492     |

**Random Forest**
| Kelas      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| BrownSpot  | 0.5152    | 0.5528 | 0.5333   | 123     |
| Healthy    | 0.3750    | 0.3415 | 0.3574   | 123     |
| Hispa      | 0.4028    | 0.4715 | 0.4345   | 123     |
| LeafBlast  | 0.4135    | 0.3496 | 0.3789   | 123     |
| **Macro avg** | 0.4266 | 0.4289 | 0.4260   | 492     |

**SVM**
| Kelas      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| BrownSpot  | 0.5481    | 0.6016 | 0.5736   | 123     |
| Healthy    | 0.3232    | 0.2602 | 0.2883   | 123     |
| Hispa      | 0.3609    | 0.3902 | 0.3750   | 123     |
| LeafBlast  | 0.3840    | 0.3902 | 0.3871   | 123     |
| **Macro avg** | 0.4041 | 0.4106 | 0.4060   | 492     |

**XGBoost**
| Kelas      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| BrownSpot  | 0.5645    | 0.5691 | 0.5668   | 123     |
| Healthy    | 0.3864    | 0.4146 | 0.4000   | 123     |
| Hispa      | 0.3864    | 0.4146 | 0.4000   | 123     |
| LeafBlast  | 0.3942    | 0.3333 | 0.3612   | 123     |
| **Macro avg** | 0.4329 | 0.4329 | 0.4320   | 492     |

### 6.2 Confusion Matrix Validation
(hasil_confusion_matrix.png)

---

## 7. Evaluasi pada Data Uji Eksternal (LabelledRice)

Dataset `LabelledRice` (3.355 gambar) tidak digunakan selama training/validasi.

| Classifier    | Val Acc | Val F1 | Ext Acc | Ext F1 | Gap Acc |
|---------------|---------|--------|---------|--------|---------|
| XGBoost       | 0.4329  | 0.4320 | 0.6855  | 0.6919 | -0.2526 |
| Random Forest | 0.4289  | 0.4260 | 0.6793  | 0.6841 | -0.2504 |
| SVM           | 0.4106  | 0.4060 | 0.5997  | 0.6053 | -0.1891 |
| MLP           | 0.3841  | 0.3802 | 0.4349  | 0.4119 | -0.0507 |

**Catatan**: Gap negatif menunjukkan performa pada data eksternal lebih tinggi dari validasi, kemungkinan disebabkan oleh perbedaan distribusi kelas antara validation set (seimbang, 123/kelas) dan LabelledRice (tidak seimbang, didominasi kelas Healthy sebanyak 1488 gambar dari 3355).

### 7.1 Confusion Matrix External Test
(ext_confusion_matrix.png)

### 7.2 Perbandingan Validation vs External Test
(ext_perbandingan_val_vs_ext.png)

**Model terbaik**: XGBoost dengan Ext F1 = 0.6919.

---

## 8. Feature Importance (XGBoost)

Top-20 fitur dari 960 dimensi yang paling berkontribusi ditampilkan dalam diagram batang horizontal:
(xgboost_feature_importance.png)

---

## 9. Cara Menjalankan Kode

1. Clone repositori:  
   `git clone https://github.com/sandi-ardi/Machine-Learning.git`
2. Install dependensi:  
   `pip install -r requirements.txt`  
   (atau manual: tensorflow, sklearn, xgboost, numpy, pandas, matplotlib, seaborn, pillow, tqdm)
3. Sesuaikan `BASE_DIR` di kode dengan path dataset lokal kamu.
4. Buka notebook `Proyek_ML_Final.ipynb` dan eksekusi semua cell secara berurutan.

### 9.1 Klasifikasi Satu Gambar Baru
```python
res = classify_image("path/to/your/rice_leaf.jpg", model_name="XGBoost")
print(res["pred_label"])
