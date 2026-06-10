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


---

## 4. Ekstraksi Fitur dengan MobileNetV3Large dan Normalisasi

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
| **MLP** | 3 hidden layer (1024, 512,256), ReLU, Adam, early stopping, StandardScaler |
| **Random Forest** | 100 trees, max_depth=None, n_jobs=-1 |
| **SVM** | Kernel RBF, C=1.0, gamma='scale', StandardScaler |
| **XGBoost** | 100 estimators, max_depth=4, lr=0.05, subsample=0.7, colsample=0.7 |

**Hasil training:**

| Classifier    | Train Acc | Val Acc | Waktu (s) |
|---------------|-----------|---------|-----------|
| XGBoost       | 0.8988    | 0.4126  | 7.6       |
| Random Forest | 0.9762    | 0.4126  | 0.4       |
| SVM           | 0.6138    | 0.4268  | 6.5       |
| MLP           | 0.8219    | 0.4472  | 11.1      |

---

## 6. Evaluasi pada Data Validasi

### 6.1 Classification Report per Model

**MLP**
| Kelas       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| BrownSpot   | 0.6034    | 0.5691 | 0.5858   | 123     |
| Healthy     | 0.3133    | 0.2114 | 0.2524   | 123     |
| Hispa       | 0.4188    | 0.5447 | 0.4735   | 123     |
| LeafBlast   | 0.4286    | 0.4634 | 0.4453   | 123     |
| **Macro avg** | 0.4410  | 0.4472 | 0.4393   | 492     |

**Random Forest**
| Kelas       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| BrownSpot   | 0.5036    | 0.5610 | 0.5308   |
| Healthy     | 0.3917    | 0.3821 | 0.3868   |
| Hispa       | 0.3481    | 0.3821 | 0.3643   |
| LeafBlast   | 0.4000    | 0.3252 | 0.3587   |
| **Macro avg** | 0.4109  | 0.4126 | 0.4102   |


**SVM**
| Kelas       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| BrownSpot   | 0.4926    | 0.5447 | 0.5174   |
| Healthy     | 0.3723    | 0.2846 | 0.3226   |
| Hispa       | 0.4014    | 0.4797 | 0.4370   |
| LeafBlast   | 0.4261    | 0.3984 | 0.4118   |
| **Macro avg** | 0.4231  | 0.4268 | 0.4222   |

**XGBoost**
| Kelas       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| BrownSpot   | 0.5175    | 0.6016 | 0.5564   |
| Healthy     | 0.3495    | 0.2927 | 0.3186   |
| Hispa       | 0.3529    | 0.4390 | 0.3913   |
| LeafBlast   | 0.4194    | 0.3171 | 0.3611   |
| **Macro avg** | 0.4098  | 0.4126 | 0.4068   |

### 6.2 Confusion Matrix Validation
(hasil_confusion_matrix.png)

---

## 7. Evaluasi pada Data Uji Eksternal (LabelledRice)

Dataset `LabelledRice` (3.355 gambar) tidak digunakan selama training/validasi.

| Classifier    | Val Acc | Val F1 | Ext Acc | Ext F1 | Gap Acc |
|-|-|-|-|-|-|---|---------|--------|---------|--------|---------|
| MLP           |  0.4472 | 0.4393 | 0.5800  | 0.5920 | -0.1329 |
| SVM           | 0.4268  | 0.4222 | 0.4808  | 0.4802 | -0.0539 |
| Random Forest | 0.4126  | 0.4102 | 0.6650  | 0.6677 | -0.2524 |
| XGBoost       | 0.4126  | 0.4068 | 0.6244  | 0.6272 | -0.2118 |

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
