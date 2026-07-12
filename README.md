<p align="center">
  <h1 align="center">🔍 Sistem Deteksi Deepfake pada Citra Digital</h1>
  <h3 align="center">Menggunakan Metode Convolutional Neural Network (CNN)</h3>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/Bahasa-Indonesia-red?style=flat-square" alt="Bahasa">
</p>

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#-deskripsi-proyek)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Fitur Utama](#-fitur-utama)
- [Struktur Direktori](#-struktur-direktori)
- [Prasyarat & Instalasi](#-prasyarat--instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Arsitektur Model CNN](#-arsitektur-model-cnn)
- [Hasil Eksperimen](#-hasil-eksperimen)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Struktur Dataset](#-struktur-dataset)
- [Lisensi](#-lisensi)
- [Kontributor](#-kontributor)

---

## 📖 Deskripsi Proyek

Proyek ini mengimplementasikan **sistem deteksi deepfake** pada citra digital menggunakan metode **Convolutional Neural Network (CNN)**. Sistem ini dirancang untuk mengklasifikasikan gambar wajah menjadi dua kategori: **real (asli)** dan **fake (palsu/deepfake)**.

Deepfake adalah teknologi berbasis kecerdasan buatan yang mampu membuat konten visual palsu yang sangat realistis. Dengan semakin meningkatnya penyalahgunaan teknologi deepfake, diperlukan sistem deteksi yang akurat dan efisien untuk mengidentifikasi konten manipulasi.

### 🎯 Tujuan Penelitian

1. **Membangun** arsitektur CNN yang optimal untuk deteksi deepfake pada citra digital
2. **Menganalisis** performa model dalam mengklasifikasikan gambar real dan fake
3. **Mengimplementasikan** sistem deteksi dalam bentuk aplikasi web yang interaktif
4. **Mengevaluasi** akurasi, presisi, recall, dan F1-score dari model yang dibangun

---

## 🏗️ Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────────┐
│                    SISTEM DETEKSI DEEPFAKE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│   │  Input   │    │  Preprocessing │    │    CNN Model          │  │
│   │  Gambar  │───▶│  & Augmentasi │───▶│                       │  │
│   │ (224x224)│    │              │    │  Conv2D → ReLU → Pool │  │
│   └─────────┘    └──────────────┘    │  Conv2D → ReLU → Pool │  │
│                                       │  Conv2D → ReLU → Pool │  │
│                                       │  Conv2D → ReLU → Pool │  │
│                                       │         │             │  │
│                                       │     Flatten           │  │
│                                       │         │             │  │
│                                       │  FC → ReLU → Dropout  │  │
│                                       │  FC → ReLU → Dropout  │  │
│                                       │  FC → Softmax         │  │
│                                       └───────────┬───────────┘  │
│                                                   │              │
│                                       ┌───────────▼───────────┐  │
│                                       │      Output           │  │
│                                       │  Real (0) / Fake (1)  │  │
│                                       │  + Confidence Score   │  │
│                                       └───────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                  Streamlit Web App                        │  │
│   │  Upload Gambar → Prediksi → Tampilkan Hasil + Visualisasi│  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🧠 **Model CNN Custom** | Arsitektur CNN yang dirancang khusus untuk deteksi deepfake |
| 📊 **Evaluasi Komprehensif** | Confusion matrix, classification report, ROC curve, dan metrik lainnya |
| 🌐 **Aplikasi Web** | Antarmuka web interaktif menggunakan Streamlit |
| 📈 **Visualisasi** | Grafik pelatihan, distribusi prediksi, dan analisis kesalahan |
| 🔄 **Data Augmentasi** | Augmentasi otomatis untuk meningkatkan generalisasi model |
| 💾 **Early Stopping** | Penghentian otomatis saat model tidak lagi membaik |
| ⚙️ **Konfigurasi YAML** | Pengaturan fleksibel melalui file konfigurasi |
| 📁 **Pipeline Terstruktur** | Alur kerja terorganisir dari persiapan data hingga deployment |

---

## 📂 Struktur Direktori

```
Deepfake-Detection/
├── 📄 README.md                    # Dokumentasi proyek
├── 📄 requirements.txt             # Dependensi Python
├── 📄 setup.py                     # Konfigurasi paket
├── 📄 .gitignore                   # File yang diabaikan Git
│
├── 📁 configs/                     # File konfigurasi
│   └── default_config.yaml         # Konfigurasi default
│
├── 📁 src/                         # Kode sumber utama
│   └── deepfake_detection/
│       ├── __init__.py
│       ├── model.py                # Arsitektur CNN
│       ├── dataset.py              # Data loading & preprocessing
│       ├── train.py                # Script pelatihan
│       ├── evaluate.py             # Script evaluasi
│       ├── predict.py              # Script prediksi
│       ├── transforms.py           # Transformasi & augmentasi data
│       └── utils.py                # Fungsi utilitas
│
├── 📁 app/                         # Aplikasi web Streamlit
│   └── streamlit_app.py            # Antarmuka web
│
├── 📁 scripts/                     # Script utilitas
│   ├── prepare_dataset.py          # Persiapan dataset
│   └── download_sample_data.py     # Download data sampel
│
├── 📁 data/                        # Dataset
│   ├── train/                      # Data pelatihan
│   │   ├── real/
│   │   └── fake/
│   ├── val/                        # Data validasi
│   │   ├── real/
│   │   └── fake/
│   └── test/                       # Data pengujian
│       ├── real/
│       └── fake/
│
├── 📁 models/                      # Model tersimpan
│   └── best_model.pth              # Model terbaik
│
├── 📁 results/                     # Hasil evaluasi
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── roc_curve.png
│   └── classification_report.txt
│
└── 📁 logs/                        # Log pelatihan
    └── training_log.txt
```

---

## 🔧 Prasyarat & Instalasi

### Persyaratan Sistem

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| **Python** | 3.8 | 3.10+ |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | - | NVIDIA dengan CUDA 11.7+ |
| **Disk** | 5 GB | 20 GB+ (termasuk dataset) |
| **OS** | Windows 10 / Ubuntu 18.04 | Ubuntu 20.04+ |

### Langkah Instalasi

#### 1️⃣ Clone Repository

```bash
git clone https://github.com/Nekonepan/Deepfake-Detection.git
cd Deepfake-Detection
```

#### 2️⃣ Buat Virtual Environment

```bash
# Menggunakan venv
python -m venv venv

# Aktivasi (Linux/MacOS)
source venv/bin/activate

# Aktivasi (Windows)
venv\Scripts\activate
```

#### 3️⃣ Instal Dependensi

```bash
# Instal semua dependensi
pip install -r requirements.txt

# Atau instal sebagai paket (mode development)
pip install -e .
```

#### 4️⃣ Verifikasi Instalasi

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Persiapan Dataset

#### Menggunakan Data Sampel (untuk pengujian)

```bash
# Buat gambar placeholder untuk menguji pipeline
python scripts/download_sample_data.py

# Persiapkan struktur dataset dari data sampel
python scripts/prepare_dataset.py --source data/sample --output data --copy
```

#### Menggunakan Dataset Asli

1. **Unduh** dataset dari sumber yang tersedia (lihat `scripts/download_sample_data.py --info-only`)
2. **Organisir** gambar ke dalam folder `data/raw/real/` dan `data/raw/fake/`
3. **Jalankan** script persiapan dataset:

```bash
python scripts/prepare_dataset.py \
    --source data/raw \
    --output data \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

---

## 🚀 Cara Penggunaan

### 1. Training Model

```bash
# Pelatihan dengan konfigurasi default
python -m deepfake_detection.train

# Pelatihan dengan konfigurasi custom
python -m deepfake_detection.train --config configs/default_config.yaml

# Pelatihan dengan parameter tertentu
python -m deepfake_detection.train \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0001
```

### 2. Evaluasi Model

```bash
# Evaluasi model pada data test
python -m deepfake_detection.evaluate \
    --model models/best_model.pth \
    --data data/test \
    --output results/
```

### 3. Prediksi Gambar

```bash
# Prediksi satu gambar
python -m deepfake_detection.predict \
    --model models/best_model.pth \
    --image path/ke/gambar.jpg

# Prediksi banyak gambar (batch)
python -m deepfake_detection.predict \
    --model models/best_model.pth \
    --input-dir path/ke/direktori_gambar/ \
    --output-dir results/predictions/
```

### 4. Menjalankan Aplikasi Web

```bash
# Jalankan aplikasi Streamlit
streamlit run app/streamlit_app.py

# Dengan port custom
streamlit run app/streamlit_app.py --server.port 8080
```

Aplikasi web akan terbuka di browser pada `http://localhost:8501`.

---

## 🧬 Arsitektur Model CNN

### Gambaran Umum

Model CNN yang digunakan terdiri dari **4 blok konvolusi** yang diikuti oleh **3 fully connected layers**. Setiap blok konvolusi terdiri dari lapisan Conv2D, Batch Normalization, fungsi aktivasi ReLU, dan Max Pooling.

### Detail Layer-by-Layer

```
╔══════════════════════════════════════════════════════════════════╗
║                    ARSITEKTUR DeepfakeCNN                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Input: RGB Image (3 × 224 × 224)                               ║
║                                                                  ║
║  ┌─── Blok Konvolusi 1 ──────────────────────────────────────┐  ║
║  │  Conv2D(3→32, kernel=3×3, padding=1)    → 32 × 224 × 224 │  ║
║  │  BatchNorm2D(32)                                          │  ║
║  │  ReLU()                                                   │  ║
║  │  MaxPool2D(kernel=2×2, stride=2)        → 32 × 112 × 112 │  ║
║  └───────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─── Blok Konvolusi 2 ──────────────────────────────────────┐  ║
║  │  Conv2D(32→64, kernel=3×3, padding=1)   → 64 × 112 × 112 │  ║
║  │  BatchNorm2D(64)                                          │  ║
║  │  ReLU()                                                   │  ║
║  │  MaxPool2D(kernel=2×2, stride=2)        → 64 × 56 × 56   │  ║
║  └───────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─── Blok Konvolusi 3 ──────────────────────────────────────┐  ║
║  │  Conv2D(64→128, kernel=3×3, padding=1)  → 128 × 56 × 56  │  ║
║  │  BatchNorm2D(128)                                         │  ║
║  │  ReLU()                                                   │  ║
║  │  MaxPool2D(kernel=2×2, stride=2)        → 128 × 28 × 28  │  ║
║  └───────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─── Blok Konvolusi 4 ──────────────────────────────────────┐  ║
║  │  Conv2D(128→256, kernel=3×3, padding=1) → 256 × 28 × 28  │  ║
║  │  BatchNorm2D(256)                                         │  ║
║  │  ReLU()                                                   │  ║
║  │  MaxPool2D(kernel=2×2, stride=2)        → 256 × 14 × 14  │  ║
║  └───────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  Flatten()                                 → 256 × 14 × 14     ║
║                                              = 50.176           ║
║                                                                  ║
║  ┌─── Fully Connected Layers ────────────────────────────────┐  ║
║  │  Linear(50176 → 512)                                      │  ║
║  │  ReLU()                                                   │  ║
║  │  Dropout(0.5)                                             │  ║
║  │                                                           │  ║
║  │  Linear(512 → 128)                                        │  ║
║  │  ReLU()                                                   │  ║
║  │  Dropout(0.5)                                             │  ║
║  │                                                           │  ║
║  │  Linear(128 → 2)                                          │  ║
║  └───────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  Output: [prob_real, prob_fake]  (Softmax)                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Ringkasan Parameter

| Layer | Input | Output | Parameter |
|-------|-------|--------|-----------|
| Conv2D-1 | 3×224×224 | 32×224×224 | 896 |
| Conv2D-2 | 32×112×112 | 64×112×112 | 18.496 |
| Conv2D-3 | 64×56×56 | 128×56×56 | 73.856 |
| Conv2D-4 | 128×28×28 | 256×28×28 | 295.168 |
| FC-1 | 50.176 | 512 | 25.690.624 |
| FC-2 | 512 | 128 | 65.664 |
| FC-3 | 128 | 2 | 258 |
| **Total** | | | **≈ 26.1M** |

### Teknik Regularisasi

- **Batch Normalization**: Normalisasi output setiap blok konvolusi untuk stabilisasi pelatihan
- **Dropout (0.5)**: Mencegah overfitting dengan menonaktifkan 50% neuron secara acak
- **Weight Decay (L2)**: Regularisasi L2 pada optimizer untuk mencegah bobot yang terlalu besar
- **Early Stopping**: Menghentikan pelatihan jika tidak ada perbaikan selama `patience` epoch

### Fungsi Loss dan Optimizer

- **Loss Function**: `CrossEntropyLoss` — cocok untuk klasifikasi biner
- **Optimizer**: `Adam` — adaptive learning rate dengan momentum
- **Scheduler**: `ReduceLROnPlateau` — mengurangi learning rate saat validasi loss stagnan

---

## 📊 Hasil Eksperimen

> ⚠️ **Catatan**: Bagian ini akan diperbarui setelah eksperimen dilakukan.

### Metrik Evaluasi

| Metrik | Nilai |
|--------|-------|
| **Akurasi** | _Akan diisi_ |
| **Presisi** | _Akan diisi_ |
| **Recall** | _Akan diisi_ |
| **F1-Score** | _Akan diisi_ |
| **AUC-ROC** | _Akan diisi_ |

### Grafik Pelatihan

> _Training loss vs validation loss dan akurasi per epoch akan ditampilkan di sini._

### Confusion Matrix

> _Confusion matrix dari evaluasi pada data test akan ditampilkan di sini._

### ROC Curve

> _Receiver Operating Characteristic curve akan ditampilkan di sini._

---

## 🛠️ Teknologi yang Digunakan

<table>
  <tr>
    <th>Kategori</th>
    <th>Teknologi</th>
    <th>Versi</th>
    <th>Kegunaan</th>
  </tr>
  <tr>
    <td rowspan="2"><b>Deep Learning</b></td>
    <td>PyTorch</td>
    <td>≥ 2.0.0</td>
    <td>Framework deep learning utama</td>
  </tr>
  <tr>
    <td>TorchVision</td>
    <td>≥ 0.15.0</td>
    <td>Transformasi gambar & model pre-trained</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Data Science</b></td>
    <td>NumPy</td>
    <td>≥ 1.24.0</td>
    <td>Komputasi numerik</td>
  </tr>
  <tr>
    <td>Pandas</td>
    <td>≥ 2.0.0</td>
    <td>Manipulasi data tabular</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Visualisasi</b></td>
    <td>Matplotlib</td>
    <td>≥ 3.7.0</td>
    <td>Grafik dan plot</td>
  </tr>
  <tr>
    <td>Seaborn</td>
    <td>≥ 0.12.0</td>
    <td>Visualisasi statistik</td>
  </tr>
  <tr>
    <td><b>Computer Vision</b></td>
    <td>OpenCV</td>
    <td>≥ 4.7.0</td>
    <td>Pemrosesan gambar</td>
  </tr>
  <tr>
    <td><b>Machine Learning</b></td>
    <td>Scikit-learn</td>
    <td>≥ 1.2.0</td>
    <td>Metrik evaluasi & utilitas ML</td>
  </tr>
  <tr>
    <td><b>Web App</b></td>
    <td>Streamlit</td>
    <td>≥ 1.28.0</td>
    <td>Antarmuka web interaktif</td>
  </tr>
  <tr>
    <td><b>Image Processing</b></td>
    <td>Pillow</td>
    <td>≥ 9.5.0</td>
    <td>Manipulasi gambar</td>
  </tr>
  <tr>
    <td><b>Utilities</b></td>
    <td>tqdm</td>
    <td>≥ 4.65.0</td>
    <td>Progress bar</td>
  </tr>
</table>

---

## 📁 Struktur Dataset

Dataset harus diorganisir dalam struktur berikut:

```
data/
├── train/                  # 70% dari total data
│   ├── real/              # Gambar wajah asli
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── fake/              # Gambar deepfake
│       ├── img_001.jpg
│       ├── img_002.jpg
│       └── ...
│
├── val/                    # 15% dari total data
│   ├── real/
│   └── fake/
│
└── test/                   # 15% dari total data
    ├── real/
    └── fake/
```

### Sumber Dataset yang Direkomendasikan

| Dataset | Deskripsi | Ukuran |
|---------|-----------|--------|
| **FaceForensics++** | Dataset benchmark standar dengan 4 metode manipulasi | ~600 GB |
| **DFDC** | Dataset terbesar dari Meta/Facebook | ~470 GB |
| **Celeb-DF v2** | Deepfake berkualitas tinggi dari wajah selebriti | ~15 GB |
| **WildDeepfake** | Deepfake dari internet (in-the-wild) | ~4 GB |

> Lihat `python scripts/download_sample_data.py --info-only` untuk informasi lengkap.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License** — lihat file [LICENSE](LICENSE) untuk detail.

```
MIT License

Copyright (c) 2024 Nekonepan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👥 Kontributor

<table>
  <tr>
    <td align="center">
      <b>Nekonepan</b><br>
      <sub>Pengembang Utama</sub><br>
      <a href="https://github.com/Nekonepan">GitHub</a>
    </td>
  </tr>
</table>

---

<p align="center">
  <b>⭐ Jika proyek ini bermanfaat, berikan bintang pada repository ini! ⭐</b>
</p>

<p align="center">
  Dibuat dengan ❤️ untuk penelitian akademis
</p>