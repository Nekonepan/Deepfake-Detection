#!/usr/bin/env python
"""
Script utama untuk melakukan prediksi deteksi deepfake pada sebuah gambar
serta menghasilkan visualisasi interpretasi Grad-CAM.

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)

Penggunaan:
    python run_predict.py --image_path path/to/image.jpg --model_path models/best_model.pth
"""

from src.predict import main

if __name__ == "__main__":
    main()
