#!/usr/bin/env python
"""
Script utama untuk menjalankan proses pelatihan model CNN deteksi deepfake.

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)

Penggunaan:
    python run_training.py --data_dir data/ --epochs 50 --batch_size 32
"""

from src.train import main

if __name__ == "__main__":
    main()
