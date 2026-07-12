"""
Modul Utilitas untuk Deteksi Deepfake.

Modul ini menyediakan fungsi utilitas dan kelas pembantu (helper) untuk
reprodusibilitas, pemilihan perangkat keras (device selection), perhitungan parameter,
pembuatan direktori, konfigurasi logging, serta mekanisme Early Stopping.

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Mengatur seed untuk reprodusibilitas hasil eksperimen.
    Menjamin hasil pelatihan yang sama jika dijalankan ulang dengan seed yang sama.

    Args:
        seed: Nilai integer untuk seed acak (default: 42).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Untuk konfigurasi multi-GPU
    
    # Pengaturan agar penambahan CUDA bersifat deterministik (meski dapat mengurangi kecepatan)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Seed untuk reprodusibilitas diatur ke: {seed}")


def get_device() -> torch.device:
    """
    Mendeteksi secara otomatis perangkat keras terbaik yang tersedia (CUDA, MPS, atau CPU).

    Returns:
        torch.device: Perangkat keras yang terdeteksi.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Menggunakan GPU (CUDA): {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Dukungan akselerasi GPU Metal untuk macOS Apple Silicon
        device = torch.device("mps")
        logging.info("Menggunakan GPU Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        logging.info("Menggunakan CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """
    Menghitung jumlah parameter yang dapat dilatih (trainable parameters) pada model.

    Args:
        model: Model PyTorch yang akan dihitung parameternya.

    Returns:
        int: Jumlah parameter terlatih.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dirs(dirs: List[Union[str, Path]]) -> None:
    """
    Membuat daftar direktori jika belum ada di sistem file.

    Args:
        dirs: Daftar path direktori yang akan dibuat.
    """
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Direktori berhasil dibuat: {path}")


def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO) -> None:
    """
    Mengonfigurasi logger sistem untuk mencatat progres ke konsol dan file log.

    Args:
        log_dir: Direktori tempat menyimpan file log.
        log_level: Tingkat pencatatan log (default: logging.INFO).
    """
    create_dirs([log_dir])
    log_file = Path(log_dir) / 'training.log'
    
    # Konfigurasi format pencatatan log
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s]  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Hapus handler yang ada untuk menghindari duplikasi log
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Handler 1: Menyimpan log ke file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Handler 2: Menampilkan log ke konsol/terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Sistem logging diaktifkan. Log disimpan di: {log_file}")


class EarlyStopping:
    """
    Mekanisme Early Stopping untuk menghentikan pelatihan model jika performa
    validasi (loss atau akurasi) tidak membaik setelah sejumlah epoch (patience).

    Metode ini mencegah overfitting dan menghemat sumber daya komputasi.
    """

    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.0,
        mode: str = 'min',
        trace_func: Any = print
    ) -> None:
        """
        Inisialisasi EarlyStopping.

        Args:
            patience: Jumlah epoch yang ditoleransi tanpa adanya perbaikan sebelum berhenti (default: 10).
            verbose: Jika True, mencetak informasi perubahan performa (default: True).
            delta: Perubahan minimum pada kuantitas yang dipantau agar dianggap sebagai perbaikan (default: 0.0).
            mode: Mode pemantauan ('min' untuk loss, 'max' untuk akurasi) (default: 'min').
            trace_func: Fungsi untuk mencetak pesan trace (default: print).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.delta = delta
        self.mode = mode
        self.trace_func = trace_func

        if mode not in ('min', 'max'):
            raise ValueError("mode harus berupa 'min' atau 'max'")

    def __call__(self, val_metric: float) -> None:
        """
        Mengevaluasi nilai metrik validasi saat ini untuk mendeteksi apakah pelatihan harus dihentikan.

        Args:
            val_metric: Nilai loss atau akurasi validasi pada epoch saat ini.
        """
        # Ubah skor agar "lebih tinggi selalu lebih baik"
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_value = val_metric
        elif score < self.best_score + self.delta:
            # Performa menurun atau tidak ada peningkatan yang signifikan
            self.counter += 1
            if self.verbose:
                metric_name = "Loss" if self.mode == 'min' else "Akurasi"
                val_str = f"{self.best_value:.6f}" if self.mode == 'min' else f"{self.best_value:.2f}%"
                self.trace_func(
                    f"EarlyStopping Counter: {self.counter} dari {self.patience} "
                    f"(Best Val {metric_name}: {val_str})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Performa membaik
            self.best_score = score
            self.best_value = val_metric
            self.counter = 0
            if self.verbose:
                metric_name = "Loss" if self.mode == 'min' else "Akurasi"
                val_str = f"{val_metric:.6f}" if self.mode == 'min' else f"{val_metric:.2f}%"
                self.trace_func(
                    f"Validation {metric_name} membaik menjadi {val_str}. Counter di-reset."
                )

