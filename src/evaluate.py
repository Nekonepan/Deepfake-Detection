"""
Modul Evaluasi dan Metrik untuk Model Deteksi Deepfake.

Modul ini menyediakan fungsi-fungsi untuk mengevaluasi performa model CNN
secara komprehensif, termasuk perhitungan metrik klasifikasi, pembuatan
confusion matrix, kurva ROC, dan visualisasi riwayat pelatihan.

Metrik yang dihitung:
    - Accuracy (Akurasi)
    - Precision (Presisi)
    - Recall (Sensitivitas)
    - F1-Score
    - AUC-ROC (Area Under ROC Curve)

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Konfigurasi logger untuk modul evaluasi
logger = logging.getLogger(__name__)

# Nama kelas untuk label
CLASS_NAMES: List[str] = ['Real', 'Fake']


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengumpulkan semua prediksi dan label sebenarnya dari DataLoader.

    Fungsi ini menjalankan inferensi pada seluruh dataset dan mengumpulkan
    prediksi model beserta probabilitas dan label sebenarnya.

    Args:
        model: Model CNN yang telah dilatih.
        data_loader: DataLoader yang berisi data untuk diprediksi.
        device: Perangkat komputasi (CUDA/MPS/CPU).

    Returns:
        Tuple berisi tiga numpy array:
            - all_labels: Array label sebenarnya (ground truth).
            - all_preds: Array label prediksi model.
            - all_probs: Array probabilitas prediksi untuk kelas positif
                         (fake) yang digunakan untuk kurva ROC.
    """
    # Set model ke mode evaluasi
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    for images, labels in data_loader:
        # Pindahkan data ke perangkat komputasi
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass untuk mendapatkan prediksi
        outputs = model(images)

        # Hitung probabilitas menggunakan softmax
        probabilities = F.softmax(outputs, dim=1)

        # Ambil prediksi kelas (argmax)
        _, predicted = torch.max(outputs, 1)

        # Kumpulkan hasil ke list
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        # Ambil probabilitas kelas fake (indeks 1) untuk kurva ROC
        all_probs.extend(probabilities[:, 1].cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Melakukan evaluasi komprehensif terhadap model pada data pengujian.

    Fungsi ini menghitung berbagai metrik klasifikasi dan menghasilkan
    visualisasi untuk analisis performa model.

    Metrik yang dihitung:
        - Accuracy: Proporsi prediksi benar dari total prediksi
        - Precision: Proporsi prediksi positif yang benar
        - Recall: Proporsi sampel positif yang berhasil terdeteksi
        - F1-Score: Rata-rata harmonis precision dan recall
        - AUC-ROC: Area di bawah kurva ROC

    Args:
        model: Model CNN yang telah dilatih.
        test_loader: DataLoader untuk data pengujian.
        device: Perangkat komputasi (CUDA/MPS/CPU).
        save_dir: Direktori untuk menyimpan plot (opsional).
                  Jika None, plot tidak disimpan.

    Returns:
        Dictionary berisi semua metrik evaluasi:
            - 'accuracy': Nilai akurasi (0-1)
            - 'precision': Nilai presisi (0-1)
            - 'recall': Nilai recall (0-1)
            - 'f1_score': Nilai F1-score (0-1)
            - 'auc_score': Nilai AUC-ROC (0-1)
            - 'confusion_matrix': Confusion matrix (numpy array 2x2)
            - 'classification_report': Laporan klasifikasi (string)
            - 'fpr': False Positive Rate untuk kurva ROC
            - 'tpr': True Positive Rate untuk kurva ROC
    """
    logger.info("Memulai evaluasi model pada data pengujian...")

    # Dapatkan semua prediksi
    all_labels, all_preds, all_probs = get_predictions(
        model, test_loader, device
    )

    # --- Hitung Metrik Klasifikasi ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average='binary', zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds, average='binary', zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)

    # --- Classification Report ---
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )

    # --- ROC Curve dan AUC ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    # Cetak hasil evaluasi
    print(
        f"\n{'='*60}"
        f"\n HASIL EVALUASI MODEL"
        f"\n{'='*60}"
        f"\n  Accuracy  : {accuracy*100:.2f}%"
        f"\n  Precision : {precision*100:.2f}%"
        f"\n  Recall    : {recall*100:.2f}%"
        f"\n  F1-Score  : {f1*100:.2f}%"
        f"\n  AUC-ROC   : {auc_score:.4f}"
        f"\n{'='*60}"
        f"\n\nLaporan Klasifikasi:\n{report}"
        f"\nConfusion Matrix:\n{cm}"
        f"\n{'='*60}\n"
    )

    # --- Simpan Visualisasi ---
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Plot confusion matrix
        plot_confusion_matrix(
            cm, save_path=str(save_path / 'confusion_matrix.png')
        )

        # Plot kurva ROC
        plot_roc_curve(
            fpr, tpr, auc_score,
            save_path=str(save_path / 'roc_curve.png')
        )

        logger.info(f"Visualisasi evaluasi disimpan di: {save_path}")

    # Susun hasil evaluasi ke dalam dictionary
    results: Dict[str, Any] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'confusion_matrix': cm,
        'classification_report': report,
        'fpr': fpr,
        'tpr': tpr,
    }

    logger.info(
        f"Evaluasi selesai. Accuracy: {accuracy*100:.2f}%, "
        f"AUC: {auc_score:.4f}"
    )

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Memvisualisasikan confusion matrix menggunakan heatmap seaborn.

    Confusion matrix menampilkan distribusi prediksi model dalam
    format tabel 2x2:
        - True Positive (TP): Fake terdeteksi sebagai Fake
        - True Negative (TN): Real terdeteksi sebagai Real
        - False Positive (FP): Real salah terdeteksi sebagai Fake
        - False Negative (FN): Fake salah terdeteksi sebagai Real

    Args:
        cm: Confusion matrix berupa numpy array 2x2.
        save_path: Path untuk menyimpan gambar (opsional).
                   Jika None, plot hanya ditampilkan.
    """
    plt.figure(figsize=(8, 6))

    # Buat heatmap dengan anotasi angka
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,
        linewidths=1,
        linecolor='black',
        annot_kws={'size': 16, 'weight': 'bold'}
    )

    plt.title(
        'Confusion Matrix\nSistem Deteksi Deepfake',
        fontsize=14, fontweight='bold', pad=15
    )
    plt.xlabel('Label Prediksi', fontsize=12, labelpad=10)
    plt.ylabel('Label Sebenarnya', fontsize=12, labelpad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix disimpan: {save_path}")

    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None
) -> None:
    """
    Memvisualisasikan kurva ROC (Receiver Operating Characteristic).

    Kurva ROC menunjukkan trade-off antara True Positive Rate (TPR)
    dan False Positive Rate (FPR) pada berbagai threshold klasifikasi.
    AUC (Area Under Curve) mengukur kemampuan diskriminasi model.

    Interpretasi AUC:
        - AUC = 1.0: Klasifikasi sempurna
        - AUC = 0.5: Klasifikasi acak (tidak lebih baik dari menebak)
        - AUC > 0.9: Klasifikasi sangat baik

    Args:
        fpr: Array False Positive Rate untuk setiap threshold.
        tpr: Array True Positive Rate untuk setiap threshold.
        auc_score: Nilai AUC (Area Under Curve).
        save_path: Path untuk menyimpan gambar (opsional).
    """
    plt.figure(figsize=(8, 6))

    # Plot kurva ROC
    plt.plot(
        fpr, tpr,
        color='#2196F3',
        linewidth=2.5,
        label=f'Model CNN (AUC = {auc_score:.4f})'
    )

    # Plot garis diagonal (klasifikasi acak)
    plt.plot(
        [0, 1], [0, 1],
        color='gray',
        linestyle='--',
        linewidth=1,
        label='Klasifikasi Acak (AUC = 0.5)'
    )

    # Area di bawah kurva ROC (shading)
    plt.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')

    plt.title(
        'Kurva ROC (Receiver Operating Characteristic)\n'
        'Sistem Deteksi Deepfake',
        fontsize=14, fontweight='bold', pad=15
    )
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12, labelpad=10)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Kurva ROC disimpan: {save_path}")

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Memvisualisasikan riwayat pelatihan model (loss dan akurasi).

    Menghasilkan dua subplot:
        1. Grafik loss training vs validasi per epoch
        2. Grafik akurasi training vs validasi per epoch

    Grafik ini berguna untuk mengidentifikasi:
        - Overfitting: gap antara training dan validasi melebar
        - Underfitting: kedua kurva masih tinggi/rendah
        - Titik konvergensi: kurva mulai mendatar

    Args:
        history: Dictionary berisi riwayat pelatihan dengan key:
                 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Path untuk menyimpan gambar (opsional).
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Grafik Loss ---
    ax1.plot(
        epochs, history['train_loss'],
        'b-', linewidth=2, label='Training Loss',
        marker='o', markersize=3
    )
    ax1.plot(
        epochs, history['val_loss'],
        'r-', linewidth=2, label='Validation Loss',
        marker='s', markersize=3
    )
    ax1.set_title(
        'Grafik Loss per Epoch',
        fontsize=13, fontweight='bold', pad=10
    )
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Grafik Akurasi ---
    ax2.plot(
        epochs, history['train_acc'],
        'b-', linewidth=2, label='Training Accuracy',
        marker='o', markersize=3
    )
    ax2.plot(
        epochs, history['val_acc'],
        'r-', linewidth=2, label='Validation Accuracy',
        marker='s', markersize=3
    )
    ax2.set_title(
        'Grafik Akurasi per Epoch',
        fontsize=13, fontweight='bold', pad=10
    )
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Akurasi (%)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Judul keseluruhan
    fig.suptitle(
        'Riwayat Pelatihan - Sistem Deteksi Deepfake',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Grafik riwayat pelatihan disimpan: {save_path}")

    plt.close()
