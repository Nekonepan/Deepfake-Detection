"""
Modul Visualisasi untuk Deteksi Deepfake.

Modul ini menyediakan fungsi untuk memvisualisasikan data, hasil pelatihan,
serta cara kerja internal Convolutional Neural Network (CNN) seperti visualisasi
feature map dan visualisasi bobot filter konvolusi.

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from .dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


def visualize_feature_maps(
    model: nn.Module,
    image: Union[Image.Image, torch.Tensor],
    layer_name: str = 'conv_block1',
    save_path: Optional[str] = None,
    max_features: int = 16
) -> None:
    """
    Memvisualisasikan feature map (aktivasi) dari lapisan konvolusi tertentu
    setelah melewatkan gambar input melalui model.

    Args:
        model: Model CNN (DeepfakeCNN) yang digunakan.
        image: Gambar input berupa PIL Image atau Tensor PyTorch.
        layer_name: Nama lapisan konvolusi yang ingin divisualisasikan
                    ('conv_block1', 'conv_block2', 'conv_block3', 'conv_block4', 'adaptive_pool').
        save_path: Path untuk menyimpan gambar visualisasi (opsional).
        max_features: Jumlah maksimum feature map yang ditampilkan (default: 16).
    """
    model.eval()
    
    # 1. Konversi gambar ke tensor jika input berupa PIL Image
    if isinstance(image, Image.Image):
        # Gunakan pipeline transformasi evaluasi (resize + normalize)
        transform = get_transforms('val')
        input_tensor = transform(image).unsqueeze(0)  # Tambah dimensi batch (1, C, H, W)
    else:
        input_tensor = image.clone()
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
    # Deteksi device model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 2. Forward pass untuk mengaktifkan pemuatan feature map
    with torch.no_grad():
        _ = model(input_tensor)
        
    # 3. Dapatkan feature map dari model
    try:
        if hasattr(model, 'get_feature_maps'):
            feature_maps = model.get_feature_maps()
        else:
            raise AttributeError("Model tidak memiliki metode 'get_feature_maps'")
            
        if layer_name not in feature_maps:
            raise KeyError(
                f"Lapisan '{layer_name}' tidak ditemukan. "
                f"Pilihan yang tersedia: {list(feature_maps.keys())}"
            )
            
        # Ambil tensor feature map dan hapus dimensi batch
        f_map = feature_maps[layer_name][0].cpu().numpy()  # Dimensi: (C, H, W)
    except Exception as e:
        logger.error(f"Gagal mendapatkan feature map: {e}")
        return
        
    # 4. Gambar visualisasi
    num_channels = f_map.shape[0]
    display_count = min(num_channels, max_features)
    
    # Hitung jumlah baris dan kolom untuk grid plot
    cols = 4
    rows = (display_count + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle(
        f"Feature Map: {layer_name}\n"
        f"({display_count} Saluran Pertama dari Total {num_channels} Saluran)",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # Ratakan array axes agar mudah diiterasi
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.ravel()
        
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < display_count:
            # Dapatkan feature map dari saluran ke-i
            channel_map = f_map[i]
            # Normalisasi nilai piksel ke rentang 0-1 untuk tampilan optimal
            if channel_map.max() - channel_map.min() > 0:
                channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min())
            
            # Tampilkan sebagai heatmap dengan colormap 'viridis'
            im = ax.imshow(channel_map, cmap='viridis')
            ax.set_title(f"Saluran {i+1}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualisasi feature map disimpan ke: {save_path}")
        
    plt.close()


def visualize_filters(
    model: nn.Module,
    layer_name: str = 'conv_block1',
    save_path: Optional[str] = None,
    max_filters: int = 16
) -> None:
    """
    Memvisualisasikan bobot filter (kernel) dari lapisan konvolusi pertama
    yang dipelajari oleh model CNN.

    Args:
        model: Model CNN yang digunakan.
        layer_name: Nama modul blok konvolusi (misal: 'conv_block1').
        save_path: Path untuk menyimpan gambar visualisasi (opsional).
        max_filters: Jumlah maksimum filter yang ditampilkan (default: 16).
    """
    try:
        # Cari lapisan konvolusi di dalam blok
        block = getattr(model, layer_name, None)
        if block is None:
            raise AttributeError(f"Model tidak memiliki modul dengan nama '{layer_name}'")
            
        conv_layer = None
        for layer in block:
            if isinstance(layer, nn.Conv2d):
                conv_layer = layer
                break
                
        if conv_layer is None:
            raise ValueError(f"Tidak ditemukan lapisan Conv2d di dalam '{layer_name}'")
            
        # Ambil bobot filter
        weights = conv_layer.weight.detach().cpu().numpy()  # Dimensi: (out_channels, in_channels, H, W)
    except Exception as e:
        logger.error(f"Gagal memvisualisasikan filter: {e}")
        return
        
    num_filters = weights.shape[0]
    display_count = min(num_filters, max_filters)
    
    cols = 4
    rows = (display_count + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle(
        f"Bobot Filter Konvolusi pada: {layer_name}\n"
        f"({display_count} Filter Pertama dari Total {num_filters} Filter)",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.ravel()
        
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < display_count:
            # Ambil filter ke-i
            w = weights[i]  # Dimensi: (in_channels, H, W)
            
            # Jika 3 channel input (RGB), tampilkan sebagai citra berwarna
            if w.shape[0] == 3:
                # Transpose ke H x W x C
                w = np.transpose(w, (1, 2, 0))
                # Normalisasi ke 0-1
                w = (w - w.min()) / (w.max() - w.min())
                ax.imshow(w)
            # Jika 1 channel input atau channel lainnya, tampilkan sebagai grayscale
            else:
                # Ambil channel pertama saja
                w_gray = w[0]
                w_gray = (w_gray - w_gray.min()) / (w_gray.max() - w_gray.min())
                ax.imshow(w_gray, cmap='gray')
                
            ax.set_title(f"Filter {i+1}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualisasi filter konvolusi disimpan ke: {save_path}")
        
    plt.close()


def visualize_dataset_samples(
    dataset: torch.utils.data.Dataset,
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Menampilkan dan memvisualisasikan contoh citra digital dari dataset beserta labelnya.

    Args:
        dataset: Objek dataset (misal: DeepfakeDataset).
        num_samples: Jumlah sampel yang akan ditampilkan (default: 8).
        save_path: Path untuk menyimpan visualisasi (opsional).
    """
    total_samples = len(dataset)
    if total_samples == 0:
        logger.warning("Dataset kosong, tidak dapat memvisualisasikan sampel.")
        return
        
    display_count = min(total_samples, num_samples)
    
    # Pilih indeks acak dari dataset
    indices = np.random.choice(total_samples, display_count, replace=False)
    
    cols = 4
    rows = (display_count + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    
    # Dapatkan nama kelas dari dataset jika ada
    class_names = {0: 'Real (Asli)', 1: 'Fake (Palsu)'}
    if hasattr(dataset, 'class_names'):
        class_names = {v: k.capitalize() for k, v in dataset.class_names.items()}
        
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.ravel()
        
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < display_count:
            idx = indices[i]
            img_tensor, label = dataset[idx]
            
            # Kembalikan normalisasi (Denormalize) tensor untuk visualisasi
            # Image = Tensor * std + mean
            img_np = img_tensor.cpu().numpy().transpose((1, 2, 0))
            mean = np.array(IMAGENET_MEAN)
            std = np.array(IMAGENET_STD)
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            
            # Warnai judul: hijau untuk real, merah untuk fake
            label_name = class_names.get(label, str(label))
            title_color = 'green' if label == 0 else 'red'
            
            ax.set_title(f"Indeks: {idx}\nLabel: {label_name}", fontsize=10, color=title_color, fontweight='bold')
        ax.axis('off')
        
    fig.suptitle(
        "Sampel Citra Digital dari Dataset",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualisasi sampel dataset disimpan ke: {save_path}")
        
    plt.close()


def plot_class_distribution(
    dataset: torch.utils.data.Dataset,
    save_path: Optional[str] = None
) -> None:
    """
    Membuat grafik batang (bar plot) untuk menampilkan distribusi kelas (Real vs Fake)
    pada dataset yang dimuat.

    Args:
        dataset: Objek dataset (misal: DeepfakeDataset).
        save_path: Path untuk menyimpan gambar grafik (opsional).
    """
    # Dapatkan distribusi kelas
    if hasattr(dataset, 'get_class_distribution'):
        dist = dataset.get_class_distribution()
        classes = list(dist.keys())
        counts = list(dist.values())
    else:
        # Fallback jika tidak ada metode get_class_distribution
        labels = [dataset[i][1] for i in range(len(dataset))]
        classes = ['real', 'fake']
        counts = [labels.count(0), labels.count(1)]
        
    # Terjemahkan nama kelas ke Bahasa Indonesia untuk grafik
    class_labels = [c.capitalize() + ' (Asli)' if c == 'real' else c.capitalize() + ' (Palsu)' for c in classes]
    
    plt.figure(figsize=(7, 5))
    
    # Warna palet: Hijau untuk Real, Merah untuk Fake
    colors = ['#4CAF50', '#F44336']
    
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=class_labels, y=counts, palette=colors, hue=class_labels, legend=False)
    
    # Tambahkan angka di atas setiap batang grafik
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 8),
            textcoords='offset points',
            fontsize=11, fontweight='bold'
        )
        
    plt.title(
        'Distribusi Kelas pada Dataset',
        fontsize=13, fontweight='bold', pad=15
    )
    plt.ylabel('Jumlah Citra', fontsize=11, labelpad=10)
    plt.xlabel('Kelas Citra', fontsize=11, labelpad=10)
    plt.ylim([0, max(counts) * 1.15])  # Berikan ruang untuk anotasi angka
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grafik distribusi kelas disimpan ke: {save_path}")
        
    plt.close()
