"""
Modul Prediksi Citra Tunggal untuk Deteksi Deepfake.

Modul ini menyediakan kelas DeepfakePredictor untuk melakukan prediksi
pada citra tunggal atau batch citra, serta implementasi Grad-CAM untuk
visualisasi area perhatian model CNN.

Grad-CAM (Gradient-weighted Class Activation Mapping) menghasilkan
heatmap yang menunjukkan area pada citra yang paling berpengaruh
terhadap keputusan klasifikasi model.

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from .model import DeepfakeCNN
from .dataset import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

# Konfigurasi logger untuk modul prediksi
logger = logging.getLogger(__name__)

# Nama kelas untuk label prediksi
CLASS_NAMES: List[str] = ['Real', 'Fake']


class GradCAM:
    """
    Implementasi Grad-CAM untuk visualisasi area perhatian CNN.

    Grad-CAM (Gradient-weighted Class Activation Mapping) menggunakan
    gradien yang mengalir ke lapisan konvolusi terakhir untuk menghasilkan
    peta lokalisasi yang menyoroti region penting pada citra input
    untuk memprediksi suatu kelas.

    Referensi:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017.

    Atribut:
        model (nn.Module): Model CNN untuk analisis.
        target_layer (nn.Module): Lapisan konvolusi target untuk Grad-CAM.
        gradients (Optional[torch.Tensor]): Gradien yang ditangkap.
        activations (Optional[torch.Tensor]): Aktivasi yang ditangkap.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    ) -> None:
        """
        Inisialisasi Grad-CAM.

        Args:
            model: Model CNN yang akan dianalisis.
            target_layer: Lapisan konvolusi target untuk menghitung
                          Grad-CAM (biasanya lapisan konvolusi terakhir).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Daftarkan hook untuk menangkap gradien dan aktivasi
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Mendaftarkan forward dan backward hook pada lapisan target.

        Forward hook menangkap aktivasi (output) dari lapisan target,
        sedangkan backward hook menangkap gradien yang mengalir
        melalui lapisan tersebut.
        """
        def forward_hook(
            module: nn.Module,
            input: Tuple[torch.Tensor, ...],
            output: torch.Tensor
        ) -> None:
            """Menangkap aktivasi dari forward pass."""
            self.activations = output.detach()

        def backward_hook(
            module: nn.Module,
            grad_input: Tuple[torch.Tensor, ...],
            grad_output: Tuple[torch.Tensor, ...]
        ) -> None:
            """Menangkap gradien dari backward pass."""
            self.gradients = grad_output[0].detach()

        # Pasang hook pada lapisan target
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Menghasilkan heatmap Grad-CAM untuk citra input.

        Langkah-langkah:
            1. Forward pass untuk mendapatkan aktivasi dan prediksi
            2. Backward pass untuk mendapatkan gradien
            3. Hitung bobot menggunakan Global Average Pooling pada gradien
            4. Hitung kombinasi linier aktivasi berbobot
            5. Terapkan ReLU dan normalisasi

        Args:
            input_tensor: Tensor citra input dengan dimensi (1, 3, H, W).
            target_class: Kelas target untuk Grad-CAM. Jika None,
                          menggunakan kelas dengan probabilitas tertinggi.

        Returns:
            Heatmap Grad-CAM berupa numpy array 2D dengan nilai 0-1.
        """
        # Set model ke mode evaluasi
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Tentukan kelas target
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Reset gradien
        self.model.zero_grad()

        # Backward pass untuk kelas target
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Pastikan gradien dan aktivasi tersedia
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradien atau aktivasi tidak tersedia. "
                "Pastikan hook terdaftar dengan benar."
            )

        # Hitung bobot menggunakan Global Average Pooling pada gradien
        # Bobot ini merepresentasikan pentingnya setiap feature map
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Hitung Grad-CAM: kombinasi linier aktivasi berbobot
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Terapkan ReLU: hanya pertahankan pengaruh positif
        cam = F.relu(cam)

        # Resize heatmap ke ukuran citra input
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        # Normalisasi ke rentang [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


class DeepfakePredictor:
    """
    Kelas untuk melakukan prediksi deepfake pada citra tunggal atau batch.

    Kelas ini mengelola proses pemuatan model, preprocessing citra,
    inferensi, dan visualisasi Grad-CAM.

    Atribut:
        model (DeepfakeCNN): Model CNN yang telah dimuat.
        device (torch.device): Perangkat komputasi.
        transform (transforms.Compose): Pipeline preprocessing citra.
        grad_cam (GradCAM): Instance Grad-CAM untuk visualisasi.

    Contoh Penggunaan:
        >>> predictor = DeepfakePredictor('checkpoints/best_model.pth')
        >>> label, confidence = predictor.predict('test_image.jpg')
        >>> print(f"Prediksi: {label} ({confidence:.2f}%)")
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'auto'
    ) -> None:
        """
        Inisialisasi DeepfakePredictor.

        Args:
            model_path: Path ke file checkpoint model (.pth).
            device: Perangkat komputasi ('cuda', 'mps', 'cpu', atau 'auto').
                    Jika 'auto', akan otomatis mendeteksi perangkat terbaik.

        Raises:
            FileNotFoundError: Jika file model tidak ditemukan.
        """
        # Validasi keberadaan file model
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {model_path}"
            )

        # Deteksi perangkat komputasi
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Muat model dari checkpoint
        self.model = self._load_model(model_path)

        # Siapkan pipeline preprocessing (sama dengan validasi/test)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

        # Inisialisasi Grad-CAM pada blok konvolusi terakhir
        # conv_block4 adalah lapisan konvolusi terakhir yang menghasilkan
        # feature map paling abstrak
        self.grad_cam = GradCAM(
            model=self.model,
            target_layer=self.model.conv_block4
        )

        logger.info(
            f"DeepfakePredictor berhasil diinisialisasi "
            f"(perangkat: {self.device})"
        )

    def _load_model(self, model_path: str) -> DeepfakeCNN:
        """
        Memuat model CNN dari file checkpoint.

        Args:
            model_path: Path ke file checkpoint (.pth).

        Returns:
            Model DeepfakeCNN yang telah dimuat dan siap untuk inferensi.
        """
        # Inisialisasi arsitektur model
        model = DeepfakeCNN(num_classes=2)

        # Muat checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        # Muat state dict model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Jika checkpoint hanya berisi state dict langsung
            model.load_state_dict(checkpoint)

        # Pindahkan ke perangkat dan set mode evaluasi
        model = model.to(self.device)
        model.eval()

        logger.info(f"Model berhasil dimuat dari: {model_path}")

        return model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Melakukan preprocessing pada citra sebelum inferensi.

        Pipeline preprocessing:
            1. Resize ke 256x256
            2. CenterCrop ke 224x224
            3. Konversi ke tensor (0-1)
            4. Normalisasi dengan mean/std ImageNet

        Args:
            image: Objek PIL Image yang akan dipreprocess.

        Returns:
            Tensor citra siap inferensi dengan dimensi (1, 3, 224, 224).
        """
        # Terapkan transformasi dan tambahkan dimensi batch
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Tambah dimensi batch: (1, 3, 224, 224)
        return tensor.to(self.device)

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Melakukan prediksi pada satu citra.

        Args:
            image_path: Path ke file citra yang akan diprediksi.

        Returns:
            Tuple berisi:
                - label (str): Label prediksi ('Real' atau 'Fake').
                - confidence (float): Tingkat kepercayaan prediksi (0-100%).

        Raises:
            FileNotFoundError: Jika file citra tidak ditemukan.
            Exception: Jika terjadi kesalahan saat memproses citra.
        """
        # Validasi keberadaan file citra
        if not Path(image_path).exists():
            raise FileNotFoundError(
                f"File citra tidak ditemukan: {image_path}"
            )

        try:
            # Buka dan preprocess citra
            image = Image.open(image_path).convert('RGB')
            input_tensor = self._preprocess(image)

            # Inferensi tanpa gradien (lebih cepat dan hemat memori)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)

            # Ambil prediksi kelas dan confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            label = CLASS_NAMES[predicted_class.item()]
            confidence_pct = confidence.item() * 100

            logger.info(
                f"Prediksi: {label} ({confidence_pct:.2f}%) - {image_path}"
            )

            return label, confidence_pct

        except Exception as e:
            logger.error(f"Gagal memprediksi citra {image_path}: {e}")
            raise

    def predict_batch(
        self,
        image_paths: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Melakukan prediksi pada batch (kumpulan) citra.

        Args:
            image_paths: Daftar path ke file citra yang akan diprediksi.

        Returns:
            List of tuples, masing-masing berisi:
                - image_path (str): Path citra.
                - label (str): Label prediksi ('Real' atau 'Fake').
                - confidence (float): Tingkat kepercayaan (0-100%).
        """
        results: List[Tuple[str, str, float]] = []

        for image_path in image_paths:
            try:
                label, confidence = self.predict(image_path)
                results.append((image_path, label, confidence))
            except Exception as e:
                logger.error(f"Gagal memprediksi: {image_path}. Error: {e}")
                results.append((image_path, 'Error', 0.0))

        # Cetak ringkasan hasil batch
        print(f"\n{'='*70}")
        print(f" HASIL PREDIKSI BATCH ({len(results)} citra)")
        print(f"{'='*70}")
        for path, label, conf in results:
            status = "✓" if label != 'Error' else "✗"
            print(f"  {status} {Path(path).name:<30} -> {label:<6} ({conf:.2f}%)")
        print(f"{'='*70}\n")

        return results

    def predict_with_gradcam(
        self,
        image_path: str,
        save_path: Optional[str] = None
    ) -> Tuple[str, float, np.ndarray]:
        """
        Melakukan prediksi dengan visualisasi Grad-CAM.

        Menghasilkan prediksi beserta heatmap Grad-CAM yang menunjukkan
        area pada citra yang paling berpengaruh terhadap keputusan
        klasifikasi model.

        Args:
            image_path: Path ke file citra yang akan diprediksi.
            save_path: Path untuk menyimpan visualisasi (opsional).
                       Jika None, visualisasi hanya ditampilkan.

        Returns:
            Tuple berisi:
                - label (str): Label prediksi ('Real' atau 'Fake').
                - confidence (float): Tingkat kepercayaan (0-100%).
                - heatmap (np.ndarray): Heatmap Grad-CAM (H x W, nilai 0-1).

        Raises:
            FileNotFoundError: Jika file citra tidak ditemukan.
        """
        # Validasi keberadaan file citra
        if not Path(image_path).exists():
            raise FileNotFoundError(
                f"File citra tidak ditemukan: {image_path}"
            )

        # Buka dan preprocess citra
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self._preprocess(original_image)

        # Aktifkan gradien untuk Grad-CAM
        input_tensor.requires_grad_(True)

        # Forward pass untuk mendapatkan prediksi
        outputs = self.model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        label = CLASS_NAMES[predicted_class.item()]
        confidence_pct = confidence.item() * 100

        # Generate heatmap Grad-CAM
        heatmap = self.grad_cam.generate(
            input_tensor,
            target_class=predicted_class.item()
        )

        # Visualisasi hasil
        if save_path is not None or True:
            self._visualize_gradcam(
                original_image=original_image,
                heatmap=heatmap,
                label=label,
                confidence=confidence_pct,
                save_path=save_path,
                image_name=Path(image_path).name
            )

        logger.info(
            f"Prediksi Grad-CAM: {label} ({confidence_pct:.2f}%) - "
            f"{image_path}"
        )

        return label, confidence_pct, heatmap

    def _visualize_gradcam(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        label: str,
        confidence: float,
        save_path: Optional[str] = None,
        image_name: str = 'image'
    ) -> None:
        """
        Membuat visualisasi Grad-CAM yang menggabungkan citra asli
        dengan heatmap.

        Menghasilkan gambar dengan 3 panel:
            1. Citra asli
            2. Heatmap Grad-CAM
            3. Overlay (gabungan citra asli + heatmap)

        Args:
            original_image: Citra asli dalam format PIL Image.
            heatmap: Heatmap Grad-CAM (numpy array 2D).
            label: Label prediksi.
            confidence: Tingkat kepercayaan prediksi.
            save_path: Path untuk menyimpan visualisasi (opsional).
            image_name: Nama file citra untuk judul plot.
        """
        # Konversi citra ke numpy array
        img_array = np.array(
            original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        )

        # Resize heatmap ke ukuran citra
        heatmap_resized = cv2.resize(
            heatmap, (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Buat colormap untuk heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay heatmap pada citra asli
        overlay = cv2.addWeighted(
            img_array, 0.6,
            heatmap_colored, 0.4,
            0
        )

        # Buat visualisasi dengan 3 panel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Citra asli
        axes[0].imshow(img_array)
        axes[0].set_title('Citra Asli', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Heatmap Grad-CAM
        im = axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Heatmap Grad-CAM', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Grad-CAM', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Warna label berdasarkan prediksi
        color = '#4CAF50' if label == 'Real' else '#F44336'

        # Judul keseluruhan
        fig.suptitle(
            f'Prediksi: {label} ({confidence:.2f}%)\n{image_name}',
            fontsize=14, fontweight='bold', color=color, y=1.02
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualisasi Grad-CAM disimpan: {save_path}")

        plt.close()


def main() -> None:
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Sistem Deteksi Deepfake - Script Prediksi & Visualisasi Grad-CAM"
    )
    parser.add_argument(
        "--model_path", type=str, default="models/best_model.pth",
        help="Path ke file bobot model (.pth)"
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path ke file citra digital yang akan diuji"
    )
    parser.add_argument(
        "--save_path", type=str, default="results/prediction_result.png",
        help="Path untuk menyimpan citra hasil visualisasi Grad-CAM"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Perangkat keras yang digunakan ('auto', 'cuda', 'mps', 'cpu')"
    )
    
    args = parser.parse_args()
    
    # Validasi path gambar
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: File citra tidak ditemukan di: {args.image_path}")
        sys.exit(1)
        
    # Validasi path model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: File model tidak ditemukan di: {args.model_path}")
        print("Silakan jalankan pelatihan terlebih dahulu atau sediakan file model yang valid.")
        sys.exit(1)
        
    print(f"Menginisialisasi prediktor menggunakan model: {args.model_path} ...")
    try:
        predictor = DeepfakePredictor(model_path=args.model_path, device=args.device)
    except Exception as e:
        print(f"Gagal menginisialisasi prediktor: {e}")
        sys.exit(1)
        
    print(f"Menganalisis gambar: {args.image_path} ...")
    try:
        # Lakukan prediksi dengan Grad-CAM
        label, confidence, _ = predictor.predict_with_gradcam(
            image_path=str(image_path),
            save_path=args.save_path
        )
        
        # Tampilkan hasil di terminal
        separator = "=" * 50
        print("\n" + separator)
        print("           HASIL DETEKSI DEEPFAKE")
        print(separator)
        print(f" Nama File   : {image_path.name}")
        
        # Warna teks hasil prediksi di terminal (Hijau untuk Real, Merah untuk Fake jika terminal mendukung)
        if label == "Real":
            print(f" Prediksi    : \033[92mREAL (ASLI)\033[0m")
        else:
            print(f" Prediksi    : \033[91mFAKE (PALSU/DEEPFAKE)\033[0m")
            
        print(f" Kepercayaan : {confidence:.2f}%")
        print(separator)
        print(f"Visualisasi Grad-CAM berhasil disimpan ke:")
        print(f" -> {args.save_path}\n")
        
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses prediksi: {e}")
        sys.exit(1)

