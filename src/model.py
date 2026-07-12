"""
Modul Arsitektur Model CNN untuk Deteksi Deepfake.

Modul ini berisi definisi arsitektur Convolutional Neural Network (CNN)
yang digunakan untuk mengklasifikasikan citra digital sebagai asli (real)
atau palsu (fake/deepfake). Arsitektur terdiri dari 4 blok konvolusi
diikuti oleh lapisan fully connected.

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class DeepfakeCNN(nn.Module):
    """
    Arsitektur CNN kustom untuk deteksi deepfake pada citra digital.

    Model ini terdiri dari 4 blok konvolusi dan 3 lapisan fully connected.
    Setiap blok konvolusi terdiri dari:
        Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

    Arsitektur Blok Konvolusi:
        - Blok 1: 3 -> 32 filter, kernel 3x3, padding 1
        - Blok 2: 32 -> 64 filter, kernel 3x3, padding 1
        - Blok 3: 64 -> 128 filter, kernel 3x3, padding 1
        - Blok 4: 128 -> 256 filter, kernel 3x3, padding 1

    Arsitektur Fully Connected:
        - FC1: 256*4*4 -> 512
        - FC2: 512 -> 128
        - FC3: 128 -> 2 (kelas: real / fake)

    Atribut:
        conv_block1 (nn.Sequential): Blok konvolusi pertama (3 -> 32 filter).
        conv_block2 (nn.Sequential): Blok konvolusi kedua (32 -> 64 filter).
        conv_block3 (nn.Sequential): Blok konvolusi ketiga (64 -> 128 filter).
        conv_block4 (nn.Sequential): Blok konvolusi keempat (128 -> 256 filter).
        adaptive_pool (nn.AdaptiveAvgPool2d): Lapisan adaptive average pooling.
        classifier (nn.Sequential): Lapisan fully connected untuk klasifikasi.

    Contoh Penggunaan:
        >>> model = DeepfakeCNN(num_classes=2)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = model(input_tensor)
        >>> print(output.shape)  # torch.Size([1, 2])
    """

    def __init__(self, num_classes: int = 2) -> None:
        """
        Inisialisasi arsitektur DeepfakeCNN.

        Args:
            num_classes: Jumlah kelas output (default: 2 untuk real/fake).
        """
        super(DeepfakeCNN, self).__init__()

        # --- Blok Konvolusi 1: 3 channel input -> 32 filter ---
        # Mengekstrak fitur dasar seperti tepi (edge) dan tekstur sederhana
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # --- Blok Konvolusi 2: 32 -> 64 filter ---
        # Mengekstrak fitur yang lebih kompleks seperti pola dan bentuk
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # --- Blok Konvolusi 3: 64 -> 128 filter ---
        # Mengekstrak fitur tingkat tinggi seperti bagian wajah
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # --- Blok Konvolusi 4: 128 -> 256 filter ---
        # Mengekstrak fitur abstrak yang membedakan citra asli dan deepfake
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # --- Adaptive Average Pooling ---
        # Mengubah ukuran feature map menjadi 4x4 agar fleksibel terhadap
        # variasi ukuran input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # --- Lapisan Fully Connected (Classifier) ---
        # Mengklasifikasikan fitur yang telah diekstrak menjadi kelas
        # real (0) atau fake (1)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

        # Menyimpan feature map untuk keperluan visualisasi
        self._feature_maps: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass melalui seluruh arsitektur CNN.

        Args:
            x: Tensor input dengan dimensi (batch_size, 3, 224, 224).

        Returns:
            Tensor output dengan dimensi (batch_size, num_classes)
            berisi logit untuk setiap kelas.
        """
        # Proses melalui setiap blok konvolusi dan simpan feature map
        x = self.conv_block1(x)
        self._feature_maps['conv_block1'] = x.detach()

        x = self.conv_block2(x)
        self._feature_maps['conv_block2'] = x.detach()

        x = self.conv_block3(x)
        self._feature_maps['conv_block3'] = x.detach()

        x = self.conv_block4(x)
        self._feature_maps['conv_block4'] = x.detach()

        # Adaptive average pooling
        x = self.adaptive_pool(x)
        self._feature_maps['adaptive_pool'] = x.detach()

        # Flatten tensor untuk masuk ke lapisan fully connected
        x = x.view(x.size(0), -1)

        # Klasifikasi melalui lapisan fully connected
        x = self.classifier(x)

        return x

    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """
        Mengembalikan feature map dari setiap blok konvolusi.

        Metode ini mengembalikan dictionary berisi feature map yang disimpan
        saat forward pass terakhir. Berguna untuk visualisasi dan analisis
        fitur yang dipelajari oleh model.

        Returns:
            Dictionary dengan nama blok sebagai key dan tensor feature map
            sebagai value. Key yang tersedia:
                - 'conv_block1': Feature map dari blok konvolusi 1
                - 'conv_block2': Feature map dari blok konvolusi 2
                - 'conv_block3': Feature map dari blok konvolusi 3
                - 'conv_block4': Feature map dari blok konvolusi 4
                - 'adaptive_pool': Feature map setelah adaptive pooling

        Raises:
            RuntimeError: Jika forward pass belum pernah dijalankan.
        """
        if not self._feature_maps:
            raise RuntimeError(
                "Feature map belum tersedia. Jalankan forward pass "
                "terlebih dahulu dengan memasukkan input ke model."
            )
        return self._feature_maps


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
    """
    Mencetak dan mengembalikan ringkasan arsitektur model CNN.

    Fungsi ini menampilkan informasi detail mengenai arsitektur model
    termasuk nama lapisan, bentuk output, dan jumlah parameter.

    Args:
        model: Model PyTorch yang akan dianalisis.
        input_size: Ukuran tensor input untuk simulasi forward pass
                    (default: (1, 3, 224, 224) sesuai standar ImageNet).

    Returns:
        String berisi ringkasan lengkap arsitektur model.
    """
    # Menyimpan informasi setiap lapisan
    summary_lines: List[str] = []
    total_params: int = 0
    trainable_params: int = 0

    # Header ringkasan
    separator = "=" * 75
    summary_lines.append(separator)
    summary_lines.append(
        "RINGKASAN ARSITEKTUR MODEL - DeepfakeCNN"
    )
    summary_lines.append(
        "Sistem Deteksi Deepfake pada Citra Digital"
    )
    summary_lines.append(separator)
    summary_lines.append(
        f"{'Nama Lapisan':<35} {'Bentuk Output':<20} {'Jumlah Param':>15}"
    )
    summary_lines.append("-" * 75)

    # Mengiterasi setiap lapisan dalam model
    for name, module in model.named_modules():
        # Lewati kontainer (Sequential, dll.) agar tidak duplikat
        if isinstance(module, (nn.Sequential, DeepfakeCNN)):
            continue

        # Hitung jumlah parameter pada lapisan ini
        params = sum(p.numel() for p in module.parameters(recurse=False))

        # Tentukan bentuk output berdasarkan jenis lapisan
        output_shape = _get_layer_output_shape(module)

        summary_lines.append(
            f"{name:<35} {output_shape:<20} {params:>15,}"
        )
        total_params += params

    # Hitung parameter yang dapat dilatih
    for param in model.parameters():
        trainable_params += param.numel() if param.requires_grad else 0

    non_trainable_params = total_params - trainable_params

    # Footer ringkasan
    summary_lines.append(separator)
    summary_lines.append(f"Total Parameter          : {total_params:>15,}")
    summary_lines.append(f"Parameter Terlatih       : {trainable_params:>15,}")
    summary_lines.append(f"Parameter Tidak Terlatih : {non_trainable_params:>15,}")
    summary_lines.append(
        f"Estimasi Ukuran Model    : {total_params * 4 / (1024 ** 2):>12.2f} MB"
    )
    summary_lines.append(separator)

    # Cetak dan kembalikan ringkasan
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    return summary_text


def _get_layer_output_shape(module: nn.Module) -> str:
    """
    Mendapatkan deskripsi bentuk output dari suatu lapisan.

    Args:
        module: Modul/lapisan PyTorch.

    Returns:
        String deskripsi bentuk output lapisan tersebut.
    """
    if isinstance(module, nn.Conv2d):
        return f"[-, {module.out_channels}, H, W]"
    elif isinstance(module, nn.BatchNorm2d):
        return f"[-, {module.num_features}, H, W]"
    elif isinstance(module, nn.Linear):
        return f"[-, {module.out_features}]"
    elif isinstance(module, nn.MaxPool2d):
        return "[-, C, H/2, W/2]"
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return f"[-, C, {module.output_size[0]}, {module.output_size[1]}]"
    elif isinstance(module, (nn.ReLU, nn.Dropout)):
        return "[sama dengan input]"
    else:
        return "[-]"
