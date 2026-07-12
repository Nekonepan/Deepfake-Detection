"""
Modul Dataset dan DataLoader untuk Deteksi Deepfake.

Modul ini menangani pemuatan dataset citra deepfake, augmentasi data,
dan pembuatan DataLoader untuk proses pelatihan, validasi, dan pengujian.

Struktur folder dataset yang diharapkan:
    data_dir/
    ├── train/
    │   ├── real/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   └── fake/
    │       ├── img001.jpg
    │       └── ...
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Konfigurasi logger untuk modul dataset
logger = logging.getLogger(__name__)

# Konstanta normalisasi menggunakan standar ImageNet
# Nilai mean dan std dihitung dari jutaan citra pada dataset ImageNet
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Ukuran citra input yang sesuai dengan arsitektur model
IMAGE_SIZE: int = 224

# Mapping label: real = 0, fake = 1
CLASS_NAMES: Dict[str, int] = {'real': 0, 'fake': 1}

# Ekstensi file citra yang didukung
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
)


class DeepfakeDataset(Dataset):
    """
    Dataset kustom untuk memuat citra deepfake.

    Kelas ini membaca citra dari struktur folder yang terorganisasi
    ke dalam subfolder 'real' dan 'fake', lalu memberikan label
    yang sesuai (real=0, fake=1).

    Atribut:
        root_dir (Path): Direktori root dataset.
        split (str): Jenis split data ('train', 'val', atau 'test').
        transform (transforms.Compose): Pipeline transformasi citra.
        image_paths (List[str]): Daftar path ke semua file citra.
        labels (List[int]): Daftar label untuk setiap citra.
        class_names (Dict[str, int]): Mapping nama kelas ke indeks.

    Contoh Penggunaan:
        >>> dataset = DeepfakeDataset(
        ...     root_dir='data/train',
        ...     transform=get_transforms('train'),
        ...     split='train'
        ... )
        >>> image, label = dataset[0]
        >>> print(f"Label: {label}, Shape: {image.shape}")
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        split: str = 'train'
    ) -> None:
        """
        Inisialisasi DeepfakeDataset.

        Args:
            root_dir: Path ke direktori yang berisi subfolder 'real' dan 'fake'.
            transform: Pipeline transformasi yang akan diterapkan pada citra.
                       Jika None, akan menggunakan transformasi default
                       berdasarkan split.
            split: Jenis pembagian data ('train', 'val', atau 'test').

        Raises:
            FileNotFoundError: Jika direktori root_dir tidak ditemukan.
            ValueError: Jika split bukan salah satu dari 'train', 'val', 'test'.
        """
        # Validasi parameter split
        valid_splits = ('train', 'val', 'test')
        if split not in valid_splits:
            raise ValueError(
                f"Split '{split}' tidak valid. "
                f"Pilihan yang tersedia: {valid_splits}"
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.class_names = CLASS_NAMES

        # Validasi keberadaan direktori
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Direktori dataset tidak ditemukan: {self.root_dir}"
            )

        # Gunakan transformasi default jika tidak diberikan
        self.transform = transform if transform is not None else get_transforms(split)

        # Muat daftar path citra dan label
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self._load_dataset()

        logger.info(
            f"Dataset '{split}' berhasil dimuat: "
            f"{len(self.image_paths)} citra "
            f"(Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})"
        )

    def _load_dataset(self) -> None:
        """
        Memuat daftar path citra dan label dari struktur folder.

        Metode ini membaca semua file citra dari subfolder 'real' dan 'fake',
        lalu menyimpan path dan label yang sesuai.

        Raises:
            FileNotFoundError: Jika subfolder 'real' atau 'fake' tidak ditemukan.
        """
        for class_name, label in self.class_names.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                logger.warning(
                    f"Direktori kelas '{class_name}' tidak ditemukan: "
                    f"{class_dir}. Melewati kelas ini."
                )
                continue

            # Kumpulkan semua file citra dengan ekstensi yang didukung
            image_count = 0
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.image_paths.append(str(file_path))
                    self.labels.append(label)
                    image_count += 1

            logger.info(
                f"  Kelas '{class_name}' ({self.split}): "
                f"{image_count} citra ditemukan"
            )

        if len(self.image_paths) == 0:
            logger.warning(
                f"Tidak ada citra yang ditemukan di {self.root_dir}. "
                f"Pastikan struktur folder sesuai (real/ dan fake/)."
            )

    def __len__(self) -> int:
        """
        Mengembalikan jumlah total citra dalam dataset.

        Returns:
            Jumlah citra dalam dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Mengambil satu sampel citra dan labelnya berdasarkan indeks.

        Args:
            idx: Indeks citra yang akan diambil.

        Returns:
            Tuple berisi:
                - image (torch.Tensor): Tensor citra setelah transformasi
                  dengan dimensi (3, 224, 224).
                - label (int): Label kelas (0=real, 1=fake).

        Raises:
            IndexError: Jika indeks di luar jangkauan dataset.
        """
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(
                f"Indeks {idx} di luar jangkauan dataset "
                f"(ukuran: {len(self.image_paths)})"
            )

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Buka citra dan konversi ke format RGB
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(
                f"Gagal memuat citra: {image_path}. Error: {e}"
            )
            # Kembalikan tensor kosong jika gagal memuat citra
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        # Terapkan transformasi pada citra
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Menghitung distribusi kelas dalam dataset.

        Returns:
            Dictionary berisi jumlah citra per kelas.
            Contoh: {'real': 5000, 'fake': 5000}
        """
        distribution: Dict[str, int] = {}
        for class_name, label in self.class_names.items():
            distribution[class_name] = self.labels.count(label)
        return distribution


def get_transforms(split: str) -> transforms.Compose:
    """
    Membuat pipeline transformasi citra sesuai dengan jenis split data.

    Untuk data training, diterapkan augmentasi data untuk meningkatkan
    keragaman data dan mencegah overfitting. Augmentasi meliputi:
        - RandomHorizontalFlip: Pencerminan horizontal secara acak
        - RandomRotation: Rotasi acak hingga 15 derajat
        - ColorJitter: Variasi warna (kecerahan, kontras, saturasi, hue)
        - RandomAffine: Transformasi affine acak (translasi dan skala)

    Untuk data validasi dan test, hanya diterapkan transformasi standar:
        - Resize: Mengubah ukuran ke 256x256
        - CenterCrop: Memotong bagian tengah menjadi 224x224
        - Normalisasi menggunakan mean dan std ImageNet

    Args:
        split: Jenis split data ('train', 'val', atau 'test').

    Returns:
        Pipeline transformasi (transforms.Compose) yang sesuai.

    Raises:
        ValueError: Jika split bukan salah satu dari 'train', 'val', 'test'.
    """
    if split == 'train':
        # Pipeline augmentasi untuk data training
        # Augmentasi membantu model belajar fitur yang lebih robust
        # dan mengurangi risiko overfitting
        return transforms.Compose([
            # Ubah ukuran citra menjadi 256x256 sebelum cropping
            transforms.Resize((256, 256)),

            # Potong secara acak menjadi 224x224 untuk variasi posisi
            transforms.RandomCrop(IMAGE_SIZE),

            # Pencerminan horizontal dengan probabilitas 50%
            # Simulasi variasi orientasi wajah
            transforms.RandomHorizontalFlip(p=0.5),

            # Rotasi acak hingga 15 derajat
            # Simulasi kemiringan kamera
            transforms.RandomRotation(degrees=15),

            # Variasi warna untuk robustness terhadap perbedaan pencahayaan
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),

            # Transformasi affine: translasi dan skala acak
            # Simulasi variasi jarak dan posisi subjek
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),

            # Konversi ke tensor PyTorch (nilai piksel 0-1)
            transforms.ToTensor(),

            # Normalisasi menggunakan statistik ImageNet
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])
    elif split in ('val', 'test'):
        # Pipeline standar untuk validasi dan pengujian
        # Tanpa augmentasi agar evaluasi konsisten dan reproducible
        return transforms.Compose([
            # Ubah ukuran menjadi 256x256
            transforms.Resize((256, 256)),

            # Potong bagian tengah menjadi 224x224
            transforms.CenterCrop(IMAGE_SIZE),

            # Konversi ke tensor PyTorch
            transforms.ToTensor(),

            # Normalisasi menggunakan statistik ImageNet
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])
    else:
        raise ValueError(
            f"Split '{split}' tidak valid. "
            f"Pilihan yang tersedia: ('train', 'val', 'test')"
        )


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membuat DataLoader untuk data training, validasi, dan pengujian.

    Fungsi ini membuat tiga DataLoader yang siap digunakan untuk proses
    pelatihan model. Data training di-shuffle untuk setiap epoch,
    sedangkan data validasi dan test tidak di-shuffle.

    Args:
        data_dir: Path ke direktori utama dataset yang berisi subfolder
                  'train/', 'val/', dan 'test/'.
        batch_size: Jumlah sampel per batch (default: 32).
        num_workers: Jumlah proses paralel untuk memuat data (default: 4).

    Returns:
        Tuple berisi tiga DataLoader:
            - train_loader: DataLoader untuk data training
            - val_loader: DataLoader untuk data validasi
            - test_loader: DataLoader untuk data pengujian

    Raises:
        FileNotFoundError: Jika direktori data_dir tidak ditemukan.

    Contoh Penggunaan:
        >>> train_loader, val_loader, test_loader = get_data_loaders(
        ...     data_dir='data/',
        ...     batch_size=32,
        ...     num_workers=4
        ... )
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Direktori dataset tidak ditemukan: {data_path}"
        )

    # Buat dataset untuk setiap split
    logger.info("Memuat dataset training...")
    train_dataset = DeepfakeDataset(
        root_dir=str(data_path / 'train'),
        transform=get_transforms('train'),
        split='train'
    )

    logger.info("Memuat dataset validasi...")
    val_dataset = DeepfakeDataset(
        root_dir=str(data_path / 'val'),
        transform=get_transforms('val'),
        split='val'
    )

    logger.info("Memuat dataset pengujian...")
    test_dataset = DeepfakeDataset(
        root_dir=str(data_path / 'test'),
        transform=get_transforms('test'),
        split='test'
    )

    # Buat DataLoader untuk setiap dataset
    # pin_memory=True mempercepat transfer data ke GPU
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Acak urutan data setiap epoch
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True          # Buang batch terakhir jika tidak penuh
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,          # Tidak perlu diacak untuk validasi
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,          # Tidak perlu diacak untuk pengujian
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Tampilkan ringkasan dataset
    logger.info(
        f"\nRingkasan Dataset:"
        f"\n  Training  : {len(train_dataset)} citra"
        f"\n  Validasi  : {len(val_dataset)} citra"
        f"\n  Pengujian : {len(test_dataset)} citra"
        f"\n  Batch Size: {batch_size}"
    )

    return train_loader, val_loader, test_loader
