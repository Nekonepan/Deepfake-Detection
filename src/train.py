"""
Modul Pipeline Pelatihan Model CNN untuk Deteksi Deepfake.

Modul ini berisi kelas Trainer yang mengatur seluruh proses pelatihan
model CNN, termasuk training loop, validasi, early stopping,
penjadwalan learning rate, dan penyimpanan checkpoint.

Judul Proyek:
    Sistem Deteksi Deepfake pada Citra Digital Menggunakan Metode
    Convolutional Neural Network (CNN)
"""

import os
import time
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import EarlyStopping

# Konfigurasi logger untuk modul pelatihan
logger = logging.getLogger(__name__)


class Trainer:
    """
    Kelas untuk mengatur proses pelatihan model CNN deteksi deepfake.

    Kelas ini mengelola seluruh pipeline pelatihan termasuk:
        - Training loop per epoch dengan progress bar
        - Validasi per epoch
        - Early stopping untuk mencegah overfitting
        - Penjadwalan learning rate (ReduceLROnPlateau)
        - Penyimpanan model terbaik berdasarkan akurasi validasi
        - Pencatatan riwayat pelatihan (loss dan akurasi)

    Atribut:
        model (nn.Module): Model CNN yang akan dilatih.
        train_loader (DataLoader): DataLoader untuk data training.
        val_loader (DataLoader): DataLoader untuk data validasi.
        criterion (nn.Module): Fungsi loss (CrossEntropyLoss).
        optimizer (Optimizer): Optimizer (Adam).
        scheduler: Scheduler learning rate (ReduceLROnPlateau).
        device (torch.device): Perangkat komputasi (CUDA/MPS/CPU).
        save_dir (Path): Direktori untuk menyimpan checkpoint.
        history (Dict): Riwayat pelatihan (loss, akurasi per epoch).
        best_val_acc (float): Akurasi validasi terbaik yang tercapai.

    Contoh Penggunaan:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=nn.CrossEntropyLoss(),
        ...     optimizer=torch.optim.Adam(model.parameters()),
        ...     scheduler=scheduler,
        ...     device=torch.device('cuda'),
        ...     save_dir='checkpoints/'
        ... )
        >>> history = trainer.train(num_epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        save_dir: str = 'checkpoints',
        early_stopping_patience: int = 10
    ) -> None:
        """
        Inisialisasi Trainer.

        Args:
            model: Model CNN yang akan dilatih.
            train_loader: DataLoader untuk data training.
            val_loader: DataLoader untuk data validasi.
            criterion: Fungsi loss untuk menghitung error.
            optimizer: Optimizer untuk memperbarui parameter model.
            scheduler: Scheduler learning rate (opsional).
            device: Perangkat komputasi. Jika None, akan otomatis
                    mendeteksi CUDA/MPS/CPU.
            save_dir: Direktori untuk menyimpan model dan checkpoint.
            early_stopping_patience: Jumlah epoch tanpa perbaikan
                                     sebelum menghentikan pelatihan.
        """
        # Deteksi perangkat komputasi secara otomatis
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Pindahkan model ke perangkat yang dipilih
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Buat direktori penyimpanan jika belum ada
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Inisialisasi early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=0.001,
            mode='max'  # Memantau akurasi validasi (semakin tinggi semakin baik)
        )

        # Inisialisasi riwayat pelatihan
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # Variabel untuk melacak model terbaik
        self.best_val_acc: float = 0.0
        self.best_model_state: Optional[Dict[str, Any]] = None

        logger.info(
            f"Trainer diinisialisasi:"
            f"\n  Perangkat       : {self.device}"
            f"\n  Jumlah Parameter: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            f"\n  Direktori Simpan: {self.save_dir}"
            f"\n  Early Stopping  : patience={early_stopping_patience}"
        )

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Melatih model untuk satu epoch penuh.

        Proses pelatihan satu epoch meliputi:
            1. Iterasi semua batch data training
            2. Forward pass untuk mendapatkan prediksi
            3. Perhitungan loss menggunakan CrossEntropyLoss
            4. Backward pass untuk menghitung gradien
            5. Update parameter menggunakan optimizer

        Args:
            epoch: Nomor epoch saat ini (untuk tampilan progress bar).

        Returns:
            Tuple berisi:
                - avg_loss (float): Rata-rata loss selama satu epoch.
                - accuracy (float): Akurasi training (persentase 0-100).
        """
        # Set model ke mode training (aktifkan Dropout dan BatchNorm training)
        self.model.train()

        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        # Progress bar untuk memantau proses training
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Training]",
            leave=True,
            ncols=100
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Pindahkan data ke perangkat komputasi
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Reset gradien dari iterasi sebelumnya
            self.optimizer.zero_grad()

            # Forward pass: hitung prediksi model
            outputs = self.model(images)

            # Hitung loss (error antara prediksi dan label sebenarnya)
            loss = self.criterion(outputs, labels)

            # Backward pass: hitung gradien melalui backpropagation
            loss.backward()

            # Gradient clipping untuk mencegah exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            # Update parameter model berdasarkan gradien
            self.optimizer.step()

            # Akumulasi statistik
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar dengan loss dan akurasi terkini
            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Hitung rata-rata loss dan akurasi untuk seluruh epoch
        if total == 0:
            logger.warning(
                f"Epoch {epoch}: Tidak ada data yang diproses. "
                f"Periksa apakah dataset training terisi dan "
                f"batch_size ({self.train_loader.batch_size}) tidak lebih besar dari jumlah data."
            )
            return 0.0, 0.0

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Mengevaluasi model pada data validasi.

        Proses validasi dilakukan tanpa menghitung gradien
        (torch.no_grad) untuk efisiensi memori dan kecepatan.

        Returns:
            Tuple berisi:
                - avg_loss (float): Rata-rata loss pada data validasi.
                - accuracy (float): Akurasi validasi (persentase 0-100).
        """
        # Set model ke mode evaluasi (nonaktifkan Dropout, BatchNorm inference)
        self.model.eval()

        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        progress_bar = tqdm(
            self.val_loader,
            desc="         [Validasi]",
            leave=True,
            ncols=100
        )

        for images, labels in progress_bar:
            # Pindahkan data ke perangkat komputasi
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass (tanpa gradien)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Akumulasi statistik
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Hitung rata-rata loss dan akurasi
        if total == 0:
            logger.warning(
                "Validasi: Tidak ada data yang diproses. "
                "Periksa apakah dataset validasi terisi."
            )
            return 0.0, 0.0

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int = 50) -> Dict[str, List[float]]:
        """
        Menjalankan pelatihan model untuk sejumlah epoch yang ditentukan.

        Proses pelatihan lengkap meliputi:
            1. Training per epoch
            2. Validasi per epoch
            3. Update learning rate scheduler
            4. Penyimpanan model terbaik
            5. Pemeriksaan early stopping
            6. Pencatatan riwayat pelatihan

        Args:
            num_epochs: Jumlah epoch maksimum untuk pelatihan (default: 50).

        Returns:
            Dictionary riwayat pelatihan berisi:
                - 'train_loss': List loss training per epoch
                - 'train_acc': List akurasi training per epoch
                - 'val_loss': List loss validasi per epoch
                - 'val_acc': List akurasi validasi per epoch
                - 'learning_rate': List learning rate per epoch
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Memulai Pelatihan Model")
        logger.info(f"Jumlah Epoch: {num_epochs}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # --- Fase Training ---
            train_loss, train_acc = self.train_one_epoch(epoch)

            # --- Fase Validasi ---
            val_loss, val_acc = self.validate()

            # Catat learning rate saat ini
            current_lr = self.optimizer.param_groups[0]['lr']

            # Simpan riwayat pelatihan
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # Hitung durasi epoch
            epoch_time = time.time() - epoch_start

            # Cetak ringkasan epoch
            print(
                f"\n{'─'*60}"
                f"\n Epoch {epoch}/{num_epochs} "
                f"({epoch_time:.1f}s)"
                f"\n  Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}%"
                f"\n  Val Loss  : {val_loss:.4f} | "
                f"Val Acc  : {val_acc:.2f}%"
                f"\n  LR: {current_lr:.6f}"
                f"\n{'─'*60}"
            )

            # --- Update Learning Rate Scheduler ---
            # ReduceLROnPlateau menurunkan LR jika val_loss tidak membaik
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # --- Simpan Model Terbaik ---
            if val_acc > self.best_val_acc:
                improvement = val_acc - self.best_val_acc
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )
                self.save_checkpoint(
                    epoch=epoch,
                    val_acc=val_acc,
                    is_best=True
                )
                logger.info(
                    f"  ★ Model terbaik baru! "
                    f"Val Acc: {val_acc:.2f}% "
                    f"(+{improvement:.2f}%)"
                )
                print(
                    f"  ★ Model terbaik baru disimpan! "
                    f"(+{improvement:.2f}%)"
                )

            # --- Simpan Checkpoint Periodik ---
            if epoch % 5 == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    val_acc=val_acc,
                    is_best=False
                )

            # --- Pemeriksaan Early Stopping ---
            self.early_stopping(val_acc)
            if self.early_stopping.early_stop:
                logger.info(
                    f"\n⚠ Early Stopping pada epoch {epoch}! "
                    f"Tidak ada perbaikan selama "
                    f"{self.early_stopping.patience} epoch."
                )
                print(
                    f"\n⚠ Early Stopping pada epoch {epoch}! "
                    f"Model terbaik: Val Acc = {self.best_val_acc:.2f}%"
                )
                break

        # Hitung total waktu pelatihan
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)

        # --- Muat Kembali Model Terbaik ---
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Model terbaik berhasil dimuat kembali.")

        # Cetak ringkasan akhir pelatihan
        print(
            f"\n{'='*60}"
            f"\n PELATIHAN SELESAI"
            f"\n{'='*60}"
            f"\n  Total Waktu       : {minutes}m {seconds}s"
            f"\n  Epoch Terakhir    : {epoch}"
            f"\n  Best Val Accuracy : {self.best_val_acc:.2f}%"
            f"\n{'='*60}\n"
        )

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        is_best: bool = False
    ) -> None:
        """
        Menyimpan checkpoint model ke disk.

        Checkpoint berisi semua informasi yang diperlukan untuk
        melanjutkan pelatihan atau melakukan inferensi:
            - State dict model
            - State dict optimizer
            - State dict scheduler
            - Epoch saat ini
            - Akurasi validasi terbaik
            - Riwayat pelatihan

        Args:
            epoch: Nomor epoch saat ini.
            val_acc: Akurasi validasi pada epoch ini.
            is_best: True jika ini adalah model terbaik sejauh ini.
        """
        checkpoint: Dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }

        # Simpan state scheduler jika ada
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            # Simpan model terbaik dengan nama khusus
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Model terbaik disimpan: {best_path}")
        else:
            # Simpan checkpoint periodik
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint disimpan: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Memuat checkpoint model dari disk untuk melanjutkan pelatihan.

        Args:
            checkpoint_path: Path ke file checkpoint (.pth).

        Returns:
            Nomor epoch terakhir dari checkpoint yang dimuat.

        Raises:
            FileNotFoundError: Jika file checkpoint tidak ditemukan.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: {checkpoint_path}"
            )

        logger.info(f"Memuat checkpoint dari: {checkpoint_path}")

        # Muat checkpoint ke perangkat yang sesuai
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Pulihkan state model dan optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Pulihkan state scheduler jika tersedia
        if (self.scheduler is not None
                and 'scheduler_state_dict' in checkpoint):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Pulihkan riwayat pelatihan
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        # Pulihkan akurasi terbaik
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

        epoch = checkpoint['epoch']
        logger.info(
            f"Checkpoint berhasil dimuat (Epoch {epoch}, "
            f"Val Acc: {checkpoint.get('val_acc', 0.0):.2f}%)"
        )

        return epoch


def main() -> None:
    import argparse
    import yaml
    from pathlib import Path
    
    # Import lokal untuk mencegah circular imports
    from .model import DeepfakeCNN, get_model_summary
    from .dataset import get_data_loaders
    from .evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_training_history
    from .utils import set_seed, get_device, setup_logging, create_dirs
    
    parser = argparse.ArgumentParser(
        description="Sistem Deteksi Deepfake - Script Pelatihan Model CNN"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Direktori utama dataset (harus berisi train/, val/, test/)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
        help="Path ke file konfigurasi YAML"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Jumlah epoch pelatihan (mengabaikan config jika diisi)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Ukuran batch (mengabaikan config jika diisi)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate awal (mengabaikan config jika diisi)"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Direktori penyimpanan model (mengabaikan config jika diisi)"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Direktori penyimpanan hasil evaluasi"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed acak untuk reprodusibilitas"
    )
    
    args = parser.parse_args()
    
    # 1. Muat Konfigurasi dari YAML
    config_dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            print(f"Peringatan: Gagal memuat file konfigurasi '{args.config}'. Menggunakan nilai default. Error: {e}")
            
    # Ekstrak nilai konfigurasi dengan fallback ke default
    model_config = config_dict.get('model', {})
    train_config = config_dict.get('training', {})
    paths_config = config_dict.get('paths', {})
    
    epochs = args.epochs if args.epochs is not None else train_config.get('num_epochs', 50)
    batch_size = args.batch_size if args.batch_size is not None else train_config.get('batch_size', 32)
    learning_rate = args.lr if args.lr is not None else train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 0.0001)
    num_workers = train_config.get('num_workers', 4)
    early_patience = train_config.get('early_stopping_patience', 10)
    scheduler_patience = train_config.get('scheduler_patience', 5)
    scheduler_factor = train_config.get('scheduler_factor', 0.1)
    
    data_dir = args.data_dir
    model_dir = args.save_dir if args.save_dir is not None else paths_config.get('model_dir', 'models')
    results_dir = args.results_dir if args.results_dir is not None else paths_config.get('results_dir', 'results')
    log_dir = paths_config.get('log_dir', 'logs')
    seed = args.seed if args.seed is not None else config_dict.get('seed', 42)
    
    # 2. Buat Direktori dan Setup Logging
    create_dirs([model_dir, results_dir, log_dir])
    setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("Memulai Proses Pelatihan Deteksi Deepfake")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(seed)
    
    # Set device
    device = get_device()
    
    # 3. Pemuatan Data (DataLoader)
    logger.info(f"Memuat dataset dari direktori: {data_dir}")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers
        )
    except Exception as e:
        logger.error(f"Gagal memuat dataset: {e}")
        logger.error("Silakan periksa apakah direktori dataset sudah benar dan terisi data.")
        return
        
    # 4. Inisialisasi Model CNN
    logger.info("Menginisialisasi arsitektur model DeepfakeCNN...")
    model = DeepfakeCNN(num_classes=model_config.get('num_classes', 2))
    model = model.to(device)
    
    # Tampilkan summary model
    get_model_summary(model)
    
    # 5. Konfigurasi Loss, Optimizer, dan Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience
    )
    
    # 6. Inisialisasi Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=model_dir,
        early_stopping_patience=early_patience
    )
    
    # Jalankan Pelatihan
    logger.info("Memulai loop pelatihan...")
    history = trainer.train(num_epochs=epochs)
    
    # 7. Evaluasi Model pada Data Pengujian (Test Set)
    logger.info("Menjalankan evaluasi pada dataset pengujian (test set)...")
    try:
        eval_results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        # Simpan hasil plot evaluasi
        plot_training_history(
            history=history,
            save_path=os.path.join(results_dir, 'training_history.png')
        )
        plot_confusion_matrix(
            cm=eval_results['confusion_matrix'],
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )
        plot_roc_curve(
            fpr=eval_results['fpr'],
            tpr=eval_results['tpr'],
            auc_score=eval_results['auc_score'],
            save_path=os.path.join(results_dir, 'roc_curve.png')
        )
        
        # Simpan laporan klasifikasi ke file teks
        report_path = os.path.join(results_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LAPORAN EVALUASI MODEL DETEKSI DEEPFAKE\n")
            f.write("=" * 50 + "\n")
            f.write(f"Akurasi Pengujian  : {eval_results['accuracy'] * 100:.2f}%\n")
            f.write(f"Presisi Pengujian  : {eval_results['precision'] * 100:.2f}%\n")
            f.write(f"Recall Pengujian   : {eval_results['recall'] * 100:.2f}%\n")
            f.write(f"F1-Score Pengujian : {eval_results['f1_score'] * 100:.2f}%\n")
            f.write(f"AUC Score          : {eval_results['auc_score']:.4f}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Laporan Detail Per-Kelas:\n")
            f.write(eval_results['classification_report'])
            
        logger.info(f"Laporan klasifikasi disimpan ke: {report_path}")
        logger.info("Proses pelatihan dan evaluasi selesai dengan sukses! 🎉")
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat evaluasi model: {e}")

