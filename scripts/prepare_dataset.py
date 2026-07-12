"""
Script untuk Mempersiapkan Dataset Deepfake Detection.

Script ini mengambil direktori sumber yang berisi gambar-gambar
dan membaginya menjadi set pelatihan, validasi, dan pengujian
dengan rasio yang dapat dikonfigurasi.

Penggunaan:
    python scripts/prepare_dataset.py \
        --source data/raw \
        --output data \
        --train-ratio 0.7 \
        --val-ratio 0.15 \
        --test-ratio 0.15

Struktur direktori sumber yang diharapkan:
    data/raw/
    ├── real/
    │   ├── gambar_001.jpg
    │   ├── gambar_002.jpg
    │   └── ...
    └── fake/
        ├── gambar_001.jpg
        ├── gambar_002.jpg
        └── ...
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def parse_arguments():
    """Mengurai argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Mempersiapkan dataset untuk pelatihan model deteksi deepfake.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python scripts/prepare_dataset.py --source data/raw --output data
  python scripts/prepare_dataset.py --source data/raw --output data --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path ke direktori sumber yang berisi folder 'real' dan 'fake'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Path ke direktori output (default: data).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proporsi data untuk pelatihan (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proporsi data untuk validasi (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporsi data untuk pengujian (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed untuk reproduktifitas pengacakan (default: 42).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Salin file alih-alih memindahkannya (default: pindahkan).",
    )

    return parser.parse_args()


def validasi_rasio(train_ratio, val_ratio, test_ratio):
    """Memvalidasi bahwa rasio pembagian berjumlah 1.0."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Rasio pembagian harus berjumlah 1.0, "
            f"tetapi mendapatkan {total:.4f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )


def dapatkan_daftar_gambar(direktori):
    """
    Mendapatkan daftar file gambar dari direktori.

    Args:
        direktori: Path ke direktori yang berisi gambar.

    Returns:
        List path file gambar yang ditemukan.
    """
    ekstensi_valid = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    daftar_gambar = []

    direktori_path = Path(direktori)
    if not direktori_path.exists():
        print(f"  [PERINGATAN] Direktori tidak ditemukan: {direktori}")
        return daftar_gambar

    for file in sorted(direktori_path.iterdir()):
        if file.is_file() and file.suffix.lower() in ekstensi_valid:
            daftar_gambar.append(file)

    return daftar_gambar


def bagi_dataset(daftar_gambar, train_ratio, val_ratio):
    """
    Membagi daftar gambar menjadi set train, val, dan test.

    Args:
        daftar_gambar: List path file gambar.
        train_ratio: Proporsi data pelatihan.
        val_ratio: Proporsi data validasi.

    Returns:
        Tuple (train_files, val_files, test_files).
    """
    total = len(daftar_gambar)
    jumlah_train = int(total * train_ratio)
    jumlah_val = int(total * val_ratio)

    train_files = daftar_gambar[:jumlah_train]
    val_files = daftar_gambar[jumlah_train : jumlah_train + jumlah_val]
    test_files = daftar_gambar[jumlah_train + jumlah_val :]

    return train_files, val_files, test_files


def salin_atau_pindahkan_file(daftar_file, direktori_tujuan, gunakan_salin=False):
    """
    Menyalin atau memindahkan file ke direktori tujuan.

    Args:
        daftar_file: List path file yang akan dipindahkan/disalin.
        direktori_tujuan: Path direktori tujuan.
        gunakan_salin: Jika True, salin file; jika False, pindahkan file.
    """
    direktori_tujuan = Path(direktori_tujuan)
    direktori_tujuan.mkdir(parents=True, exist_ok=True)

    for file in daftar_file:
        tujuan = direktori_tujuan / file.name
        if gunakan_salin:
            shutil.copy2(str(file), str(tujuan))
        else:
            shutil.move(str(file), str(tujuan))


def cetak_statistik(nama_kelas, train_files, val_files, test_files):
    """Mencetak statistik pembagian dataset untuk satu kelas."""
    total = len(train_files) + len(val_files) + len(test_files)
    if total == 0:
        print(f"  [{nama_kelas.upper()}] Tidak ada gambar ditemukan.")
        return

    print(f"  [{nama_kelas.upper()}]")
    print(f"    Total gambar : {total}")
    print(f"    Pelatihan    : {len(train_files):>6} ({len(train_files)/total*100:.1f}%)")
    print(f"    Validasi     : {len(val_files):>6} ({len(val_files)/total*100:.1f}%)")
    print(f"    Pengujian    : {len(test_files):>6} ({len(test_files)/total*100:.1f}%)")
    print()


def main():
    """Fungsi utama untuk mempersiapkan dataset."""
    args = parse_arguments()

    print("=" * 60)
    print("  PERSIAPAN DATASET - Sistem Deteksi Deepfake")
    print("=" * 60)
    print()

    # Validasi rasio pembagian
    validasi_rasio(args.train_ratio, args.val_ratio, args.test_ratio)

    # Tetapkan seed untuk reproduktifitas
    random.seed(args.seed)

    print(f"  Direktori sumber  : {args.source}")
    print(f"  Direktori output  : {args.output}")
    print(f"  Rasio pembagian   : train={args.train_ratio}, "
          f"val={args.val_ratio}, test={args.test_ratio}")
    print(f"  Mode              : {'Salin' if args.copy else 'Pindahkan'}")
    print(f"  Seed              : {args.seed}")
    print()

    # Definisikan kelas-kelas yang diharapkan
    kelas_kelas = ["real", "fake"]
    split_names = ["train", "val", "test"]

    # Proses setiap kelas
    total_keseluruhan = {"train": 0, "val": 0, "test": 0}

    for kelas in kelas_kelas:
        direktori_sumber = Path(args.source) / kelas

        # Dapatkan daftar gambar
        daftar_gambar = dapatkan_daftar_gambar(direktori_sumber)

        if not daftar_gambar:
            print(f"  [PERINGATAN] Tidak ada gambar ditemukan di: {direktori_sumber}")
            continue

        # Acak urutan gambar
        random.shuffle(daftar_gambar)

        # Bagi dataset
        train_files, val_files, test_files = bagi_dataset(
            daftar_gambar, args.train_ratio, args.val_ratio
        )

        # Cetak statistik untuk kelas ini
        cetak_statistik(kelas, train_files, val_files, test_files)

        # Salin/pindahkan file ke struktur yang sesuai
        split_data = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, files in split_data.items():
            direktori_tujuan = Path(args.output) / split_name / kelas
            salin_atau_pindahkan_file(files, direktori_tujuan, args.copy)
            total_keseluruhan[split_name] += len(files)

    # Cetak ringkasan keseluruhan
    print("-" * 60)
    print("  RINGKASAN KESELURUHAN")
    print("-" * 60)
    grand_total = sum(total_keseluruhan.values())
    if grand_total > 0:
        for split_name in split_names:
            jumlah = total_keseluruhan[split_name]
            print(
                f"    {split_name.capitalize():10s} : {jumlah:>6} "
                f"({jumlah/grand_total*100:.1f}%)"
            )
        print(f"    {'Total':10s} : {grand_total:>6}")
    else:
        print("    Tidak ada gambar yang diproses.")

    print()
    print("  Struktur direktori yang dibuat:")
    print(f"    {args.output}/")
    for split_name in split_names:
        print(f"    ├── {split_name}/")
        for kelas in kelas_kelas:
            simbol = "└" if kelas == kelas_kelas[-1] else "├"
            dir_path = Path(args.output) / split_name / kelas
            jumlah = len(list(dir_path.glob("*"))) if dir_path.exists() else 0
            print(f"    │   {simbol}── {kelas}/ ({jumlah} gambar)")
    print()
    print("  [SELESAI] Dataset berhasil dipersiapkan!")
    print("=" * 60)


if __name__ == "__main__":
    main()
