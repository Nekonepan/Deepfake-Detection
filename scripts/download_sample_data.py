"""
Script untuk Mengunduh dan Membuat Data Sampel untuk Pengujian.

Script ini menyediakan:
1. Informasi tentang dataset deepfake populer yang dapat diunduh
2. Pembuatan gambar placeholder untuk pengujian pipeline

Penggunaan:
    python scripts/download_sample_data.py
    python scripts/download_sample_data.py --output data/sample --count 20

Dataset deepfake populer:
- FaceForensics++
- Deepfake Detection Challenge (DFDC)
- Celeb-DF
"""

import argparse
import os
import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[ERROR] Library Pillow tidak ditemukan.")
    print("Silakan instal dengan: pip install Pillow")
    exit(1)


# ============================================================
# Informasi Dataset Deepfake Populer
# ============================================================

INFORMASI_DATASET = """
╔══════════════════════════════════════════════════════════════╗
║        DATASET DEEPFAKE POPULER UNTUK PENELITIAN            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. FaceForensics++ (FF++)                                   ║
║     ─────────────────────────────────────────────            ║
║     Deskripsi : Dataset benchmark standar untuk deteksi       ║
║                 deepfake dengan 1000 video asli dan           ║
║                 manipulasi menggunakan 4 metode               ║
║     Metode    : Deepfakes, Face2Face, FaceSwap, NeuralTex    ║
║     Ukuran    : ~600GB (video lengkap)                       ║
║     Akses     : Memerlukan persetujuan akademis               ║
║     URL       : https://github.com/ondyari/FaceForensics     ║
║                                                              ║
║  2. Deepfake Detection Challenge (DFDC)                      ║
║     ─────────────────────────────────────────────            ║
║     Deskripsi : Dataset terbesar dari Facebook/Meta           ║
║                 dengan lebih dari 100.000 video klip          ║
║     Ukuran    : ~470GB                                       ║
║     Akses     : Registrasi di Kaggle                         ║
║     URL       : https://www.kaggle.com/c/deepfake-detection  ║
║                 -challenge                                    ║
║                                                              ║
║  3. Celeb-DF (v2)                                            ║
║     ─────────────────────────────────────────────            ║
║     Deskripsi : Dataset deepfake berkualitas tinggi dari      ║
║                 wajah selebriti dengan 590 video asli         ║
║                 dan 5639 video deepfake                       ║
║     Ukuran    : ~15GB                                        ║
║     Akses     : Formulir permintaan di website                ║
║     URL       : https://github.com/yuezunli/celeb-deepfake   ║
║                 detection                                     ║
║                                                              ║
║  4. WildDeepfake                                             ║
║     ─────────────────────────────────────────────            ║
║     Deskripsi : Dataset dari deepfake yang ditemukan di       ║
║                 internet (in-the-wild)                        ║
║     Ukuran    : ~4GB                                         ║
║     Akses     : Formulir permintaan                          ║
║     URL       : https://github.com/deepfakeinthewild/        ║
║                 deepfake-in-the-wild                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

CATATAN: Sebagian besar dataset memerlukan persetujuan untuk
         tujuan penelitian akademis. Pastikan untuk mengikuti
         ketentuan penggunaan masing-masing dataset.
"""


def parse_arguments():
    """Mengurai argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Membuat data sampel placeholder untuk pengujian pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample",
        help="Direktori output untuk data sampel (default: data/sample).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Jumlah gambar sampel per kelas (default: 10).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Ukuran gambar dalam piksel (default: 224).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed untuk reproduktifitas (default: 42).",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Hanya tampilkan informasi dataset tanpa membuat gambar.",
    )
    return parser.parse_args()


def buat_gambar_placeholder_real(ukuran, indeks, seed):
    """
    Membuat gambar placeholder yang merepresentasikan gambar 'real'.

    Gambar real dibuat dengan pola warna yang lebih natural dan halus,
    menggunakan gradien lembut dan noise minimal.

    Args:
        ukuran: Ukuran gambar (piksel).
        indeks: Nomor indeks gambar.
        seed: Seed untuk pengacakan.

    Returns:
        Objek PIL.Image.
    """
    random.seed(seed + indeks)

    # Warna dasar natural (tone kulit, latar belakang alami)
    warna_dasar = random.choice([
        (210, 180, 160),  # Tone kulit terang
        (180, 150, 130),  # Tone kulit medium
        (160, 130, 110),  # Tone kulit gelap
        (200, 210, 220),  # Latar belakang biru muda
        (220, 215, 200),  # Latar belakang krem
    ])

    # Buat gambar dengan gradien halus
    img = Image.new("RGB", (ukuran, ukuran), warna_dasar)
    draw = ImageDraw.Draw(img)

    # Tambahkan gradien vertikal halus
    for y in range(ukuran):
        rasio = y / ukuran
        r = int(warna_dasar[0] * (1 - rasio * 0.2))
        g = int(warna_dasar[1] * (1 - rasio * 0.15))
        b = int(warna_dasar[2] * (1 - rasio * 0.1))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        draw.line([(0, y), (ukuran - 1, y)], fill=(r, g, b))

    # Tambahkan bentuk elips (simulasi wajah sederhana)
    cx, cy = ukuran // 2, ukuran // 2
    rx, ry = ukuran // 4, ukuran // 3
    warna_elips = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in warna_dasar)
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=warna_elips,
        outline=tuple(max(0, c - 30) for c in warna_elips),
    )

    # Tambahkan label "REAL" dan nomor
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    label = f"REAL #{indeks + 1:03d}"
    teks_warna = (50, 150, 50)  # Hijau untuk real
    draw.text((10, 10), label, fill=teks_warna, font=font)
    draw.text((10, ukuran - 25), "SAMPLE / PLACEHOLDER", fill=(100, 100, 100), font=font)

    return img


def buat_gambar_placeholder_fake(ukuran, indeks, seed):
    """
    Membuat gambar placeholder yang merepresentasikan gambar 'fake'.

    Gambar fake dibuat dengan artefak visual yang lebih jelas,
    termasuk distorsi warna, pola grid, dan noise yang lebih tinggi.

    Args:
        ukuran: Ukuran gambar (piksel).
        indeks: Nomor indeks gambar.
        seed: Seed untuk pengacakan.

    Returns:
        Objek PIL.Image.
    """
    random.seed(seed + indeks + 1000)

    # Warna dasar dengan sedikit distorsi
    warna_dasar = random.choice([
        (200, 170, 180),  # Tone pink-ish (distorsi warna)
        (170, 190, 160),  # Tone kehijauan
        (190, 180, 200),  # Tone ungu muda
        (210, 200, 170),  # Tone kuning pucat
    ])

    img = Image.new("RGB", (ukuran, ukuran), warna_dasar)
    draw = ImageDraw.Draw(img)

    # Tambahkan gradien dengan distorsi
    for y in range(ukuran):
        rasio = y / ukuran
        # Tambahkan osilasi untuk mensimulasikan artefak
        offset = int(10 * (0.5 + 0.5 * (((y * 7) % 50) / 50.0)))
        r = int(warna_dasar[0] * (1 - rasio * 0.3) + offset)
        g = int(warna_dasar[1] * (1 - rasio * 0.2))
        b = int(warna_dasar[2] * (1 - rasio * 0.25) - offset)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        draw.line([(0, y), (ukuran - 1, y)], fill=(r, g, b))

    # Tambahkan pola grid (artefak kompresi/generasi)
    jarak_grid = ukuran // 8
    for i in range(0, ukuran, jarak_grid):
        warna_grid = (
            max(0, warna_dasar[0] - 30),
            max(0, warna_dasar[1] - 30),
            max(0, warna_dasar[2] - 30),
        )
        draw.line([(i, 0), (i, ukuran - 1)], fill=warna_grid, width=1)
        draw.line([(0, i), (ukuran - 1, i)], fill=warna_grid, width=1)

    # Tambahkan elips dengan batas yang tidak halus (simulasi artefak deepfake)
    cx, cy = ukuran // 2, ukuran // 2
    rx, ry = ukuran // 4, ukuran // 3

    # Beberapa elips tumpang tindih untuk mensimulasikan blending yang buruk
    for offset_x, offset_y in [(0, 0), (3, -2), (-2, 3)]:
        warna_elips = tuple(
            max(0, min(255, c + random.randint(-40, 40))) for c in warna_dasar
        )
        draw.ellipse(
            [cx - rx + offset_x, cy - ry + offset_y,
             cx + rx + offset_x, cy + ry + offset_y],
            fill=warna_elips,
        )

    # Tambahkan noise berupa titik-titik acak
    for _ in range(ukuran * 2):
        x = random.randint(0, ukuran - 1)
        y = random.randint(0, ukuran - 1)
        warna_noise = tuple(random.randint(0, 255) for _ in range(3))
        draw.point((x, y), fill=warna_noise)

    # Tambahkan label "FAKE" dan nomor
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    label = f"FAKE #{indeks + 1:03d}"
    teks_warna = (200, 50, 50)  # Merah untuk fake
    draw.text((10, 10), label, fill=teks_warna, font=font)
    draw.text((10, ukuran - 25), "SAMPLE / PLACEHOLDER", fill=(100, 100, 100), font=font)

    return img


def main():
    """Fungsi utama untuk membuat data sampel."""
    args = parse_arguments()

    print("=" * 60)
    print("  DOWNLOAD & PEMBUATAN DATA SAMPEL")
    print("  Sistem Deteksi Deepfake")
    print("=" * 60)
    print()

    # Tampilkan informasi dataset
    print(INFORMASI_DATASET)

    if args.info_only:
        print("[INFO] Mode informasi saja. Tidak membuat gambar sampel.")
        return

    # Tetapkan seed
    random.seed(args.seed)

    # Buat direktori output
    dir_real = Path(args.output) / "real"
    dir_fake = Path(args.output) / "fake"
    dir_real.mkdir(parents=True, exist_ok=True)
    dir_fake.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    print(f"  Membuat {args.count} gambar placeholder per kelas...")
    print(f"  Ukuran gambar : {args.size}x{args.size} piksel")
    print(f"  Direktori     : {args.output}")
    print("-" * 60)
    print()

    # Buat gambar placeholder REAL
    print("  [1/2] Membuat gambar placeholder REAL...")
    for i in range(args.count):
        img = buat_gambar_placeholder_real(args.size, i, args.seed)
        nama_file = f"real_sample_{i + 1:03d}.png"
        path_simpan = dir_real / nama_file
        img.save(str(path_simpan))
        print(f"    ✓ Tersimpan: {path_simpan}")

    print()

    # Buat gambar placeholder FAKE
    print("  [2/2] Membuat gambar placeholder FAKE...")
    for i in range(args.count):
        img = buat_gambar_placeholder_fake(args.size, i, args.seed)
        nama_file = f"fake_sample_{i + 1:03d}.png"
        path_simpan = dir_fake / nama_file
        img.save(str(path_simpan))
        print(f"    ✓ Tersimpan: {path_simpan}")

    # Cetak ringkasan
    print()
    print("=" * 60)
    print("  RINGKASAN")
    print("=" * 60)
    print(f"  Gambar REAL dibuat  : {args.count}")
    print(f"  Gambar FAKE dibuat  : {args.count}")
    print(f"  Total gambar        : {args.count * 2}")
    print(f"  Lokasi              : {args.output}/")
    print()
    print("  Struktur direktori:")
    print(f"    {args.output}/")
    print(f"    ├── real/ ({args.count} gambar)")
    print(f"    └── fake/ ({args.count} gambar)")
    print()
    print("  [CATATAN]")
    print("  Gambar ini hanya placeholder untuk menguji pipeline.")
    print("  Untuk pelatihan yang sesungguhnya, gunakan dataset")
    print("  asli seperti FaceForensics++ atau DFDC.")
    print()
    print("  Untuk mempersiapkan dataset asli, jalankan:")
    print("    python scripts/prepare_dataset.py \\")
    print(f"        --source {args.output} --output data")
    print()
    print("  [SELESAI] Data sampel berhasil dibuat!")
    print("=" * 60)


if __name__ == "__main__":
    main()
