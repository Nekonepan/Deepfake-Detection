"""
Setup konfigurasi untuk paket Deepfake Detection.

Sistem Deteksi Deepfake pada Citra Digital
Menggunakan Metode Convolutional Neural Network (CNN)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Membaca file requirements.txt
this_directory = Path(__file__).parent
requirements_path = this_directory / "requirements.txt"

if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

# Membaca README untuk deskripsi panjang
readme_path = this_directory / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="deepfake-detection",
    version="1.0.0",
    description=(
        "Sistem Deteksi Deepfake pada Citra Digital "
        "Menggunakan Metode Convolutional Neural Network (CNN)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nekonepan",
    url="https://github.com/Nekonepan/Deepfake-Detection",
    packages=find_packages(exclude=["tests*", "scripts*", "configs*", "data*", "results*", "logs*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run_training=src.train:main",
            "run_predict=src.predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "deepfake",
        "detection",
        "cnn",
        "deep-learning",
        "image-classification",
        "computer-vision",
    ],
)
