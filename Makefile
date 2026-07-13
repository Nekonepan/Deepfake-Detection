# ============================================================
#  Makefile — Sistem Deteksi Deepfake (CNN)
# ============================================================
#  Penggunaan:
#    make help          Tampilkan semua target yang tersedia
#    make setup         Instal semua dependensi
#    make train         Jalankan training dengan konfigurasi default
#    make predict       Prediksi satu gambar
#    make app           Jalankan Streamlit web app
#    make all           Setup → Sample Data → Train → App
# ============================================================

# ── Shell & Flags ──────────────────────────────────────────────
SHELL  := /bin/bash
.DEFAULT_GOAL := help

# ── Python (otomatis pakai venv jika ada) ──────────────────────
VENV_DIR    := venv
PYTHON      := $(if $(wildcard $(VENV_DIR)/bin/python),$(VENV_DIR)/bin/python,python3)
PIP         := $(if $(wildcard $(VENV_DIR)/bin/pip),$(VENV_DIR)/bin/pip,pip3)
STREAMLIT   := $(if $(wildcard $(VENV_DIR)/bin/streamlit),$(VENV_DIR)/bin/streamlit,streamlit)

# ── Konfigurasi Default (bisa di-override dari CLI) ────────────
#   Contoh: make train EPOCHS=100 BATCH_SIZE=64 LR=0.0001
CONFIG      := configs/default_config.yaml
DATA_DIR    := data
EPOCHS      := 50
BATCH_SIZE  := 32
LR          := 0.001
SEED        := 42
SAVE_DIR    := models
RESULTS_DIR := results
LOG_DIR     := logs

# ── Prediksi ───────────────────────────────────────────────────
MODEL_PATH  := models/best_model.pth
IMAGE_PATH  ?=
SAVE_PATH   := results/prediction_result.png
DEVICE      := auto

# ── Sample Data ────────────────────────────────────────────────
SAMPLE_COUNT := 50
SAMPLE_DIR   := data/sample

# ── Prepare Dataset ───────────────────────────────────────────
RAW_DIR     ?= data/raw
TRAIN_RATIO := 0.7
VAL_RATIO   := 0.15
TEST_RATIO  := 0.15

# ── Streamlit ─────────────────────────────────────────────────
PORT        := 8501

# ══════════════════════════════════════════════════════════════
#  TARGETS
# ══════════════════════════════════════════════════════════════

.PHONY: help setup venv install train train-config train-custom \
        predict app data-sample data-prepare clean clean-all \
        dirs info all

# ── Help ──────────────────────────────────────────────────────
help: ## Tampilkan daftar semua target
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║       🔍 Sistem Deteksi Deepfake — Makefile                ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Contoh:"
	@echo "  make train                          Training default (50 epoch)"
	@echo "  make train EPOCHS=100 LR=0.0001     Training custom"
	@echo "  make predict IMAGE_PATH=foto.jpg    Prediksi satu gambar"
	@echo "  make app PORT=8080                  Streamlit di port 8080"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────
venv: ## Buat virtual environment baru
	python3 -m venv $(VENV_DIR)
	@echo "✅ Virtual environment dibuat di $(VENV_DIR)/"
	@echo "   Aktifkan: source $(VENV_DIR)/bin/activate"

install: ## Instal dependensi dari requirements.txt
	$(PIP) install -r requirements.txt
	@echo "✅ Semua dependensi berhasil diinstal"

setup: venv install ## Buat venv + instal dependensi (full setup)
	@echo "✅ Setup selesai! Aktifkan venv: source $(VENV_DIR)/bin/activate"

# ── Direktori ─────────────────────────────────────────────────
dirs: ## Buat direktori yang diperlukan (models, results, logs)
	@mkdir -p $(SAVE_DIR) $(RESULTS_DIR) $(LOG_DIR)
	@echo "✅ Direktori siap: $(SAVE_DIR)/ $(RESULTS_DIR)/ $(LOG_DIR)/"

# ── Training ──────────────────────────────────────────────────
train: dirs ## Training model dengan konfigurasi default
	$(PYTHON) run_training.py \
		--config $(CONFIG) \
		--data_dir $(DATA_DIR) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--lr $(LR) \
		--seed $(SEED) \
		--save_dir $(SAVE_DIR) \
		--results_dir $(RESULTS_DIR)

train-quick: dirs ## Training cepat (10 epoch, untuk testing pipeline)
	$(PYTHON) run_training.py \
		--config $(CONFIG) \
		--data_dir $(DATA_DIR) \
		--epochs 10 \
		--batch_size 16 \
		--lr 0.001 \
		--seed $(SEED) \
		--save_dir $(SAVE_DIR) \
		--results_dir $(RESULTS_DIR)

train-full: dirs ## Training penuh (100 epoch, batch 64, lr kecil)
	$(PYTHON) run_training.py \
		--config $(CONFIG) \
		--data_dir $(DATA_DIR) \
		--epochs 100 \
		--batch_size 64 \
		--lr 0.0001 \
		--seed $(SEED) \
		--save_dir $(SAVE_DIR) \
		--results_dir $(RESULTS_DIR)

# ── Prediksi ──────────────────────────────────────────────────
predict: ## Prediksi satu gambar (wajib: IMAGE_PATH=path/ke/gambar.jpg)
	@if [ -z "$(IMAGE_PATH)" ]; then \
		echo "❌ Error: IMAGE_PATH wajib diisi!"; \
		echo "   Contoh: make predict IMAGE_PATH=path/ke/gambar.jpg"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "❌ Error: Model tidak ditemukan di $(MODEL_PATH)"; \
		echo "   Jalankan 'make train' terlebih dahulu."; \
		exit 1; \
	fi
	$(PYTHON) run_predict.py \
		--image_path $(IMAGE_PATH) \
		--model_path $(MODEL_PATH) \
		--save_path $(SAVE_PATH) \
		--device $(DEVICE)

# ── Streamlit App ─────────────────────────────────────────────
app: ## Jalankan Streamlit web app
	$(STREAMLIT) run app.py --server.port $(PORT)

app-public: ## Jalankan Streamlit (akses dari jaringan luar)
	$(STREAMLIT) run app.py \
		--server.port $(PORT) \
		--server.address 0.0.0.0 \
		--server.headless true

# ── Data ──────────────────────────────────────────────────────
data-sample: ## Buat data sampel placeholder untuk testing
	$(PYTHON) scripts/download_sample_data.py \
		--output $(SAMPLE_DIR) \
		--count $(SAMPLE_COUNT)
	@echo "✅ Data sampel dibuat di $(SAMPLE_DIR)/"

data-prepare: ## Bagi dataset mentah ke train/val/test split
	@if [ -z "$(RAW_DIR)" ] || [ ! -d "$(RAW_DIR)" ]; then \
		echo "❌ Error: RAW_DIR tidak ditemukan: $(RAW_DIR)"; \
		echo "   Contoh: make data-prepare RAW_DIR=data/raw"; \
		exit 1; \
	fi
	$(PYTHON) scripts/prepare_dataset.py \
		--source $(RAW_DIR) \
		--output $(DATA_DIR) \
		--train-ratio $(TRAIN_RATIO) \
		--val-ratio $(VAL_RATIO) \
		--test-ratio $(TEST_RATIO)
	@echo "✅ Dataset berhasil di-split ke $(DATA_DIR)/{train,val,test}/"

# ── Info ──────────────────────────────────────────────────────
info: ## Tampilkan info project dan environment
	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo "  📋 Project Info"
	@echo "═══════════════════════════════════════════"
	@echo "  Python   : $$($(PYTHON) --version 2>&1)"
	@echo "  Pip      : $$($(PIP) --version 2>&1 | cut -d' ' -f1-2)"
	@echo "  PyTorch  : $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "  CUDA     : $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
	@echo "  Device   : $$($(PYTHON) -c 'import torch; print("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")' 2>/dev/null || echo 'N/A')"
	@echo ""
	@echo "  Data dir : $(DATA_DIR)/"
	@echo "  Train    : $$(find $(DATA_DIR)/train -type f 2>/dev/null | wc -l) files"
	@echo "  Val      : $$(find $(DATA_DIR)/val -type f 2>/dev/null | wc -l) files"
	@echo "  Test     : $$(find $(DATA_DIR)/test -type f 2>/dev/null | wc -l) files"
	@echo "  Model    : $(if $(wildcard $(MODEL_PATH)),✅ $(MODEL_PATH),❌ belum ada)"
	@echo "═══════════════════════════════════════════"
	@echo ""

# ── Clean ─────────────────────────────────────────────────────
clean: ## Hapus hasil training (models, results, logs, __pycache__)
	rm -rf $(RESULTS_DIR)/*.png $(RESULTS_DIR)/*.txt
	rm -rf $(LOG_DIR)/*.log
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ File hasil dan cache dibersihkan"

clean-all: clean ## Hapus semua (termasuk model dan venv)
	rm -rf $(SAVE_DIR)/*.pth
	rm -rf $(VENV_DIR)
	@echo "✅ Semua file dibersihkan (termasuk model dan venv)"

# ── Combo ─────────────────────────────────────────────────────
all: setup data-sample train app ## Full pipeline: setup → sample data → train → app
