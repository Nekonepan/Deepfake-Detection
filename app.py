"""
Sistem Deteksi Deepfake pada Citra Digital
Menggunakan Metode Convolutional Neural Network (CNN)

Aplikasi web berbasis Streamlit untuk mendeteksi gambar deepfake
menggunakan model CNN yang telah dilatih.
"""

import streamlit as st
import os
import random
import time
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ── Konfigurasi Halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Impor modul model (opsional, jika sudah ada) ──────────────────────────
MODEL_AVAILABLE = False
try:
    from src.model import DeepfakeCNN
    from src.predict import DeepfakePredictor

    MODEL_AVAILABLE = True
except ImportError:
    DeepfakeCNN = None
    DeepfakePredictor = None

MODEL_PATH = os.path.join("models", "best_model.pth")
RESULTS_DIR = "results"

# ── Warna & Tema ──────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "accent": "#FF6584",
    "bg": "#0F0F1A",
    "card": "#1A1A2E",
    "success": "#00D9A6",
    "danger": "#FF4757",
    "text": "#E0E0E0",
    "text_muted": "#8888AA",
}

# ══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS – Glassmorphism Dark Theme
# ══════════════════════════════════════════════════════════════════════════


def inject_custom_css():
    """Menyuntikkan CSS kustom untuk tampilan premium."""
    st.markdown(
        """
    <style>
    /* ── Import Font ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root & Body ─────────────────────────────── */
    :root {
        --primary: #6C63FF;
        --accent: #FF6584;
        --bg: #0F0F1A;
        --card: #1A1A2E;
        --success: #00D9A6;
        --danger: #FF4757;
        --text: #E0E0E0;
        --muted: #8888AA;
    }

    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #16162A 40%, #1A1A2E 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }


    /* ── Animated background ─────────────────────── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 600px 600px at 20% 20%, rgba(108,99,255,0.06) 0%, transparent 70%),
            radial-gradient(ellipse 500px 500px at 80% 80%, rgba(255,101,132,0.05) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        animation: bgShift 20s ease-in-out infinite alternate;
    }
    @keyframes bgShift {
        0%   { opacity: .6; }
        50%  { opacity: 1; }
        100% { opacity: .6; }
    }

    /* ── Scrollbar ────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary), var(--accent));
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122280, #1A1A2E80) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(108,99,255,0.15) !important;
    }
    /* ── Force Sidebar Always Visible & Hide Collapse Buttons ── */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        transform: none !important;
        visibility: visible !important;
        display: block !important;
    }
    /* Ensure main page content aligns properly alongside the forced sidebar */
    .stAppViewContainer {
        display: flex !important;
        flex-direction: row !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: var(--text) !important;
    }

    /* ── Glassmorphism Card ───────────────────────── */
    .glass-card {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(108,99,255,0.12);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        transition: all 0.35s cubic-bezier(.25,.8,.25,1);
    }
    .glass-card:hover {
        border-color: rgba(108,99,255,0.35);
        box-shadow: 0 8px 32px rgba(108,99,255,0.12);
        transform: translateY(-3px);
    }

    /* ── Gradient Text ───────────────────────────── */
    .gradient-text {
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    .gradient-text-teal {
        background: linear-gradient(135deg, #00D9A6, #6C63FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* ── Hero Section ────────────────────────────── */
    .hero-section {
        text-align: center;
        padding: 50px 20px 40px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1.15;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 50%, #6C63FF 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s linear infinite;
    }
    @keyframes shimmer {
        to { background-position: 200% center; }
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--muted);
        max-width: 720px;
        margin: 0 auto 30px;
        line-height: 1.7;
    }

    /* ── Feature Card ────────────────────────────── */
    .feature-card {
        background: rgba(26, 26, 46, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108,99,255,0.1);
        border-radius: 16px;
        padding: 30px 24px;
        text-align: center;
        transition: all 0.35s cubic-bezier(.25,.8,.25,1);
        height: 100%;
    }
    .feature-card:hover {
        border-color: rgba(108,99,255,0.35);
        box-shadow: 0 12px 40px rgba(108,99,255,0.15);
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2.6rem;
        margin-bottom: 14px;
        display: block;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: var(--muted);
        line-height: 1.6;
    }

    /* ── Result Card ─────────────────────────────── */
    .result-card {
        background: rgba(26, 26, 46, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        border: 2px solid;
        transition: all 0.4s ease;
    }
    .result-real {
        border-color: rgba(0,217,166,0.4);
        box-shadow: 0 0 40px rgba(0,217,166,0.1);
    }
    .result-fake {
        border-color: rgba(255,71,87,0.4);
        box-shadow: 0 0 40px rgba(255,71,87,0.1);
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 10px 0 6px;
        letter-spacing: 2px;
    }
    .result-confidence {
        font-size: 1rem;
        color: var(--muted);
        margin-bottom: 18px;
    }

    /* ── Confidence Bar ──────────────────────────── */
    .confidence-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        height: 14px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 1.2s cubic-bezier(.25,.8,.25,1);
    }
    .bar-real { background: linear-gradient(90deg, #00D9A6, #6C63FF); }
    .bar-fake { background: linear-gradient(90deg, #FF4757, #FF6584); }

    /* ── Stat Card ────────────────────────────────── */
    .stat-card {
        background: rgba(26, 26, 46, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108,99,255,0.1);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: rgba(108,99,255,0.3);
        transform: translateY(-2px);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #00D9A6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 4px;
    }

    /* ── Step Card ────────────────────────────────── */
    .step-card {
        background: rgba(26, 26, 46, 0.45);
        border: 1px solid rgba(108,99,255,0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .step-card:hover {
        border-color: rgba(108,99,255,0.25);
    }
    .step-number {
        display: inline-block;
        width: 36px; height: 36px;
        line-height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        color: #fff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 10px;
    }

    /* ── Architecture Diagram ────────────────────── */
    .arch-box {
        background: rgba(108,99,255,0.08);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--text);
        margin: 4px 0;
    }
    .arch-arrow {
        text-align: center;
        color: var(--primary);
        font-size: 1.3rem;
        line-height: 1;
        padding: 2px 0;
    }

    /* ── Info Badge ───────────────────────────────── */
    .info-badge {
        display: inline-block;
        background: rgba(108,99,255,0.12);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 8px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: var(--primary);
        font-weight: 600;
        margin: 2px 4px;
    }
    .warning-badge {
        display: inline-block;
        background: rgba(255,165,0,0.1);
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 0.85rem;
        color: #FFA500;
        font-weight: 500;
    }

    /* ── Table Styling ───────────────────────────── */
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        font-size: 0.9rem;
    }
    .styled-table th {
        background: rgba(108,99,255,0.12);
        color: var(--primary);
        padding: 12px 18px;
        text-align: left;
        font-weight: 600;
        border-bottom: 1px solid rgba(108,99,255,0.15);
    }
    .styled-table td {
        padding: 10px 18px;
        color: var(--text);
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .styled-table tr:hover td {
        background: rgba(108,99,255,0.04);
    }

    /* ── Buttons ──────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #8B7DFF) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1) !important;
        box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108,99,255,0.45) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── File Uploader ───────────────────────────── */
    section[data-testid="stFileUploader"] {
        background: rgba(26,26,46,0.4);
        border: 2px dashed rgba(108,99,255,0.25);
        border-radius: 16px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: rgba(108,99,255,0.5);
    }

    /* ── Progress Bar ────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6C63FF, #FF6584) !important;
        border-radius: 8px !important;
    }

    /* ── Footer ───────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 40px 20px 20px;
        color: var(--muted);
        font-size: 0.82rem;
        border-top: 1px solid rgba(108,99,255,0.08);
        margin-top: 60px;
    }
    .footer a {
        color: var(--primary);
        text-decoration: none;
    }

    /* ── Divider ──────────────────────────────────── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108,99,255,0.3), transparent);
        margin: 30px 0;
    }

    /* ── Demo Badge ──────────────────────────────── */
    .demo-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(255,165,0,0.12), rgba(255,101,132,0.12));
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 0.8rem;
        color: #FFA500;
        font-weight: 600;
        letter-spacing: 0.5px;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* ── Hide Streamlit defaults ─────────────────── */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Markdown inside app ──────────────────────── */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #FFFFFF !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: var(--text) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════


def render_glass_card(content_html: str) -> None:
    """Menampilkan kartu glassmorphism."""
    st.markdown(
        f'<div class="glass-card">{content_html}</div>', unsafe_allow_html=True
    )


def render_feature_card(icon: str, title: str, description: str) -> None:
    """Menampilkan kartu fitur."""
    st.markdown(
        f"""
        <div class="feature-card">
            <span class="feature-icon">{icon}</span>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{description}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_stat_card(value: str, label: str) -> None:
    """Menampilkan kartu statistik."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_divider() -> None:
    """Menampilkan garis pembatas bergradien."""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Menampilkan footer aplikasi."""
    st.markdown(
        """
        <div class="footer">
            <p>🔍 <strong>Sistem Deteksi Deepfake</strong> — Proyek Akademik</p>
            <p>Menggunakan Metode Convolutional Neural Network (CNN)</p>
            <p style="margin-top:8px; font-size:0.75rem;">
                Dibuat dengan ❤️ menggunakan Streamlit &amp; PyTorch
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def check_model_exists() -> bool:
    """Memeriksa apakah file model terlatih tersedia."""
    return os.path.exists(MODEL_PATH)


def check_results_exist() -> dict:
    """Memeriksa file hasil pelatihan yang tersedia."""
    found = {}
    if os.path.isdir(RESULTS_DIR):
        for fname in os.listdir(RESULTS_DIR):
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.isfile(fpath):
                found[fname] = fpath
    return found


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════


def render_sidebar() -> str:
    """Menampilkan sidebar navigasi dan mengembalikan halaman yang dipilih."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 20px 0 10px;">
                <span style="font-size: 2.5rem;">🔍</span>
                <h2 style="margin:8px 0 2px;">
                    <span class="gradient-text">Deepfake</span>
                    <span style="color:#FFFFFF; font-weight:300;"> Detector</span>
                </h2>
                <p style="color: var(--muted); font-size: 0.8rem; margin:0;">
                    Deteksi Citra Digital v1.0
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        render_divider()

        page = st.radio(
            "Navigasi",
            ["🏠 Beranda", "🔍 Deteksi", "📊 Evaluasi Model", "📖 Tentang"],
            label_visibility="collapsed",
        )

        render_divider()

        # Status sistem
        model_exists = check_model_exists()
        status_color = COLORS["success"] if model_exists else COLORS["danger"]
        status_text = "Siap" if model_exists else "Belum Dilatih"
        status_icon = "✅" if model_exists else "⚠️"

        st.markdown(
            f"""
            <div class="glass-card" style="padding:16px;">
                <p style="font-size:0.8rem; color:var(--muted); margin:0 0 8px;">
                    STATUS MODEL
                </p>
                <p style="margin:0; font-size:0.95rem;">
                    {status_icon}
                    <span style="color:{status_color}; font-weight:600;">
                        {status_text}
                    </span>
                </p>
                <p style="font-size:0.75rem; color:var(--muted); margin:6px 0 0;">
                    {"Model terlatih ditemukan" if model_exists else "Mode simulasi aktif"}
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="text-align:center; padding:10px 0; font-size:0.72rem; color:var(--muted);">
                <p>© 2025 Deepfake Detection</p>
                <p>Proyek Tugas Akhir</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    return page


# ══════════════════════════════════════════════════════════════════════════
#  HALAMAN 1 – BERANDA
# ══════════════════════════════════════════════════════════════════════════


def page_beranda() -> None:
    """Menampilkan halaman beranda."""
    # Hero Section
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-title">
                Sistem Deteksi Deepfake<br>pada Citra Digital
            </div>
            <div class="hero-subtitle">
                Menggunakan Metode <strong style="color:#6C63FF;">
                Convolutional Neural Network (CNN)</strong> untuk menganalisis
                dan mengidentifikasi keaslian gambar wajah secara otomatis
                dengan tingkat akurasi tinggi.
            </div>
            <div>
                <span class="info-badge">🧠 Deep Learning</span>
                <span class="info-badge">🖼️ Analisis Citra</span>
                <span class="info-badge">🔒 Keamanan Digital</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Fitur Utama
    st.markdown(
        '<h2 style="text-align:center; margin-bottom:8px;">'
        '<span class="gradient-text">Fitur Utama</span></h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; color:var(--muted); margin-bottom:28px;">'
        "Sistem dilengkapi dengan berbagai fitur untuk deteksi deepfake yang akurat</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_feature_card(
            "🔍",
            "Deteksi Real-Time",
            "Unggah gambar dan dapatkan hasil prediksi secara instan menggunakan model CNN.",
        )
    with col2:
        render_feature_card(
            "🧠",
            "Arsitektur CNN",
            "Model deep learning yang dirancang khusus untuk mendeteksi manipulasi wajah.",
        )
    with col3:
        render_feature_card(
            "📊",
            "Evaluasi Komprehensif",
            "Analisis performa model lengkap dengan metrik akurasi, precision, dan recall.",
        )
    with col4:
        render_feature_card(
            "🔥",
            "Grad-CAM Visualisasi",
            "Visualisasi area gambar yang menjadi fokus perhatian model saat deteksi.",
        )

    render_divider()

    # Penjelasan Deepfake
    col_left, col_right = st.columns([3, 2])

    with col_left:
        render_glass_card(
            """
            <h3 style="margin-top:0;">
                <span class="gradient-text">Apa itu Deepfake?</span>
            </h3>
            <p style="color: var(--text); line-height: 1.8; font-size: 0.95rem;">
                <strong>Deepfake</strong> adalah teknologi berbasis kecerdasan buatan
                yang mampu membuat konten visual palsu dengan tingkat realisme tinggi.
                Teknologi ini menggunakan teknik <em>deep learning</em>, khususnya
                <em>Generative Adversarial Networks</em> (GANs), untuk memanipulasi
                atau mensintesis gambar dan video wajah seseorang.
            </p>
            <p style="color: var(--text); line-height: 1.8; font-size: 0.95rem;">
                Dengan meningkatnya penyebaran konten deepfake, diperlukan sistem
                deteksi yang dapat mengidentifikasi manipulasi secara otomatis.
                Sistem ini menggunakan <strong>Convolutional Neural Network (CNN)</strong>
                untuk menganalisis pola-pola halus pada citra digital yang tidak
                terlihat oleh mata manusia.
            </p>
        """
        )

    with col_right:
        render_glass_card(
            """
            <h3 style="margin-top:0;">
                <span class="gradient-text-teal">Mengapa Deteksi Penting?</span>
            </h3>
            <ul style="color: var(--text); line-height: 2.2; font-size: 0.92rem;
                        list-style: none; padding-left: 0;">
                <li>🛡️ Mencegah penyebaran informasi palsu</li>
                <li>⚖️ Melindungi integritas bukti digital</li>
                <li>🔒 Menjaga privasi dan identitas individu</li>
                <li>📰 Memerangi hoax dan disinformasi</li>
                <li>🏛️ Mendukung keamanan siber nasional</li>
                <li>🎓 Mendorong riset kecerdasan buatan</li>
            </ul>
        """
        )

    render_divider()

    # Arsitektur Sistem
    st.markdown(
        '<h2 style="text-align:center; margin-bottom:20px;">'
        '<span class="gradient-text">Arsitektur Sistem</span></h2>',
        unsafe_allow_html=True,
    )

    arch_col1, arch_col2, arch_col3 = st.columns([1, 3, 1])
    with arch_col2:
        st.markdown(
            """
            <div class="glass-card" style="padding:30px 20px;">
                <div class="arch-box">📷 Input Citra Digital (JPG / PNG)</div>
                <div class="arch-arrow">⬇</div>
                <div class="arch-box">🔄 Preprocessing (Resize 224×224, Normalisasi)</div>
                <div class="arch-arrow">⬇</div>
                <div class="arch-box">🧠 Feature Extraction — Conv2D + ReLU + MaxPool (×3 Block)</div>
                <div class="arch-arrow">⬇</div>
                <div class="arch-box">📐 Flatten + Fully Connected Layers</div>
                <div class="arch-arrow">⬇</div>
                <div class="arch-box">🎯 Sigmoid Output → Probabilitas (Real / Fake)</div>
                <div class="arch-arrow">⬇</div>
                <div class="arch-box" style="border-color: rgba(0,217,166,0.35);
                    background: rgba(0,217,166,0.06);">
                    ✅ Hasil Prediksi + Skor Kepercayaan
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Statistik
    render_divider()
    st.markdown(
        '<h2 style="text-align:center; margin-bottom:20px;">'
        '<span class="gradient-text">Ringkasan Sistem</span></h2>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_stat_card("CNN", "Arsitektur Model")
    with s2:
        render_stat_card("224²", "Ukuran Input")
    with s3:
        render_stat_card("2", "Kelas Output")
    with s4:
        render_stat_card("PyTorch", "Framework")


# ══════════════════════════════════════════════════════════════════════════
#  HALAMAN 2 – DETEKSI
# ══════════════════════════════════════════════════════════════════════════


def simulate_prediction(image: Image.Image) -> dict:
    """Simulasi prediksi ketika model belum tersedia."""
    prediction = random.choice(["REAL", "FAKE"])
    confidence = random.uniform(0.65, 0.98)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "is_simulation": True,
    }


def real_prediction(image: Image.Image) -> dict:
    """Prediksi menggunakan model terlatih."""
    temp_path = None
    grad_cam_path = None
    try:
        predictor = DeepfakePredictor(MODEL_PATH)
        
        # Save PIL Image to a temporary file because predict_with_gradcam expects a path
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_upload.png")
        image.save(temp_path)
        
        # Path for saving Grad-CAM heatmap visualization
        grad_cam_path = os.path.join(temp_dir, "temp_grad_cam.png")
        
        # Run prediction with Grad-CAM
        label, confidence, _ = predictor.predict_with_gradcam(
            image_path=temp_path,
            save_path=grad_cam_path
        )
        
        # Load the generated Grad-CAM visualization image fully into memory
        grad_cam_img = None
        if os.path.exists(grad_cam_path):
            with Image.open(grad_cam_path) as img:
                grad_cam_img = img.copy()
            
        return {
            "prediction": label.upper(),  # Convert to uppercase (REAL/FAKE)
            "confidence": confidence / 100.0,  # Convert percentage (0-100) to decimal (0-1)
            "is_simulation": False,
            "grad_cam": grad_cam_img,
        }
    except Exception as e:
        st.warning(f"Gagal memuat model atau memproses gambar: {e}")
        return simulate_prediction(image)
    finally:
        # Clean up temp files
        for path in (temp_path, grad_cam_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


def page_deteksi() -> None:
    """Menampilkan halaman deteksi deepfake."""
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:10px;">
            <h1><span class="gradient-text">🔍 Deteksi Deepfake</span></h1>
            <p style="color:var(--muted); max-width:600px; margin:0 auto;">
                Unggah gambar wajah untuk menganalisis keasliannya menggunakan
                model Convolutional Neural Network
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Mode indicator
    model_exists = check_model_exists()
    if not model_exists:
        st.markdown(
            '<div style="text-align:center; margin: 10px 0 20px;">'
            '<span class="demo-badge">🎮 MODE SIMULASI — Model belum dilatih</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    render_divider()

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown(
            '<h3><span class="gradient-text-teal">Unggah Gambar</span></h3>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Pilih file gambar untuk dianalisis",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown(
                '<p style="color:var(--muted); font-size:0.85rem; margin-top:14px;">'
                "📎 Gambar yang diunggah:</p>",
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)

            # Info gambar
            w, h = image.size
            file_size_kb = uploaded_file.size / 1024
            st.markdown(
                f"""
                <div class="glass-card" style="padding:14px 20px; margin-top:10px;">
                    <table style="width:100%; font-size:0.85rem;">
                        <tr>
                            <td style="color:var(--muted);">Nama File</td>
                            <td style="color:var(--text); text-align:right;">
                                {uploaded_file.name}</td>
                        </tr>
                        <tr>
                            <td style="color:var(--muted);">Dimensi</td>
                            <td style="color:var(--text); text-align:right;">
                                {w} × {h} piksel</td>
                        </tr>
                        <tr>
                            <td style="color:var(--muted);">Ukuran</td>
                            <td style="color:var(--text); text-align:right;">
                                {file_size_kb:.1f} KB</td>
                        </tr>
                    </table>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Preprocessing preview
            render_divider()
            st.markdown(
                '<h3><span class="gradient-text-teal">'
                "Langkah Preprocessing</span></h3>",
                unsafe_allow_html=True,
            )

            p1, p2, p3 = st.columns(3)
            with p1:
                resized = image.resize((224, 224))
                st.image(resized, caption="1. Resize (224×224)", use_container_width=True)
            with p2:
                gray = resized.convert("L").convert("RGB")
                st.image(gray, caption="2. Grayscale Preview", use_container_width=True)
            with p3:
                enhanced = ImageEnhance.Contrast(resized).enhance(1.5)
                st.image(
                    enhanced,
                    caption="3. Peningkatan Kontras",
                    use_container_width=True,
                )

    with col_result:
        st.markdown(
            '<h3><span class="gradient-text-teal">Hasil Analisis</span></h3>',
            unsafe_allow_html=True,
        )

        if uploaded_file is not None:
            analyze_btn = st.button(
                "🔍  Analisis Gambar",
                use_container_width=True,
                type="primary",
            )

            if analyze_btn:
                # Animasi progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                steps = [
                    ("Memuat gambar...", 25),
                    ("Preprocessing...", 50),
                    ("Menganalisis dengan CNN...", 80),
                    ("Menghasilkan prediksi...", 100),
                ]

                for step_label, step_pct in steps:
                    status_text.markdown(
                        f'<p style="color:var(--primary); font-size:0.9rem;">'
                        f"⏳ {step_label}</p>",
                        unsafe_allow_html=True,
                    )
                    # Animasi bertahap
                    current = progress_bar.empty() if False else None
                    time.sleep(0.5 + random.uniform(0.1, 0.4))
                    progress_bar.progress(step_pct)

                status_text.empty()
                progress_bar.empty()

                # Prediksi
                image = Image.open(uploaded_file).convert("RGB")
                if model_exists and MODEL_AVAILABLE:
                    result = real_prediction(image)
                else:
                    result = simulate_prediction(image)

                prediction = result["prediction"]
                confidence = result["confidence"]
                is_fake = prediction == "FAKE"

                # Simpan di session state
                st.session_state["last_result"] = result

                # Tampilkan hasil
                card_class = "result-fake" if is_fake else "result-real"
                label_color = COLORS["danger"] if is_fake else COLORS["success"]
                bar_class = "bar-fake" if is_fake else "bar-real"
                icon = "⚠️" if is_fake else "✅"

                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <span style="font-size:3rem;">{icon}</span>
                        <div class="result-label" style="color:{label_color};">
                            {prediction}
                        </div>
                        <div class="result-confidence">
                            Tingkat Kepercayaan: {confidence*100:.1f}%
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill {bar_class}"
                                 style="width:{confidence*100:.1f}%;"></div>
                        </div>
                        <p style="color:var(--muted); font-size:0.82rem; margin-top:14px;">
                            {"Gambar terdeteksi sebagai hasil manipulasi (deepfake)"
                             if is_fake else
                             "Gambar terdeteksi sebagai asli (bukan deepfake)"}
                        </p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                # Peringatan mode simulasi
                if result.get("is_simulation"):
                    st.markdown(
                        """
                        <div class="warning-badge" style="margin-top:16px;
                            display:block; text-align:center;">
                            ⚠️ Hasil ini adalah <strong>simulasi</strong>.
                            Latih model terlebih dahulu untuk prediksi akurat.
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Grad-CAM placeholder
                render_divider()
                st.markdown(
                    '<h4><span class="gradient-text">Grad-CAM Heatmap</span></h4>',
                    unsafe_allow_html=True,
                )

                grad_cam_data = result.get("grad_cam")
                if grad_cam_data is not None:
                    st.image(
                        grad_cam_data,
                        caption="Grad-CAM — Area Fokus Model",
                        use_container_width=True,
                    )
                else:
                    # Placeholder heatmap simulation
                    img_resized = image.resize((224, 224))
                    blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=8))
                    heatmap_sim = Image.blend(
                        img_resized,
                        blurred,
                        alpha=0.4,
                    )
                    st.image(
                        heatmap_sim,
                        caption="Simulasi Heatmap (Grad-CAM belum tersedia)",
                        use_container_width=True,
                    )
                    st.markdown(
                        '<p style="color:var(--muted); font-size:0.8rem; text-align:center;">'
                        "Grad-CAM akan tersedia setelah model dilatih.</p>",
                        unsafe_allow_html=True,
                    )

            elif "last_result" in st.session_state:
                result = st.session_state["last_result"]
                prediction = result["prediction"]
                confidence = result["confidence"]
                is_fake = prediction == "FAKE"
                card_class = "result-fake" if is_fake else "result-real"
                label_color = COLORS["danger"] if is_fake else COLORS["success"]
                bar_class = "bar-fake" if is_fake else "bar-real"
                icon = "⚠️" if is_fake else "✅"

                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <span style="font-size:3rem;">{icon}</span>
                        <div class="result-label" style="color:{label_color};">
                            {prediction}
                        </div>
                        <div class="result-confidence">
                            Tingkat Kepercayaan: {confidence*100:.1f}%
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill {bar_class}"
                                 style="width:{confidence*100:.1f}%;"></div>
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                render_glass_card(
                    """
                    <div style="text-align:center; padding:20px;">
                        <span style="font-size:2.5rem;">🖱️</span>
                        <p style="color:var(--muted); margin-top:12px;">
                            Klik tombol <strong>"Analisis Gambar"</strong>
                            untuk memulai deteksi
                        </p>
                    </div>
                """
                )
        else:
            render_glass_card(
                """
                <div style="text-align:center; padding:40px 20px;">
                    <span style="font-size:3.5rem;">📤</span>
                    <h3 style="color:var(--text); margin:16px 0 8px;">
                        Belum Ada Gambar
                    </h3>
                    <p style="color:var(--muted);">
                        Unggah gambar melalui panel di sebelah kiri untuk
                        memulai analisis deepfake.
                    </p>
                    <div style="margin-top:16px;">
                        <span class="info-badge">JPG</span>
                        <span class="info-badge">JPEG</span>
                        <span class="info-badge">PNG</span>
                    </div>
                </div>
            """
            )


# ══════════════════════════════════════════════════════════════════════════
#  HALAMAN 3 – EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════


def page_evaluasi() -> None:
    """Menampilkan halaman evaluasi model."""
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:10px;">
            <h1><span class="gradient-text">📊 Evaluasi Model</span></h1>
            <p style="color:var(--muted); max-width:600px; margin:0 auto;">
                Analisis performa dan arsitektur model Convolutional Neural Network
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    render_divider()

    # Arsitektur Model
    st.markdown(
        '<h2><span class="gradient-text-teal">Arsitektur Model CNN</span></h2>',
        unsafe_allow_html=True,
    )

    col_arch, col_params = st.columns([3, 2], gap="large")

    with col_arch:
        render_glass_card(
            """
            <h4 style="margin-top:0; color:#FFFFFF;">📐 Struktur Layer</h4>
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Layer</th>
                        <th>Tipe</th>
                        <th>Output Shape</th>
                        <th>Parameter</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>Conv2D + ReLU</td>
                        <td>[B, 32, 224, 224]</td>
                        <td>896</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>MaxPool2D</td>
                        <td>[B, 32, 112, 112]</td>
                        <td>0</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>Conv2D + ReLU</td>
                        <td>[B, 64, 112, 112]</td>
                        <td>18.496</td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>MaxPool2D</td>
                        <td>[B, 64, 56, 56]</td>
                        <td>0</td>
                    </tr>
                    <tr>
                        <td>5</td>
                        <td>Conv2D + ReLU</td>
                        <td>[B, 128, 56, 56]</td>
                        <td>73.856</td>
                    </tr>
                    <tr>
                        <td>6</td>
                        <td>MaxPool2D</td>
                        <td>[B, 128, 28, 28]</td>
                        <td>0</td>
                    </tr>
                    <tr>
                        <td>7</td>
                        <td>Flatten</td>
                        <td>[B, 100.352]</td>
                        <td>0</td>
                    </tr>
                    <tr>
                        <td>8</td>
                        <td>Linear + ReLU</td>
                        <td>[B, 512]</td>
                        <td>51.380.736</td>
                    </tr>
                    <tr>
                        <td>9</td>
                        <td>Dropout (0.5)</td>
                        <td>[B, 512]</td>
                        <td>0</td>
                    </tr>
                    <tr>
                        <td>10</td>
                        <td>Linear + Sigmoid</td>
                        <td>[B, 1]</td>
                        <td>513</td>
                    </tr>
                </tbody>
            </table>
        """
        )

    with col_params:
        render_stat_card("~51.5M", "Total Parameter")
        st.markdown("<br>", unsafe_allow_html=True)
        render_stat_card("3", "Blok Konvolusi")
        st.markdown("<br>", unsafe_allow_html=True)
        render_stat_card("2", "Fully Connected")

        st.markdown("<br>", unsafe_allow_html=True)
        render_glass_card(
            """
            <h4 style="margin-top:0; color:#FFFFFF;">⚙️ Hyperparameter</h4>
            <table class="styled-table">
                <tr><th>Parameter</th><th>Nilai</th></tr>
                <tr><td>Learning Rate</td><td>0.001</td></tr>
                <tr><td>Batch Size</td><td>32</td></tr>
                <tr><td>Epochs</td><td>50</td></tr>
                <tr><td>Optimizer</td><td>Adam</td></tr>
                <tr><td>Loss Function</td><td>BCELoss</td></tr>
                <tr><td>Dropout Rate</td><td>0.5</td></tr>
                <tr><td>Input Size</td><td>224 × 224</td></tr>
            </table>
        """
        )

    render_divider()

    # Hasil Pelatihan
    st.markdown(
        '<h2><span class="gradient-text-teal">Hasil Pelatihan</span></h2>',
        unsafe_allow_html=True,
    )

    results = check_results_exist()

    if results:
        # Cari gambar grafik yang tersedia
        image_extensions = {".png", ".jpg", ".jpeg"}
        result_images = {
            k: v
            for k, v in results.items()
            if os.path.splitext(k)[1].lower() in image_extensions
        }

        if result_images:
            # Kelompokkan dan tampilkan
            col_a, col_b = st.columns(2)

            image_list = list(result_images.items())
            for idx, (fname, fpath) in enumerate(image_list):
                target_col = col_a if idx % 2 == 0 else col_b
                with target_col:
                    label = fname.replace("_", " ").replace(".png", "").title()
                    st.markdown(
                        f'<h4 style="color:var(--text);">📈 {label}</h4>',
                        unsafe_allow_html=True,
                    )
                    st.image(fpath, use_container_width=True)
        else:
            render_glass_card(
                """
                <div style="text-align:center; padding:30px;">
                    <span style="font-size:2.5rem;">📁</span>
                    <p style="color:var(--muted); margin-top:12px;">
                        Ditemukan file hasil pelatihan, namun tidak ada
                        grafik/gambar yang tersedia.
                    </p>
                </div>
            """
            )

        # Tampilkan daftar file
        render_divider()
        st.markdown(
            '<h4 style="color:var(--text);">📂 File Hasil Pelatihan</h4>',
            unsafe_allow_html=True,
        )

        table_rows = ""
        for fname, fpath in results.items():
            fsize = os.path.getsize(fpath) / 1024
            fext = os.path.splitext(fname)[1]
            table_rows += f"""
                <tr>
                    <td>{fname}</td>
                    <td><span class="info-badge">{fext}</span></td>
                    <td>{fsize:.1f} KB</td>
                </tr>
            """

        render_glass_card(
            f"""
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Nama File</th>
                        <th>Tipe</th>
                        <th>Ukuran</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
        """
        )

    else:
        render_glass_card(
            """
            <div style="text-align:center; padding:50px 20px;">
                <span style="font-size:3.5rem;">🔬</span>
                <h3 style="color:var(--text); margin:16px 0 8px;">
                    Belum Ada Hasil Pelatihan
                </h3>
                <p style="color:var(--muted); max-width:500px; margin:0 auto;">
                    Hasil evaluasi model (confusion matrix, ROC curve, grafik
                    akurasi/loss) akan ditampilkan setelah proses pelatihan selesai.
                </p>
                <div style="margin-top:20px;">
                    <span class="warning-badge">
                        ⚠️ Latih model terlebih dahulu dengan menjalankan
                        script pelatihan
                    </span>
                </div>
            </div>
        """
        )

    render_divider()

    # Metrik yang akan ditampilkan
    st.markdown(
        '<h2><span class="gradient-text-teal">Metrik Evaluasi</span></h2>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_feature_card(
            "🎯",
            "Akurasi",
            "Persentase prediksi benar dari seluruh data uji.",
        )
    with m2:
        render_feature_card(
            "📏",
            "Precision",
            "Ketepatan prediksi positif terhadap seluruh prediksi positif.",
        )
    with m3:
        render_feature_card(
            "🔍",
            "Recall",
            "Kemampuan model menemukan seluruh data positif aktual.",
        )
    with m4:
        render_feature_card(
            "⚖️",
            "F1-Score",
            "Rata-rata harmonik dari precision dan recall.",
        )


# ══════════════════════════════════════════════════════════════════════════
#  HALAMAN 4 – TENTANG
# ══════════════════════════════════════════════════════════════════════════


def page_tentang() -> None:
    """Menampilkan halaman tentang proyek."""
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:10px;">
            <h1><span class="gradient-text">📖 Tentang Proyek</span></h1>
            <p style="color:var(--muted); max-width:600px; margin:0 auto;">
                Informasi lengkap mengenai proyek deteksi deepfake ini
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    render_divider()

    # Deskripsi Proyek
    render_glass_card(
        """
        <h3 style="margin-top:0;">
            <span class="gradient-text">Deskripsi Proyek</span>
        </h3>
        <p style="color: var(--text); line-height: 1.8; font-size: 0.95rem;">
            Proyek ini berjudul <strong>"Sistem Deteksi Deepfake pada Citra Digital
            Menggunakan Metode Convolutional Neural Network (CNN)"</strong>.
            Penelitian ini bertujuan untuk membangun sistem yang mampu membedakan
            antara citra wajah asli dan citra wajah hasil manipulasi deepfake
            secara otomatis.
        </p>
        <p style="color: var(--text); line-height: 1.8; font-size: 0.95rem;">
            Sistem ini memanfaatkan arsitektur <strong>Convolutional Neural Network
            (CNN)</strong> yang diimplementasikan menggunakan framework
            <strong>PyTorch</strong>. Model dilatih menggunakan dataset citra wajah
            yang terdiri dari gambar asli (real) dan gambar palsu (fake) untuk
            mempelajari pola-pola halus yang membedakan keduanya.
        </p>
    """
    )

    # Metodologi CNN
    st.markdown(
        '<h2><span class="gradient-text-teal">Metodologi CNN</span></h2>',
        unsafe_allow_html=True,
    )

    col_method_l, col_method_r = st.columns(2, gap="large")

    with col_method_l:
        render_glass_card(
            """
            <h4 style="margin-top:0; color:#FFFFFF;">
                🧠 Convolutional Neural Network
            </h4>
            <p style="color:var(--text); line-height:1.8; font-size:0.92rem;">
                CNN adalah arsitektur deep learning yang dirancang khusus untuk
                memproses data yang memiliki topologi grid, seperti citra digital.
                CNN bekerja dengan mengekstrak fitur secara hierarkis melalui
                operasi konvolusi.
            </p>
            <p style="color:var(--text); line-height:1.8; font-size:0.92rem;">
                <strong>Komponen utama CNN:</strong>
            </p>
            <ul style="color:var(--text); line-height:2; font-size:0.92rem;">
                <li><strong>Convolutional Layer</strong> — Mengekstrak fitur
                    lokal dari citra menggunakan filter/kernel</li>
                <li><strong>Activation (ReLU)</strong> — Memperkenalkan
                    non-linearitas pada model</li>
                <li><strong>Pooling Layer</strong> — Mengurangi dimensi
                    spasial fitur untuk efisiensi</li>
                <li><strong>Fully Connected Layer</strong> — Mengklasifikasikan
                    fitur yang telah diekstrak</li>
                <li><strong>Dropout</strong> — Mencegah overfitting dengan
                    menonaktifkan neuron secara acak</li>
            </ul>
        """
        )

    with col_method_r:
        render_glass_card(
            """
            <h4 style="margin-top:0; color:#FFFFFF;">
                ⚙️ Proses Operasi Konvolusi
            </h4>
            <div class="arch-box" style="margin-bottom:8px;">
                Input Image (224 × 224 × 3)
            </div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-box" style="margin-bottom:8px;
                background:rgba(108,99,255,0.12);">
                Kernel/Filter (3×3) melakukan sliding pada citra<br>
                menghasilkan Feature Map
            </div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-box" style="margin-bottom:8px;">
                Aktivasi ReLU: f(x) = max(0, x)<br>
                Menghilangkan nilai negatif
            </div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-box" style="margin-bottom:8px;
                background:rgba(108,99,255,0.12);">
                Max Pooling (2×2)<br>
                Reduksi dimensi, mempertahankan fitur penting
            </div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-box" style="border-color:rgba(0,217,166,0.3);
                background:rgba(0,217,166,0.06);">
                Proses diulang beberapa kali (3 blok)<br>
                menghasilkan fitur semakin abstrak
            </div>
        """
        )

    render_divider()

    # Alur Deteksi
    st.markdown(
        '<h2><span class="gradient-text-teal">'
        "Alur Deteksi Deepfake</span></h2>",
        unsafe_allow_html=True,
    )

    steps = [
        ("1", "Input Gambar", "Pengguna mengunggah citra wajah dalam format JPG/PNG."),
        ("2", "Preprocessing", "Resize ke 224×224, normalisasi piksel ke rentang [0,1]."),
        (
            "3",
            "Ekstraksi Fitur",
            "CNN mengekstrak fitur melalui tiga blok konvolusi bertingkat.",
        ),
        (
            "4",
            "Klasifikasi",
            "Fully connected layers mengklasifikasikan fitur menjadi Real/Fake.",
        ),
        (
            "5",
            "Output Prediksi",
            "Sigmoid menghasilkan probabilitas, ditampilkan sebagai skor kepercayaan.",
        ),
    ]

    step_cols = st.columns(5)
    for idx, (num, title, desc) in enumerate(steps):
        with step_cols[idx]:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{num}</div>
                    <div style="font-weight:700; color:#FFFFFF;
                        font-size:0.95rem; margin-bottom:6px;">
                        {title}
                    </div>
                    <div style="color:var(--muted); font-size:0.82rem;
                        line-height:1.5;">
                        {desc}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    render_divider()

    # Dataset
    col_ds, col_ref = st.columns(2, gap="large")

    with col_ds:
        render_glass_card(
            """
            <h3 style="margin-top:0;">
                <span class="gradient-text">📁 Informasi Dataset</span>
            </h3>
            <p style="color:var(--text); line-height:1.8; font-size:0.92rem;">
                Dataset yang digunakan terdiri dari kumpulan citra wajah yang
                terbagi menjadi dua kelas:
            </p>
            <table class="styled-table">
                <tr>
                    <th>Kelas</th>
                    <th>Deskripsi</th>
                </tr>
                <tr>
                    <td style="color:#00D9A6; font-weight:600;">
                        ✅ Real
                    </td>
                    <td>Citra wajah asli yang belum dimanipulasi</td>
                </tr>
                <tr>
                    <td style="color:#FF4757; font-weight:600;">
                        ❌ Fake
                    </td>
                    <td>Citra wajah hasil deepfake / manipulasi AI</td>
                </tr>
            </table>
            <p style="color:var(--muted); font-size:0.85rem; margin-top:12px;">
                Dataset dibagi menjadi <strong>Training Set</strong> (80%) dan
                <strong>Testing Set</strong> (20%) untuk validasi model.
            </p>
        """
        )

    with col_ref:
        render_glass_card(
            """
            <h3 style="margin-top:0;">
                <span class="gradient-text">📚 Referensi</span>
            </h3>
            <ol style="color:var(--text); line-height:2.2; font-size:0.88rem;
                padding-left:20px;">
                <li>Goodfellow, I., et al. (2014). <em>Generative Adversarial
                    Networks</em>. NeurIPS.</li>
                <li>Rossler, A., et al. (2019). <em>FaceForensics++: Learning
                    to Detect Manipulated Facial Images</em>. ICCV.</li>
                <li>LeCun, Y., et al. (1998). <em>Gradient-based learning applied
                    to document recognition</em>. Proc. IEEE.</li>
                <li>He, K., et al. (2016). <em>Deep Residual Learning for Image
                    Recognition</em>. CVPR.</li>
                <li>Selvaraju, R., et al. (2017). <em>Grad-CAM: Visual
                    Explanations from Deep Networks</em>. ICCV.</li>
            </ol>
            <p style="color:var(--muted); font-size:0.82rem; margin-top:8px;">
                <em>Daftar referensi akan diperbarui sesuai perkembangan
                penelitian.</em>
            </p>
        """
        )

    render_divider()

    # Tim / Penulis
    st.markdown(
        '<h2 style="text-align:center;">'
        '<span class="gradient-text-teal">👥 Tim Peneliti</span></h2>',
        unsafe_allow_html=True,
    )

    tc1, tc2, tc3 = st.columns([1, 2, 1])
    with tc2:
        render_glass_card(
            """
            <div style="text-align:center;">
                <span style="font-size:3rem;">👨‍💻</span>
                <h3 style="color:#FFFFFF; margin:12px 0 4px;">
                    [Nama Penulis]
                </h3>
                <p style="color:var(--primary); font-size:0.9rem; margin:0;">
                    Mahasiswa — Program Studi Informatika
                </p>
                <p style="color:var(--muted); font-size:0.85rem; margin:4px 0 16px;">
                    [Nama Universitas]
                </p>
                <div class="custom-divider"></div>
                <p style="color:var(--muted); font-size:0.85rem;">
                    <strong style="color:var(--text);">Dosen Pembimbing:</strong><br>
                    [Nama Dosen Pembimbing 1]<br>
                    [Nama Dosen Pembimbing 2]
                </p>
            </div>
        """
        )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Fungsi utama aplikasi."""
    inject_custom_css()
    page = render_sidebar()

    if page == "🏠 Beranda":
        page_beranda()
    elif page == "🔍 Deteksi":
        page_deteksi()
    elif page == "📊 Evaluasi Model":
        page_evaluasi()
    elif page == "📖 Tentang":
        page_tentang()

    render_footer()


if __name__ == "__main__":
    main()
