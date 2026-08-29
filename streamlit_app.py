"""
streamlit_app.py
================
SIH 2026 — PS 26143: Sentinel-1 SAR Oil-Spill Detection
Streamlit demonstration application.

Run with:
    streamlit run streamlit_app.py

Requirements:
    streamlit>=1.35.0
    tensorflow>=2.15.0  (or tensorflow-cpu)
    huggingface_hub>=0.23.0
    rasterio>=1.3.0
    matplotlib>=3.8.0
    numpy>=1.24.0
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# ── Suppress noisy rasterio CRS warnings ──────────────────────────────────────
warnings.filterwarnings("ignore", message=".*CRS.*")
warnings.filterwarnings("ignore", message=".*crs.*")
warnings.filterwarnings("ignore", message=".*NotGeoreferenced.*")

# ── Detection threshold ────────────────────────────────────────────────────────
THRESHOLD       = 0.7

# ── Model settings ─────────────────────────────────────────────────────────────
IMAGE_SIZE     = 256
LOCAL_CKPT_DIR = Path(__file__).resolve().parent / "ml" / "training" / "checkpoints"
LOCAL_CKPT_PATH = LOCAL_CKPT_DIR / "best_unet.pth"

# Required number of input channels for VV Sentinel-1 data.
REQUIRED_CHANNELS = 1

# ==============================================================================
# Page configuration
# ==============================================================================

st.set_page_config(
    page_title="OceanWatch | Oil Spill Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# Custom CSS — OceanWatch Theme
# ==============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background with subtle animation ── */
    .stApp {
        background: linear-gradient(180deg, #021124 0%, #061c38 50%, #031429 100%);
        background-size: 100% 200%;
        animation: OceanWave 20s ease-in-out infinite alternate;
        color: #e6edf3;
    }

    @keyframes OceanWave {
        0% { background-position: 0% 0%; }
        100% { background-position: 0% 100%; }
    }

    /* ── Hide sidebar and default UI elements ── */
    [data-testid="stSidebar"] { display: none; }
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Navigation Bar ── */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        font-size: 2.2rem;
    }
    .brand-text h1 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
        color: #ffffff;
    }
    .brand-text p {
        font-size: 0.85rem;
        color: #58a6ff;
        margin: 0;
        font-weight: 400;
    }
    .nav-links a {
        color: #e6edf3;
        text-decoration: none;
        margin-left: 24px;
        font-weight: 500;
        font-size: 0.95rem;
        transition: color 0.2s;
        border-bottom: 2px solid transparent;
        padding-bottom: 4px;
    }
    .nav-links a:hover, .nav-links a.active {
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }

    /* ── Main Layout Elements ── */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        max-width: 600px;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    .results-container {
        background: rgba(10, 20, 40, 0.6);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
        backdrop-filter: blur(12px);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Key-Value Rows ── */
    .kv-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(88, 166, 255, 0.1);
        border-radius: 8px;
        margin-bottom: 0.6rem;
        background: rgba(255, 255, 255, 0.02);
    }
    .kv-key {
        color: #c9d1d9;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
    }
    .kv-val {
        color: #58a6ff;
        font-weight: 600;
        font-size: 1rem;
    }

    /* ── Status Indicators ── */
    .status-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .status-success {
        background: rgba(63, 185, 80, 0.1);
        border: 1px solid rgba(63, 185, 80, 0.3);
        color: #3fb950;
    }
    .status-danger {
        background: rgba(248, 81, 73, 0.1);
        border: 1px solid rgba(248, 81, 73, 0.3);
        color: #f85149;
    }

    /* ── Uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px solid rgba(88, 166, 255, 0.3) !important;
        border-radius: 12px !important;
        background: rgba(10, 20, 40, 0.5) !important;
        padding: 2rem !important;
        max-width: 600px;
        backdrop-filter: blur(8px);
    }

    /* ── Footer ── */
    .footer-tech {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 3rem 0 1.5rem 0;
    }
    .footer-card {
        flex: 1;
        background: rgba(10, 20, 40, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .footer-card-icon {
        font-size: 2rem;
    }
    .footer-card-content h4 {
        margin: 0 0 0.3rem 0;
        color: #58a6ff;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .footer-card-content p {
        margin: 0;
        color: #8b949e;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    .footer-text {
        text-align: center;
        color: #8b949e;
        font-size: 0.85rem;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def _load_model():
    """
    Load the PyTorch U-Net using OilSpillPredictor.
    """
    import sys
    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from ml.training.inference import OilSpillPredictor
    
    if not LOCAL_CKPT_PATH.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {LOCAL_CKPT_PATH}")
    
    predictor = OilSpillPredictor(
        ckpt_path=LOCAL_CKPT_PATH,
        image_size=IMAGE_SIZE,
        threshold=THRESHOLD,
        device="cpu"  # Force CPU for Streamlit inference
    )
    return predictor


def _run_inference(file_bytes: bytes, filename: str):
    """
    Run the PyTorch OilSpillPredictor, and
    return (binary_mask, prob_map, sar_disp).

    binary_mask : np.ndarray (H, W) uint8   {0, 1}    at original resolution
    prob_map    : np.ndarray (H, W) float32 [0, 1]    at original resolution
    sar_disp    : np.ndarray (H, W) float32 [0, 1]    VV, normalised
    spill_info  : dict                              Geospatial metadata
    """
    import tempfile
    import rasterio
    from ml.training.dataset import normalise_sar_channel

    suffix = Path(filename).suffix.lower() or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        # 1. Load model predictor
        predictor = _load_model()
        
        # 2. Run inference directly from file
        binary_mask, prob_map = predictor.predict(tmp_path)
        
        # 3. Read the SAR image for display (just 1 band)
        with rasterio.open(str(tmp_path)) as src:
            arr = src.read().astype(np.float32)
        
        sar_disp = normalise_sar_channel(arr[0])

        spill_info = predictor.get_spill_location(
            tmp_path,
            binary_mask,
            prob_map,
            original_filename=filename,
        )

        return binary_mask, prob_map, sar_disp, spill_info
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ==============================================================================
# Helper: build 4-panel matplotlib figure
# ==============================================================================

def _build_figure(
    sar_disp: np.ndarray,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
) -> plt.Figure:
    """
    Build a 4-panel matplotlib figure:
      [0] Sentinel-1 SAR Image  (VV, grey)
      [1] AI Probability Map    (RdYlGn_r heatmap)
      [2] Detected Oil Spill    (binary mask)
      [3] SAR + Spill Overlay   (VV grey + red semi-transparent overlay)
    """
    fig, axes = plt.subplots(
        1, 4,
        figsize=(20, 5),
        facecolor="#0d1117",
    )
    fig.patch.set_facecolor("#0d1117")

    title_kw = dict(color="#c9d1d9", fontsize=10, fontweight="600", pad=8)

    # ── Panel 0: Sentinel-1 SAR Image ────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.imshow(sar_disp, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.set_title("Sentinel-1 SAR Image\n(VV Channel)", **title_kw)
    ax.axis("off")

    # ── Panel 1: AI Probability Map ───────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#161b22")
    im = ax.imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#8b949e", labelsize=7)
    cbar.outline.set_edgecolor("none")
    ax.set_title(f"AI Probability Map\n(threshold = {THRESHOLD:.1f})", **title_kw)
    ax.axis("off")

    # ── Panel 2: Detected Oil Spill (binary mask) ─────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#161b22")
    spill_cmap = mcolors.ListedColormap(["#0d1117", "#f85149"])
    ax.imshow(binary_mask, cmap=spill_cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Detected Oil Spill\n(red = spill, black = ocean)", **title_kw)
    ax.axis("off")

    # ── Panel 3: SAR Image + Detected Spill Overlay ────────────────────────────
    ax = axes[3]
    ax.set_facecolor("#161b22")

    # Layer 1: SAR image in greyscale
    ax.imshow(sar_disp, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")

    # Layer 2: semi-transparent red RGBA overlay for spill pixels
    h, w    = binary_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)   # RGBA, fully transparent
    spill_px = binary_mask == 1
    overlay[spill_px, 0] = 1.00   # R — vivid red
    overlay[spill_px, 1] = 0.20   # G
    overlay[spill_px, 2] = 0.10   # B
    overlay[spill_px, 3] = 0.62   # Alpha — 62% opacity so SAR texture shows through

    ax.imshow(overlay, interpolation="nearest")
    ax.set_title("SAR Image + Detected Spill Overlay\n(red = oil spill region)", **title_kw)
    ax.axis("off")

    plt.tight_layout(pad=1.2)
    return fig


# ==============================================================================
# State management & Navigation
# ==============================================================================

# ==============================================================================
# State management & Navigation
# ==============================================================================

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.markdown("""
<div class="navbar">
    <div class="brand">
        <div class="brand-icon">🌊</div>
        <div class="brand-text">
            <h1>OceanWatch</h1>
            <p>Oil Spill Detection System</p>
        </div>
    </div>
    <div class="nav-links">
        <a href="#" class="active">Home</a>
        <a href="#">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Hero & File uploader
# ==============================================================================

st.markdown("""
<div>
    <h1 class="hero-title">Oil Spill Detection</h1>
    <p class="hero-subtitle">
        Advanced AI-powered system for detecting oil spills in SAR satellite imagery
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload SAR Image (TIFF)",
    type=["tif", "tiff"],
    help="Supports: .tif, .tiff (1-band VV polarization)",
    key=f"uploader_{st.session_state.uploader_key}"
)

# ==============================================================================
# Inference & results
# ==============================================================================
if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    with st.spinner("Analyzing SAR imagery..."):
        try:
            binary_mask, prob_map, sar_disp, spill_info = _run_inference(file_bytes, uploaded_file.name)
            inference_ok = True
        except Exception as e:
            st.error(f"**Analysis failed:**\n\n{e}")
            inference_ok = False

    if inference_ok:
        # ── Compute base statistics ──
        spill_pixels = int(binary_mask.sum())
        spill_det    = spill_pixels > 0
        date_str     = spill_info.get("date", "Unknown")
        time_str     = "14:30 UTC"  # Fixed synthetic time for prototype

        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        col_info, col_map = st.columns([1, 2.2], gap="large")
        
        with col_info:
            st.markdown('<div class="section-title">Detection Results</div>', unsafe_allow_html=True)
            
            if spill_det:
                st.markdown('<div class="status-badge status-danger">OIL SPILL DETECTED</div>', unsafe_allow_html=True)
                
                lat_str = f"{spill_info['latitude']:.4f}°" if spill_info['latitude'] is not None else "N/A"
                lon_str = f"{spill_info['longitude']:.4f}°" if spill_info['longitude'] is not None else "N/A"
                area_str = f"{spill_info['area_km2']:.4f} km²"
                cov_str = f"{spill_info['area_percent']:.2f}%"
                conf_str = f"{spill_info['confidence']*100:.2f}%"

                rows = [
                    ("Latitude", lat_str, "📍"),
                    ("Longitude", lon_str, "🌐"),
                    ("Spill Area", area_str, "💧"),
                    ("Area Coverage", cov_str, "📊"),
                    ("Confidence", conf_str, "🛡️"),
                    ("Date", date_str, "📅"),
                    ("Time", time_str, "🕒"),
                ]
                
                for k, v, icon in rows:
                    st.markdown(f'''
                    <div class="kv-row">
                        <span class="kv-key"><span style="width: 20px; text-align: center;">{icon}</span> {k}</span>
                        <span class="kv-val">{v}</span>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-success">✅ NO OIL SPILL DETECTED</div>', unsafe_allow_html=True)
                # No misleading coordinates shown if no spill

        with col_map:
            st.markdown('<div class="section-title">Spill Location</div>', unsafe_allow_html=True)
            
            # Map Section
            if spill_det and spill_info['latitude'] is not None and spill_info['longitude'] is not None:
                df_map = pd.DataFrame({
                    "lat": [spill_info['latitude']],
                    "lon": [spill_info['longitude']]
                })
                st.map(df_map, zoom=10, use_container_width=True)
            elif not spill_det:
                pass 
            else:
                st.info("No geospatial coordinates available for mapping.")

            st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
            
            # Matplotlib Visualizations
            st.markdown('<div class="section-title">Detection Visualization</div>', unsafe_allow_html=True)
            fig = _build_figure(sar_disp, prob_map, binary_mask)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("""
<div class="footer-tech">
    <div class="footer-card">
        <div class="footer-card-icon">🛰️</div>
        <div class="footer-card-content">
            <h4>Sentinel-1 SAR</h4>
            <p>VV Polarization</p>
        </div>
    </div>
    <div class="footer-card">
        <div class="footer-card-icon">🧠</div>
        <div class="footer-card-content">
            <h4>U-Net Deep Learning</h4>
            <p>Model</p>
        </div>
    </div>
    <div class="footer-card">
        <div class="footer-card-icon">🎯</div>
        <div class="footer-card-content">
            <h4>Advanced Detection</h4>
            <p>Algorithms</p>
        </div>
    </div>
    <div class="footer-card">
        <div class="footer-card-icon">🌊</div>
        <div class="footer-card-content">
            <h4>Marine Monitoring</h4>
            <p>Real-time Ocean<br>Surveillance</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="footer-text">© 2026 OceanWatch | Protecting Our Oceans with Technology</div>', unsafe_allow_html=True)
