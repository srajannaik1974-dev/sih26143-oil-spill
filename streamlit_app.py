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
    page_title="SIH 2026 | Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS — clean, dark SIH-demo aesthetic
# ==============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
        color: #e6edf3;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #006064 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-banner p {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.75);
        margin: 0;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #58a6ff !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #58a6ff;
        border-left: 4px solid #1f6feb;
        padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* ── Warning banner ── */
    .warning-card {
        background: rgba(210, 153, 34, 0.12);
        border: 1px solid rgba(210, 153, 34, 0.35);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: #e3b341;
        margin-top: 1rem;
    }

    /* ── Spill detected badge ── */
    .badge-spill {
        display: inline-block;
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid rgba(248, 81, 73, 0.4);
        color: #f85149;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-clean {
        display: inline-block;
        background: rgba(63, 185, 80, 0.15);
        border: 1px solid rgba(63, 185, 80, 0.4);
        color: #3fb950;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(22, 27, 34, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    /* ── Uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(88, 166, 255, 0.35) !important;
        border-radius: 12px !important;
        background: rgba(88, 166, 255, 0.04) !important;
    }

    /* ── Visualization captions ── */
    .viz-caption {
        font-size: 0.8rem;
        color: #8b949e;
        text-align: center;
        margin-top: 0.4rem;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
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

        return binary_mask, prob_map, sar_disp
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
# Sidebar
# ==============================================================================

with st.sidebar:
    st.markdown("## 🛢️ Oil Spill Detector")
    st.markdown("**SIH 2026 · Problem Statement 26143**")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This tool uses a **U-Net deep learning model** trained on "
        "Sentinel-1 SAR satellite imagery to detect oil spills at sea.\n\n"
        "Upload a 1-channel (VV) Sentinel-1 TIFF to begin."
    )
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown(f"- **Architecture:** U-Net (Keras/TensorFlow)")
    st.markdown(f"- **Input size:** {IMAGE_SIZE}×{IMAGE_SIZE} px")
    st.markdown(f"- **Detection threshold:** `{THRESHOLD}`")
    st.markdown(f"- **Sensor:** Sentinel-1 SAR (VV)")
    st.markdown("---")
    st.markdown("### Team")
    st.markdown("SIH 2026 · Batch 2024 · AIML")


# ==============================================================================
# Hero banner
# ==============================================================================

st.markdown(
    """
    <div class="hero-banner">
        <h1>🛢️ AI Oil Spill Detection — Sentinel-1 SAR</h1>
        <p>
            SIH 2026 · Problem Statement 26143 &nbsp;|&nbsp;
            U-Net Semantic Segmentation &nbsp;|&nbsp;
            Sentinel-1 VV SAR Imagery
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# File uploader
# ==============================================================================

st.markdown('<p class="section-header">📁 Upload SAR Image</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a Sentinel-1 SAR TIFF file (.tif / .tiff) — 1 band required (VV)",
    type=["tif", "tiff"],
    help=(
        "The file must be a 1-channel Sentinel-1 SAR TIFF "
        "(band 1 = VV polarisation). "
        "Typical file size: 5–200 MB."
    ),
    key="sar_upload",
)

# ==============================================================================
# Inference & results
# ==============================================================================

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    st.markdown('<p class="section-header">⚙️ Running Inference…</p>', unsafe_allow_html=True)

    with st.spinner("Loading model and running inference — please wait…"):
        try:
            binary_mask, prob_map, sar_disp = _run_inference(file_bytes, uploaded_file.name)
            inference_ok = True
        except FileNotFoundError as e:
            st.error(f"**Model not found:**\n\n{e}")
            inference_ok = False
        except ValueError as e:
            st.error(
                f"**Invalid TIFF file:**\n\n{e}\n\n"
                "Please upload a Sentinel-1 SAR TIFF with exactly 1 band (VV)."
            )
            inference_ok = False
        except Exception as e:
            st.error(f"**Inference failed:**\n\n{e}")
            inference_ok = False

    if inference_ok:
        # ── Compute statistics ─────────────────────────────────────────────────
        h, w         = binary_mask.shape
        total_pixels = h * w
        spill_pixels = int(binary_mask.sum())
        spill_pct    = 100.0 * spill_pixels / total_pixels
        mean_conf    = float(np.mean(prob_map))
        max_conf     = float(np.max(prob_map))
        spill_det    = spill_pixels > 0

        # ── Detection status badge ──────────────────────────────────────────────
        st.markdown('<p class="section-header">📊 Prediction Results</p>', unsafe_allow_html=True)

        if spill_det:
            st.markdown(
                '<span class="badge-spill">⚠️ OIL SPILL DETECTED</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-clean">✅ NO OIL SPILL DETECTED</span>',
                unsafe_allow_html=True,
            )

        st.markdown("")  # spacer

        # ── Metrics row ────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Predicted Spill Area", f"{spill_pct:.2f}%")
        with c2:
            st.metric("Spill Pixels", f"{spill_pixels:,}")
        with c3:
            st.metric("Mean Confidence", f"{mean_conf:.3f}")
        with c4:
            st.metric("Peak Confidence", f"{max_conf:.3f}")
        with c5:
            st.metric("Detection Threshold", f"{THRESHOLD:.1f}")

        # ── Visualizations ─────────────────────────────────────────────────────
        st.markdown('<p class="section-header">🖼️ Visualizations</p>', unsafe_allow_html=True)

        fig = _build_figure(sar_disp, prob_map, binary_mask)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Captions ───────────────────────────────────────────────────────────
        cap_cols = st.columns(4)
        captions = [
            "Sentinel-1 SAR Image · VV polarisation, greyscale",
            f"AI Probability Map · Confidence heat-map (threshold = {THRESHOLD:.1f})",
            "Detected Oil Spill · Binary mask — red = predicted spill region",
            "SAR + Detected Spill Overlay · Red region = oil spill on SAR image",
        ]
        for col, cap in zip(cap_cols, captions):
            with col:
                st.markdown(f'<p class="viz-caption">{cap}</p>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Prototype warning ──────────────────────────────────────────────────
        st.markdown(
            """
            <div class="warning-card">
                ⚠️ <strong>Prototype AI Estimate — Not a Verified Real-World Measurement.</strong><br>
                This tool is a proof-of-concept developed for SIH 2026 (Problem Statement 26143).
                Predictions are generated by a U-Net model trained on publicly available Sentinel-1 SAR data
                and have <strong>not</strong> been validated against ground-truth field surveys.
                Do not use this output for operational maritime safety, environmental response,
                or regulatory decisions. Always consult certified remote-sensing analysts
                and verified sensor data.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Image & prediction details expander ───────────────────────────────
        with st.expander("📐 Image & Prediction Details", expanded=False):
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.markdown("**Uploaded File**")
                st.markdown(f"- Filename: `{uploaded_file.name}`")
                st.markdown(f"- File size: `{len(file_bytes) / 1024:.1f} KB`")
                st.markdown(f"- Image resolution: `{w} × {h} px`")
                st.markdown(f"- Total pixels: `{total_pixels:,}`")
            with detail_cols[1]:
                st.markdown("**Model Configuration**")
                st.markdown(f"- Architecture: `U-Net (1-channel input)`")
                st.markdown(f"- Input resize: `{IMAGE_SIZE} × {IMAGE_SIZE} px`")
                st.markdown(f"- Detection threshold: `{THRESHOLD}`")
                st.markdown(f"- Inference device: `CPU`")

else:
    # ── Placeholder when no file is uploaded ──────────────────────────────────
    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px dashed rgba(88,166,255,0.2);
            border-radius: 12px;
            padding: 2.5rem;
            text-align: center;
            color: #8b949e;
            margin-top: 1rem;
        ">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">🛰️</p>
            <p style="font-size: 1rem; font-weight: 600; color: #58a6ff;">
                Upload a Sentinel-1 SAR TIFF to begin analysis
            </p>
            <p style="font-size: 0.85rem; margin: 0;">
                The model accepts 1-channel (VV) GeoTIFF files from the Sentinel-1 SAR sensor.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prototype warning on the landing page too ──────────────────────────────
    st.markdown(
        """
        <div class="warning-card">
            ⚠️ <strong>Prototype AI Estimate — Not a Verified Real-World Measurement.</strong>
            This tool is a proof-of-concept developed for SIH 2026 (Problem Statement 26143).
            Do not use this output for operational decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )
