"""
streamlit_app.py
================
SIH 2026 -- PS 26143: Sentinel-1 SAR Oil-Spill Detection
Streamlit demonstration application.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import base64
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from streamlit_folium import st_folium
import streamlit as st

warnings.filterwarnings("ignore", message=".*CRS.*")
warnings.filterwarnings("ignore", message=".*crs.*")
warnings.filterwarnings("ignore", message=".*NotGeoreferenced.*")

THRESHOLD         = 0.7
IMAGE_SIZE        = 256
LOCAL_CKPT_DIR    = Path(__file__).resolve().parent / "ml" / "training" / "checkpoints"
LOCAL_CKPT_PATH   = LOCAL_CKPT_DIR / "best_unet.pth"
REQUIRED_CHANNELS = 1

# ==============================================================================
# Page configuration
# ==============================================================================

st.set_page_config(
    page_title="OceanWatch | Oil Spill Detection",
    page_icon="\U0001f30a",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# CSS
# ==============================================================================

_OBG_PATH = Path(__file__).resolve().parent / "ocean_bg.jpg"
try:
    _OBG_B64 = base64.b64encode(_OBG_PATH.read_bytes()).decode()
    _OBG_CSS = f"url('data:image/jpeg;base64,{_OBG_B64}')"
except Exception:
    _OBG_B64 = ""
    _OBG_CSS = "linear-gradient(160deg,#020c1b 0%,#081e38 35%,#071530 65%,#020e20 100%)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── OVERRIDE STREAMLIT VARIABLES ── */
:root {{
    --text-color: #ffffff !important;
    --secondary-text-color: #ffffff !important;
    --font: 'Inter', sans-serif !important;
    color-scheme: dark !important;
}}

/* ── OCEAN BACKGROUND WITH INTEGRATED OVERLAY ──────────────── */
html, body {{
    background: linear-gradient(rgba(2, 10, 24, 0.62), rgba(2, 10, 24, 0.62)), {_OBG_CSS} center center / cover no-repeat fixed !important;
    background-color: #04111f !important;
    min-height: 100vh;
    font-family: 'Inter', sans-serif !important;
}}

/* ── GENERAL TEXT FORCE WHITE (base rules) ─────────────────── */
html, body, .stApp, 
.stApp p, .stApp span, .stApp label, .stApp li, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown p, .stMarkdown span, .stMarkdown div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {{
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ── DISABLE STREAMLIT RUNNING/RERUN FADE/BLUR ────────────── */
.stApp, .stApp *,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] *,
[data-testid="stMain"], [data-testid="stMain"] *,
.element-container, .element-container *,
[data-testid="stVerticalBlock"] > div,
[data-st-mode="running"] *,
[data-baseweb="tab-panel"] {{
    opacity: 1 !important;
    transition: none !important;
    filter: none !important;
}}

/* ── ALL STREAMLIT WRAPPERS: transparent ──────────────────── */
.stApp, .stApp > div, .stApp > div > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"], [data-testid="stMain"] > div,
.main, .block-container,
.stVerticalBlock,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="column"] {{
    background: transparent !important;
    background-color: transparent !important;
}}

/* ── HIDE STREAMLIT CHROME ────────────────────────────────── */
[data-testid="stSidebar"], [data-testid="stHeader"],
header, #MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"] {{ display: none !important; visibility: hidden !important; }}

/* ── LAYOUT ───────────────────────────────────────────────── */
.block-container {{
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px;
    margin: 0 auto;
}}

/* ── NAVBAR ───────────────────────────────────────────────── */
.ow-navbar {{ display:flex; justify-content:space-between; align-items:center;
    padding:1rem 0 0.85rem; border-bottom:1px solid rgba(99,179,237,0.14); margin-bottom:1rem; }}
.ow-brand  {{ display:flex; align-items:center; gap:12px; }}
.ow-brand-icon {{ font-size:2rem; line-height:1; }}
.ow-brand-text h1 {{ font-size:1.35rem; font-weight:700; margin:0; line-height:1.1;
    color:#ffffff !important; letter-spacing:0.02em; }}
.ow-brand-text p  {{ font-size:0.75rem; color:#63b3ed !important; margin:0;
    font-weight:400; letter-spacing:0.02em; }}
.ow-status-badges {{ display:flex; align-items:center; gap:8px; }}
.ow-status-badge  {{ display:flex; align-items:center; gap:6px; font-size:0.7rem; font-weight:700;
    padding:4px 10px; border-radius:4px; background:rgba(6,18,44,0.85);
    backdrop-filter:blur(6px); letter-spacing:0.03em; }}
.status-complete {{ border:1px solid rgba(72,187,120,0.5);  color:#48bb78 !important; }}
.status-model    {{ border:1px solid rgba(99,179,237,0.4);  color:#63b3ed !important; }}
.status-data     {{ border:1px solid rgba(159,122,234,0.5); color:#b794f4 !important; }}
.status-env      {{ border:1px solid rgba(79,209,197,0.5);  color:#4fd1c5 !important; }}
.ow-nav-links {{ display:flex; align-items:center; gap:1.5rem; }}
.ow-nav-links a {{ color:#a0aec0 !important; text-decoration:none; font-weight:500;
    font-size:0.85rem; transition:color 0.2s; padding-bottom:2px;
    border-bottom:2px solid transparent; }}
.ow-nav-links a.active, .ow-nav-links a:hover {{ color:#63b3ed !important; border-bottom-color:#63b3ed; }}

/* ── HERO ─────────────────────────────────────────────────── */
.ow-hero {{ padding:2.5rem 0 1.5rem; }}
.ow-hero h2 {{ font-size:2.2rem; font-weight:700; color:#ffffff !important;
    margin:0 0 0.6rem; line-height:1.15; text-shadow:0 2px 16px rgba(0,0,0,0.8); }}
.ow-hero p  {{ font-size:0.95rem; color:#b0cce0 !important;
    max-width:520px; line-height:1.6; margin:0 0 2rem; }}

/* ── UPLOAD ZONE ──────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: rgba(4, 18, 50, 0.88) !important;
    border: 2px solid rgba(99,179,237,0.65) !important;
    border-radius: 14px !important;
    padding: 1.6rem !important;
    max-width: 600px !important;
    backdrop-filter: blur(14px) !important;
    box-shadow: 0 0 28px rgba(99,179,237,0.22), 0 0 70px rgba(37,99,235,0.10) !important;
    transition: box-shadow 0.3s ease, border-color 0.3s ease !important;
}}
[data-testid="stFileUploader"]:hover {{
    box-shadow: 0 0 40px rgba(99,179,237,0.40), 0 0 90px rgba(37,99,235,0.18) !important;
    border-color: rgba(147,197,253,0.9) !important;
}}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label * {{
    color: #93c5fd !important; font-weight:700 !important; font-size:1.05rem !important;
}}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] * {{ color:#93c5fd !important; }}
[data-testid="stFileUploaderDropzone"] {{
    background: rgba(20,45,110,0.35) !important;
    border: 1.5px dashed rgba(147,197,253,0.55) !important;
    border-radius: 10px !important;
}}
[data-testid="stFileUploaderFileName"] {{ color:#7dd3fc !important; font-weight:600 !important; }}
[data-testid="stFileUploaderFile"]  *  {{ color:#dde4f0 !important; }}
small {{ color:#94a3b8 !important; }}

/* ── PANELS ───────────────────────────────────────────────── */
.ow-panel {{
    background: rgba(3, 12, 30, 0.90) !important;
    border: 1px solid rgba(99,179,237,0.22) !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
}}
.ow-panel-title {{
    color: #63b3ed !important; font-size:0.92rem !important; font-weight:700 !important;
    letter-spacing:0.04em !important; padding-bottom:0.6rem !important;
    border-bottom:1px solid rgba(99,179,237,0.12) !important; margin-bottom:0.9rem !important;
    display:flex; align-items:center; gap:8px;
}}

/* ── BADGES ───────────────────────────────────────────────── */
.ow-detected-badge {{ display:flex; align-items:center; gap:10px; padding:0.6rem 0.85rem;
    border-radius:7px; font-weight:700; font-size:0.9rem; margin-bottom:0.8rem;
    text-transform:uppercase; letter-spacing:0.02em; }}
.ow-detected-badge.spill    {{ background:rgba(34,100,60,0.30); border:1px solid rgba(72,187,120,0.4);  color:#4ade80 !important; }}
.ow-detected-badge.no-spill {{ background:rgba(20,50,100,0.30); border:1px solid rgba(99,179,237,0.35); color:#7dd3fc !important; }}
.ow-detected-badge .badge-icon {{ width:18px; height:18px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; font-size:0.75rem; flex-shrink:0; }}
.ow-detected-badge.spill    .badge-icon {{ background:rgba(72,187,120,0.15); border:1px solid rgba(72,187,120,0.4); }}
.ow-detected-badge.no-spill .badge-icon {{ background:rgba(99,179,237,0.2);  border:1px solid rgba(99,179,237,0.5); }}

/* ── METRIC ROWS ──────────────────────────────────────────── */
.ow-metric-row {{ display:flex; justify-content:space-between; align-items:center;
    padding:0.5rem 0.75rem; border:1px solid rgba(99,179,237,0.08); border-radius:6px;
    margin-bottom:0.4rem; background:rgba(255,255,255,0.025); transition:background 0.15s; }}
.ow-metric-row:hover {{ background:rgba(99,179,237,0.06); }}
.ow-metric-key  {{ color:#a8c0d6 !important; font-weight:500; font-size:0.8rem;
    display:flex; align-items:center; gap:8px; }}
.ow-metric-icon {{ font-size:0.85rem; opacity:0.75; width:18px; text-align:center; }}
.ow-metric-val  {{ color:#7dd3fc !important; font-weight:600; font-size:0.82rem; }}

/* ── DRIFT CARDS ──────────────────────────────────────────── */
.drift-metrics-row {{ display:flex; gap:8px; margin-bottom:0.8rem; flex-wrap:wrap; }}
.drift-metric-card {{ flex:1; min-width:110px; background:rgba(4,14,38,0.92) !important;
    border:1px solid rgba(99,179,237,0.18); border-radius:8px; padding:8px;
    text-align:center; backdrop-filter:blur(8px); }}
.drift-metric-card .card-icon  {{ font-size:1.1rem; margin-bottom:4px; }}
.drift-metric-card .card-title {{ font-size:0.62rem; color:#7ea8c9 !important;
    font-weight:600; text-transform:uppercase; margin-bottom:4px; letter-spacing:0.02em; }}
.drift-metric-card .card-val   {{ font-size:0.75rem; font-weight:600;
    color:#dde8f5 !important; line-height:1.2; }}
.icon-red   {{ color:#f87171 !important; }}
.icon-green {{ color:#4ade80 !important; }}
.icon-blue  {{ color:#60a5fa !important; }}

/* ── MAP ──────────────────────────────────────────────────── */
.map-container {{ border:1px solid rgba(99,179,237,0.13); border-radius:8px; overflow:hidden; }}

/* ── THUMBS ───────────────────────────────────────────────── */
.thumb-label {{ text-align:center; font-size:0.62rem; color:#90a4b7 !important;
    margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.01em; }}
.stPyplot {{ border-radius:6px; overflow:hidden; border:1px solid rgba(99,179,237,0.1); }}

/* ── FOOTER ───────────────────────────────────────────────── */
.ow-footer {{ margin-top:2.5rem; border-top:1px solid rgba(99,179,237,0.09); padding-top:1.4rem; }}
.ow-footer-grid {{ display:flex; gap:0; margin-bottom:1.3rem; }}
.ow-footer-item {{ flex:1; display:flex; align-items:flex-start; gap:0.85rem;
    padding:0.9rem 1.1rem; border-right:1px solid rgba(99,179,237,0.07); }}
.ow-footer-item:first-child {{ padding-left:0; }}
.ow-footer-item:last-child  {{ border-right:none; }}
.ow-footer-icon {{ font-size:1.9rem; opacity:0.65; flex-shrink:0; margin-top:0.1rem; }}
.ow-footer-content h4 {{ font-size:0.87rem; font-weight:600; color:#63b3ed !important; margin:0 0 0.22rem; }}
.ow-footer-content p  {{ font-size:0.75rem; color:#8fa8bc !important; margin:0; line-height:1.4; }}
.ow-footer-copy   {{ text-align:center; color:#5a7a90 !important; font-size:0.78rem; padding-bottom:1.4rem; }}
.ow-footer-copy a {{ color:#63b3ed !important; text-decoration:none; }}

/* ── BUTTONS ──────────────────────────────────────────────── */
.stButton > button {{
    color: #e2e8f0 !important;
    background: rgba(20,50,120,0.82) !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    font-weight: 600 !important;
}}
.stButton > button:hover {{
    background: rgba(37,99,235,0.9) !important;
    border-color: #60a5fa !important; color:#fff !important;
}}

/* ── ALERTS ───────────────────────────────────────────────── */
.stInfo, [data-testid="stInfo"] {{
    background:rgba(15,40,100,0.80) !important; color:#bfdbfe !important;
    border-color:rgba(99,179,237,0.35) !important;
}}
.stError, [data-testid="stNotification"] {{
    background:rgba(100,20,20,0.80) !important; color:#fca5a5 !important;
    border-color:rgba(239,68,68,0.45) !important;
}}
.stWarning {{ background:rgba(100,45,10,0.80) !important; color:#fde68a !important; }}
.stSpinner > div {{ color:#93c5fd !important; }}

/* ── LIVE BADGE ───────────────────────────────────────────── */
@keyframes livePulse {{
    0%, 100% {{ opacity:1; transform:scale(1); }}
    50%       {{ opacity:0.4; transform:scale(1.35); }}
}}
.live-badge {{
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.55);
    color:#34d399 !important; font-size:0.68rem; font-weight:700;
    letter-spacing:0.06em; padding:3px 11px; border-radius:20px;
}}
.live-dot {{
    width:7px; height:7px; border-radius:50%; background:#34d399;
    animation:livePulse 1.2s ease-in-out infinite;
}}

/* ── COORD FLASH ──────────────────────────────────────────── */
@keyframes fadeFlash {{
    0%   {{ background:rgba(99,179,237,0.25); }}
    100% {{ background:transparent; }}
}}
.coord-cell {{ animation:fadeFlash 0.8s ease-out; color:#93c5fd !important; font-weight:600 !important; }}
</style>
""", unsafe_allow_html=True)



# ==============================================================================
# Backend: model load (unchanged)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def _load_model():
    import sys
    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from ml.training.inference import OilSpillPredictor
    if not LOCAL_CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {LOCAL_CKPT_PATH}")
    return OilSpillPredictor(
        ckpt_path=LOCAL_CKPT_PATH, image_size=IMAGE_SIZE,
        threshold=THRESHOLD, device="cpu",
    )


# ==============================================================================
# Backend: inference runner (unchanged logic)
# ==============================================================================

def _run_inference(file_bytes: bytes, filename: str):
    """Run OilSpillPredictor. Returns (binary_mask, prob_map, sar_disp, spill_info, img_h, img_w)."""
    import rasterio, tempfile
    from ml.training.dataset import normalise_sar_channel
    suffix = Path(filename).suffix.lower() or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        predictor = _load_model()
        binary_mask, prob_map = predictor.predict(tmp_path)
        with rasterio.open(str(tmp_path)) as src:
            arr = src.read().astype(np.float32)
            img_h, img_w = src.height, src.width
        sar_disp = normalise_sar_channel(arr[0])
        spill_info = predictor.get_spill_location(
            tmp_path, binary_mask, prob_map, original_filename=filename)
        return binary_mask, prob_map, sar_disp, spill_info, img_h, img_w
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ==============================================================================
# Backend: 4-panel matplotlib figure (unchanged)
# ==============================================================================

def _build_figure(sar_disp, prob_map, binary_mask) -> plt.Figure:
    """Build the standard 4-panel SAR analysis figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    title_kw = dict(color="#c9d1d9", fontsize=10, fontweight="600", pad=8)

    axes[0].set_facecolor("#161b22")
    axes[0].imshow(sar_disp, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    axes[0].set_title("Sentinel-1 SAR Image\n(VV Channel)", **title_kw)
    axes[0].axis("off")

    axes[1].set_facecolor("#161b22")
    im = axes[1].imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#8b949e", labelsize=7)
    cbar.outline.set_edgecolor("none")
    axes[1].set_title(f"AI Probability Map\n(threshold = {THRESHOLD:.1f})", **title_kw)
    axes[1].axis("off")

    axes[2].set_facecolor("#161b22")
    axes[2].imshow(binary_mask,
                   cmap=mcolors.ListedColormap(["#0d1117", "#f85149"]),
                   vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Detected Oil Spill\n(red=spill, black=ocean)", **title_kw)
    axes[2].axis("off")

    axes[3].set_facecolor("#161b22")
    axes[3].imshow(sar_disp, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    h, w = binary_mask.shape
    ov = np.zeros((h, w, 4), dtype=np.float32)
    px = binary_mask == 1
    ov[px, 0] = 1.0; ov[px, 1] = 0.20; ov[px, 2] = 0.10; ov[px, 3] = 0.62
    axes[3].imshow(ov, interpolation="nearest")
    axes[3].set_title("SAR + Spill Overlay\n(red=oil spill region)", **title_kw)
    axes[3].axis("off")

    plt.tight_layout(pad=1.2)
    return fig


def _make_thumb(img_data, cmap="gray", vmin=0, vmax=1) -> plt.Figure:
    """Render a single SAR channel as a small square thumbnail figure."""
    f, a = plt.subplots(figsize=(3, 3), facecolor="#0d1117")
    a.set_facecolor("#0d1117")
    a.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax)
    a.axis("off")
    f.tight_layout(pad=0)
    return f


# ==============================================================================
# Map helper: build full Folium map with spill + origin + trajectory + AIS
# ==============================================================================

def _build_drift_map(
    spill_lat: float,
    spill_lon: float,
    origin_lat: float,
    origin_lon: float,
    traj_pts: list,
    spill_area: float,
    spill_cov: float,
    spill_conf: float,
    rel_time_str: str,
    n_pts: int,
    candidate_vessels: Optional[List[Dict[str, Any]]] = None,
) -> folium.Map:
    """
    Build an interactive Folium map showing:
      - Red circle marker + label at detected spill centroid
      - Green circle marker + label at estimated origin
      - Dashed blue polyline for the backward trajectory
      - Subtle white dashed uncertainty circle around the estimated origin (20 km radius)
      - Blue circle markers + labels for candidate vessels within range (dynamic simulation updates)
      - Automatic fit_bounds to show all points
      - Street / Satellite / Ocean tile layer selector
    """
    all_lats = [spill_lat, origin_lat] + [pt.latitude  for pt in traj_pts]
    all_lons = [spill_lon, origin_lon] + [pt.longitude for pt in traj_pts]

    # Include candidate vessels in boundary calculations
    if candidate_vessels:
        for cand in candidate_vessels:
            all_lats.append(cand["latitude"])
            all_lons.append(cand["longitude"])

    center_lat = (min(all_lats) + max(all_lats)) / 2.0
    center_lon = (min(all_lons) + max(all_lons)) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,
        control_scale=True,
    )

    # Tile layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap contributors",
        name="Street Map",
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Ocean Basemap",
        name="Ocean",
        max_zoom=13,
    ).add_to(m)

    # Backward trajectory polyline
    if traj_pts:
        traj_coords = [[pt.latitude, pt.longitude] for pt in traj_pts]
        folium.PolyLine(
            locations=traj_coords,
            color="#3b82f6",  # Cyan-blue line
            weight=3.0,
            opacity=0.85,
            tooltip="Backward Trajectory",
        ).add_to(m)

        # Waypoint direction arrows along the path
        step = max(1, len(traj_pts) // 5)
        for i in range(0, len(traj_pts), step):
            pt = traj_pts[i]
            elapsed_min = i * 30
            folium.CircleMarker(
                location=[pt.latitude, pt.longitude],
                radius=4,
                color="#60a5fa",
                fill=True,
                fill_color="#1d4ed8",
                fill_opacity=0.9,
                tooltip=f"Trajectory Waypoint (T-{elapsed_min} min)\nLat: {pt.latitude:.4f}\nLon: {pt.longitude:.4f}",
            ).add_to(m)

    # Search/Uncertainty radius (20 km) around estimated origin
    folium.Circle(
        location=[origin_lat, origin_lon],
        radius=20000,  # 20km in meters
        color="#ffffff",
        weight=1.5,
        fill=False,
        dash_array="5 5",
        opacity=0.45,
        tooltip="Origin Search Radius (20 km)",
    ).add_to(m)

    # Detected spill marker (RED)
    spill_popup_html = (
        f'<div style="font-family:Inter,sans-serif;min-width:200px;padding:4px">'
        f'<div style="font-weight:700;font-size:14px;color:#ef4444;margin-bottom:6px">&#128308; Detected Oil Spill</div>'
        f'<table style="width:100%;font-size:12px;border-collapse:collapse">'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Latitude</td>'
        f'<td style="font-weight:600">{spill_lat:.4f}&deg;</td></tr>'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Longitude</td>'
        f'<td style="font-weight:600">{spill_lon:.4f}&deg;</td></tr>'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Spill Area</td>'
        f'<td style="font-weight:600">{spill_area:.2f} km&sup2;</td></tr>'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Coverage</td>'
        f'<td style="font-weight:600">{spill_cov:.2f}%</td></tr>'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Confidence</td>'
        f'<td style="font-weight:600">{spill_conf:.2f}%</td></tr>'
        f'</table></div>'
    )
    folium.CircleMarker(
        location=[spill_lat, spill_lon],
        radius=12,
        color="#ef4444",
        weight=2.5,
        fill=True,
        fill_color="#ef4444",
        fill_opacity=0.35,
        tooltip=f"Detected Spill ({spill_lat:.4f}\u00b0, {spill_lon:.4f}\u00b0)",
        popup=folium.Popup(spill_popup_html, max_width=260),
    ).add_to(m)
    
    folium.Marker(
        location=[spill_lat, spill_lon],
        icon=folium.DivIcon(
            html=(
                '<div style="background:rgba(239,68,68,0.95);color:#fff;font-size:9px;font-weight:700;'
                'padding:3px 6px;border-radius:4px;white-space:nowrap;'
                'box-shadow:0 2px 6px rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.25);'
                'font-family:Inter,sans-serif;margin-top:-16px;margin-left:14px">'
                'DETECTED SPILL<br><span style="font-size:8px;font-weight:500;opacity:0.9">2018-09-26 14:30 UTC</span></div>'
            ),
            icon_anchor=(0, 0),
        ),
    ).add_to(m)

    # Estimated origin marker (GREEN)
    origin_popup_html = (
        f'<div style="font-family:Inter,sans-serif;min-width:200px;padding:4px">'
        f'<div style="font-weight:700;font-size:14px;color:#22c55e;margin-bottom:6px">&#128994; Estimated Spill Origin</div>'
        f'<table style="width:100%;font-size:12px;border-collapse:collapse">'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Latitude</td>'
        f'<td style="font-weight:600">{origin_lat:.4f}&deg;</td></tr>'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Longitude</td>'
        f'<td style="font-weight:600">{origin_lon:.4f}&deg;</td></tr>'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Est. Release</td>'
        f'<td style="font-weight:600">{rel_time_str}</td></tr>'
        f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Traj. Points</td>'
        f'<td style="font-weight:600">{n_pts}</td></tr>'
        f'</table>'
        f'<div style="font-size:10px;color:#9ca3af;margin-top:6px">* Synthetic env. data - prototype only</div>'
        f'</div>'
    )
    folium.CircleMarker(
        location=[origin_lat, origin_lon],
        radius=12,
        color="#22c55e",
        weight=2.5,
        fill=True,
        fill_color="#22c55e",
        fill_opacity=0.35,
        tooltip=f"Estimated Origin ({origin_lat:.4f}\u00b0, {origin_lon:.4f}\u00b0)",
        popup=folium.Popup(origin_popup_html, max_width=260),
    ).add_to(m)
    
    folium.Marker(
        location=[origin_lat, origin_lon],
        icon=folium.DivIcon(
            html=(
                '<div style="background:rgba(34,197,94,0.95);color:#fff;font-size:9px;font-weight:700;'
                'padding:3px 6px;border-radius:4px;white-space:nowrap;'
                'box-shadow:0 2px 6px rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.25);'
                'font-family:Inter,sans-serif;margin-top:-16px;margin-left:14px">'
                'ESTIMATED ORIGIN<br><span style="font-size:8px;font-weight:500;opacity:0.9">2018-09-26 08:30 UTC</span></div>'
            ),
            icon_anchor=(0, 0),
        ),
    ).add_to(m)

    # Candidate vessels markers (BLUE)
    if candidate_vessels:
        for cand in candidate_vessels:
            cand_lat = cand["latitude"]
            cand_lon = cand["longitude"]
            mmsi = cand["mmsi"]
            vessel_name = cand.get("vessel_name") or "Unknown"
            vessel_type = cand.get("vessel_type") or "Unknown"
            min_dist = cand.get("minimum_distance_km", 0.0)
            time_diff = cand.get("time_difference_minutes", 0.0)
            score = cand.get("candidate_score", 0.0)
            closest_time = cand.get("closest_record_time", "")

            popup_html = (
                f'<div style="font-family:Inter,sans-serif;min-width:210px;padding:4px">'
                f'<div style="font-weight:700;font-size:13px;color:#2563eb;margin-bottom:6px">&#128674; Candidate Vessel</div>'
                f'<table style="width:100%;font-size:11px;border-collapse:collapse">'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">MMSI</td><td style="font-weight:600">{mmsi}</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Name</td><td style="font-weight:600">{vessel_name}</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Type</td><td style="font-weight:600">{vessel_type}</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Current Lat</td><td style="font-weight:600">{cand_lat:.6f}&deg; N</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Current Lon</td><td style="font-weight:600">{cand_lon:.6f}&deg; E</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Min Distance</td><td style="font-weight:600">{min_dist:.2f} km</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Observation Time</td><td style="font-weight:600">{closest_time}</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Time Difference</td><td style="font-weight:600">{time_diff:.1f} min</td></tr>'
                f'<tr><td style="color:#6b7280;padding:2px 6px 2px 0">Attribution Score</td><td style="font-weight:700;color:#2563eb">{score:.1f}/100</td></tr>'
                f'</table></div>'
            )

            # Blue circle marker for the candidate
            folium.CircleMarker(
                location=[cand_lat, cand_lon],
                radius=10,
                color="#2563eb",
                weight=2.5,
                fill=True,
                fill_color="#2563eb",
                fill_opacity=0.35,
                tooltip=f"Candidate Vessel: {vessel_name} (MMSI: {mmsi})",
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(m)

            # DIV label text
            label_text = vessel_name if vessel_name != "Unknown" else f"MMSI {mmsi}"
            folium.Marker(
                location=[cand_lat, cand_lon],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="background:rgba(37,99,235,0.95);color:#fff;font-size:9px;font-weight:700;'
                        f'padding:3px 6px;border-radius:4px;white-space:nowrap;'
                        f'box-shadow:0 2px 6px rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.25);'
                        f'font-family:Inter,sans-serif;margin-top:-16px;margin-left:14px">'
                        f'CANDIDATE: {label_text}</div>'
                    ),
                    icon_anchor=(0, 0),
                ),
            ).add_to(m)

    # Custom Floating Legend
    legend_html = (
        '<div style="position:fixed;top:20px;right:20px;z-index:999;'
        'background:rgba(10,25,50,0.85);border:1px solid rgba(99,179,237,0.2);'
        'border-radius:6px;padding:10px 14px;font-family:Inter,sans-serif;'
        'font-size:10px;color:#e2e8f0;backdrop-filter:blur(6px);box-shadow:0 4px 15px rgba(0,0,0,0.5)">'
        '<div style="margin-bottom:6px;display:flex;align-items:center;gap:6px">'
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e"></span>'
        '<span>Estimated Origin (Backtracked)</span></div>'
        '<div style="margin-bottom:6px;display:flex;align-items:center;gap:6px">'
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444"></span>'
        '<span>Detected Spill (Satellite)</span></div>'
        '<div style="margin-bottom:6px;display:flex;align-items:center;gap:6px">'
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#2563eb"></span>'
        '<span>AIS Candidate Vessel</span></div>'
        '<div style="margin-bottom:6px;display:flex;align-items:center;gap:6px">'
        '<span style="display:inline-block;width:12px;height:2px;background:#3b82f6"></span>'
        '<span>Backtracked Trajectory</span></div>'
        '<div style="display:flex;align-items:center;gap:6px">'
        '<span style="display:inline-block;width:12px;height:12px;border:1px dashed #ffffff;border-radius:50%"></span>'
        '<span>Search Radius (20 km)</span></div>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    # Fit bounds with padding
    sw = [min(all_lats) - 0.05, min(all_lons) - 0.05]
    ne = [max(all_lats) + 0.05, max(all_lons) + 0.05]
    m.fit_bounds([sw, ne])

    folium.LayerControl(position="topleft").add_to(m)
    return m


def _deg_to_compass(deg: float) -> str:
    """Convert wind or current direction degree to compass string."""
    val = int((deg / 22.5) + 0.5)
    arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return arr[val % 16]


def _build_trajectory_chart(traj_pts: list, spill_lat: float, spill_lon: float, origin_lat: float, origin_lon: float) -> plt.Figure:
    """Generate a clean, high-tech trajectory chart to show Longitude vs Latitude."""
    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor="#0a1f3d")
    ax.set_facecolor("#05122b")
    
    # Extract trajectory lats/lons
    lats = [pt.latitude for pt in traj_pts]
    lons = [pt.longitude for pt in traj_pts]
    
    # Plot trajectory path
    ax.plot(lons, lats, color="#60a5fa", linewidth=2, zorder=2)
    
    # Add direction arrows along the trajectory (origin -> detection)
    pts_forward = traj_pts[::-1]
    for i in range(len(pts_forward) - 1):
        p1 = pts_forward[i]
        p2 = pts_forward[i+1]
        ax.annotate('', xy=(p2.longitude, p2.latitude), xytext=(p1.longitude, p1.latitude),
                    arrowprops=dict(arrowstyle="-|>", color="#60a5fa", lw=1.5, mutation_scale=10, zorder=3))
                    
    # Plot origin and detection markers
    ax.scatter([origin_lon], [origin_lat], color="#22c55e", s=90, zorder=5, edgecolors="#ffffff", linewidths=1)
    ax.scatter([spill_lon], [spill_lat], color="#ef4444", s=90, zorder=5, edgecolors="#ffffff", linewidths=1)
    
    # Text labels
    ax.text(origin_lon, origin_lat + 0.004, "ESTIMATED ORIGIN", color="#22c55e", fontsize=7.5, fontweight="bold", ha="center")
    ax.text(spill_lon, spill_lat - 0.009, "DETECTED SPILL", color="#ef4444", fontsize=7.5, fontweight="bold", ha="center")
    
    # Styling
    ax.tick_params(colors="#a0aec0", labelsize=8)
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['top'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.spines['right'].set_color('#1e293b')
    
    ax.set_xlabel("Longitude (\u00b0)", color="#a0aec0", fontsize=8.5)
    ax.set_ylabel("Latitude (\u00b0)", color="#a0aec0", fontsize=8.5)
    ax.grid(color="#1e293b", linestyle=":", linewidth=0.5)
    
    plt.tight_layout()
    return fig


# ==============================================================================
# Session state
# ==============================================================================

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Live tracking is always on — no manual toggle
if "sim_step" not in st.session_state:
    st.session_state.sim_step = 0

# ==============================================================================
# NAVBAR
# ==============================================================================

st.markdown("""
<div class="ow-navbar" style="display:flex;justify-content:space-between;align-items:center;padding:1rem 0 0.85rem;border-bottom:1px solid rgba(99,179,237,0.18);margin-bottom:1rem;">
  <div class="ow-brand" style="display:flex;align-items:center;gap:12px;">
    <div style="font-size:2rem;line-height:1;">\U0001f30a</div>
    <div>
      <h1 style="font-size:1.4rem;font-weight:700;margin:0;color:#ffffff !important;letter-spacing:0.02em;">OCEANWATCH</h1>
      <p style="font-size:0.75rem;color:#63b3ed !important;margin:0;font-weight:400;">Oil Spill Detection &amp; Backtracking System</p>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="display:flex;align-items:center;gap:6px;font-size:0.7rem;font-weight:700;padding:4px 10px;border-radius:4px;background:rgba(6,20,48,0.88);border:1px solid rgba(72,187,120,0.5);color:#48bb78 !important;backdrop-filter:blur(6px);letter-spacing:0.03em;">
       <span>&#9679;</span> ANALYSIS COMPLETE
    </div>
    <div style="display:flex;align-items:center;gap:6px;font-size:0.7rem;font-weight:700;padding:4px 10px;border-radius:4px;background:rgba(6,20,48,0.88);border:1px solid rgba(99,179,237,0.4);color:#63b3ed !important;backdrop-filter:blur(6px);letter-spacing:0.03em;">
       <span>&#129504;</span> MODEL: U-NET
    </div>
    <div style="display:flex;align-items:center;gap:6px;font-size:0.7rem;font-weight:700;padding:4px 10px;border-radius:4px;background:rgba(6,20,48,0.88);border:1px solid rgba(159,122,234,0.5);color:#b794f4 !important;backdrop-filter:blur(6px);letter-spacing:0.03em;">
       <span>&#128752;</span> DATA: SENTINEL-1 SAR
    </div>
    <div style="display:flex;align-items:center;gap:6px;font-size:0.7rem;font-weight:700;padding:4px 10px;border-radius:4px;background:rgba(6,20,48,0.88);border:1px solid rgba(79,209,197,0.5);color:#4fd1c5 !important;backdrop-filter:blur(6px);letter-spacing:0.03em;">
       <span>&#9881;</span> ENVIRONMENT: SYNTHETIC
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:1.5rem;">
    <a href="#" style="color:#63b3ed !important;text-decoration:none;font-weight:600;font-size:0.85rem;">Home</a>
    <a href="#" style="color:#94a3b8 !important;text-decoration:none;font-weight:500;font-size:0.85rem;">About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# FILE UPLOADER (always visible)
# ==============================================================================

uploaded_file = st.file_uploader(
    "Upload SAR Image (TIFF)",
    type=["tif", "tiff"],
    help="Sentinel-1 VV polarization, 1-band GeoTIFF",
    key=f"uploader_{st.session_state.uploader_key}",
    label_visibility="visible",
)

# ==============================================================================
# HERO — only shown before upload
# ==============================================================================

if uploaded_file is None:
    st.markdown("""
    <div style="padding:2.5rem 0 1.5rem;">
      <h2 style="font-size:2.2rem;font-weight:700;color:#ffffff !important;margin:0 0 0.6rem;line-height:1.15;text-shadow:0 2px 20px rgba(0,0,0,0.9);">Oil Spill Detection &amp; Drift Analysis</h2>
      <p style="font-size:0.95rem;color:#b8d4e8 !important;max-width:520px;line-height:1.6;margin:0 0 2rem;">Advanced AI-powered system for detecting oil spills in Sentinel-1 SAR imagery
         and backtracking their probable origin using drift simulation.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# INFERENCE & RESULTS
# ==============================================================================

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    with st.spinner("\U0001f50d Analyzing SAR imagery\u2026"):
        try:
            binary_mask, prob_map, sar_disp, spill_info, img_h, img_w = _run_inference(
                file_bytes, uploaded_file.name
            )
            inference_ok = True
        except Exception as exc:
            st.error(f"**Analysis failed:** {exc}")
            inference_ok = False

    if inference_ok:
        # Derived scalars
        spill_pixels = int(binary_mask.sum())
        spill_det    = spill_pixels > 0
        date_str     = spill_info.get("date", "Unknown")
        time_str     = "14:30:00"

        lat_val  = spill_info.get("latitude")
        lon_val  = spill_info.get("longitude")
        area_km2 = spill_info.get("area_km2", 0.0) or 0.0
        area_pct = spill_info.get("area_percent", 0.0) or 0.0
        conf_val = spill_info.get("confidence", 0.0) or 0.0

        lat_str  = f"{lat_val:.6f}\u00b0 N" if lat_val is not None else "N/A"
        lon_str  = f"{lon_val:.6f}\u00b0 E" if lon_val is not None else "N/A"
        area_str = f"{area_km2:.2f} km\u00b2"
        cov_str  = f"{area_pct:.2f}%"
        conf_str = f"{conf_val * 100:.2f}%"

        # Run Member 2 drift analysis safely
        drift_result = None
        drift_ok     = False
        drift_err    = ""
        if spill_det and lat_val is not None and lon_val is not None:
            try:
                from drift_adapter import run_drift_analysis
                drift_result = run_drift_analysis(spill_info, duration_hours=6.0, step_minutes=30.0)
                drift_ok = True
            except Exception as _de:
                drift_err = str(_de)

        # Run Member 3 AIS analysis safely (origin + release_time -> AIS search)
        ais_result = None
        ais_ok     = False
        ais_err    = ""
        if drift_ok and drift_result is not None:
            try:
                from ais_adapter import run_ais_analysis
                ais_result = run_ais_analysis(
                    probable_latitude=drift_result.probable_latitude,
                    probable_longitude=drift_result.probable_longitude,
                    estimated_release_time=drift_result.estimated_release_time,
                    ais_csv_path=None,  # will default to sih_demo_ais.csv
                    search_radius_km=20.0,
                    time_window_minutes=60.0,
                    top_n_candidates=5
                )
                ais_ok = True
            except Exception as _ae:
                ais_err = str(_ae)

        # ---- LIVE VESSEL MOVEMENT ANIMATION (always-on) ----
        if ais_ok and ais_result and ais_result.get("candidate_vessels"):
            sim_step = st.session_state.get("sim_step", 0)

            for cand in ais_result["candidate_vessels"]:
                pts = cand.get("ais_records", [])
                if len(pts) >= 1:
                    start_pt = pts[0]
                    # Extrapolate destination if only one point available
                    if len(pts) == 1:
                        end_lat = start_pt["latitude"] + 0.012
                        end_lon = start_pt["longitude"] - 0.004
                    else:
                        end_pt = pts[-1]
                        end_lat = end_pt["latitude"]
                        end_lon = end_pt["longitude"]

                    start_lat = start_pt["latitude"]
                    start_lon = start_pt["longitude"]

                    # Smooth interpolation over 100 steps
                    t_frac = (sim_step % 100) / 100.0
                    cand["latitude"]  = start_lat + t_frac * (end_lat - start_lat)
                    cand["longitude"] = start_lon + t_frac * (end_lon - start_lon)

        # Header row: Live badge + New Analysis button
        col_hdr_left, col_hdr_right = st.columns([5, 1])
        with col_hdr_left:
            st.markdown("""
            <div style='display:flex;align-items:center;gap:10px;padding:0.3rem 0'>
              <span class='live-badge'><span class='live-dot'></span>LIVE TRACKING</span>
            </div>
            """, unsafe_allow_html=True)
        with col_hdr_right:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("\u2190 New Analysis", use_container_width=True):
                st.session_state.uploader_key += 1
                st.rerun()

        # ================================================================
        # MAIN DASHBOARD TWO-COLUMN LAYOUT (Left 25%, Right 75%)
        # ================================================================
        col_left, col_right = st.columns([0.9, 3.1], gap="medium")

        # ---- LEFT COLUMN: Detection Results + SAR Overview ----
        with col_left:
            st.markdown('<div class="ow-panel">', unsafe_allow_html=True)
            st.markdown('<div class="ow-panel-title">DETECTION RESULTS</div>', unsafe_allow_html=True)

            if spill_det:
                st.markdown("""
                <div class="ow-detected-badge spill">
                  <div class="badge-icon">\u2713</div>
                  <span>Oil Spill Detected</span>
                </div>""", unsafe_allow_html=True)

                for icon, key, val in [
                    ("\U0001f4cd", "Latitude",      lat_str),
                    ("\U0001f310", "Longitude",     lon_str),
                    ("\U0001f4a7", "Spill Area",    area_str),
                    ("\U0001f4ca", "Area Coverage", cov_str),
                ]:
                    st.markdown(f"""
                    <div class="ow-metric-row">
                      <span class="ow-metric-key"><span class="ow-metric-icon">{icon}</span>{key}</span>
                      <span class="ow-metric-val">{val}</span>
                    </div>""", unsafe_allow_html=True)
                
                # Confidence in green
                st.markdown(f"""
                <div class="ow-metric-row">
                  <span class="ow-metric-key"><span class="ow-metric-icon">\U0001f6e1</span>Confidence</span>
                  <span class="ow-metric-val" style="color: #48bb78; font-weight: 700;">{conf_str}</span>
                </div>""", unsafe_allow_html=True)

                for icon, key, val in [
                    ("\U0001f4c5", "Date",          date_str),
                    ("\U0001f550", "Time (UTC)",    time_str),
                ]:
                    st.markdown(f"""
                    <div class="ow-metric-row">
                      <span class="ow-metric-key"><span class="ow-metric-icon">{icon}</span>{key}</span>
                      <span class="ow-metric-val">{val}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="ow-detected-badge no-spill">
                  <div class="badge-icon">\u2713</div>
                  <span>No Spill Detected</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # SAR Overview (2x2 thumbnails)
            st.markdown('<div class="ow-panel" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown('<div class="ow-panel-title">SAR IMAGERY OVERVIEW</div>', unsafe_allow_html=True)
            
            grid_row1_col1, grid_row1_col2 = st.columns(2, gap="small")
            grid_row2_col1, grid_row2_col2 = st.columns(2, gap="small")

            with grid_row1_col1:
                f = _make_thumb(sar_disp, cmap="gray")
                st.pyplot(f, width="stretch")
                st.markdown('<div class="thumb-label">SAR Intensity</div>', unsafe_allow_html=True)
                plt.close(f)

            with grid_row1_col2:
                f = _make_thumb(binary_mask, cmap=mcolors.ListedColormap(["#0d1117", "#f85149"]))
                st.pyplot(f, width="stretch")
                st.markdown('<div class="thumb-label">Detected Spill</div>', unsafe_allow_html=True)
                plt.close(f)

            with grid_row2_col1:
                f = _make_thumb(prob_map, cmap="RdYlGn_r")
                st.pyplot(f, width="stretch")
                st.markdown('<div class="thumb-label">Probability Map</div>', unsafe_allow_html=True)
                plt.close(f)

            with grid_row2_col2:
                f3, a3 = plt.subplots(figsize=(3, 3), facecolor="#0d1117")
                a3.set_facecolor("#0d1117")
                a3.imshow(sar_disp, cmap="gray", vmin=0, vmax=1)
                _h2, _w2 = binary_mask.shape
                _ov2 = np.zeros((_h2, _w2, 4), dtype=np.float32)
                _px2 = binary_mask == 1
                _ov2[_px2, 0] = 1.0; _ov2[_px2, 1] = 0.20; _ov2[_px2, 2] = 0.10; _ov2[_px2, 3] = 0.62
                a3.imshow(_ov2, interpolation="nearest")
                a3.axis("off"); f3.tight_layout(pad=0)
                st.pyplot(f3, width="stretch")
                st.markdown('<div class="thumb-label">Overlay</div>', unsafe_allow_html=True)
                plt.close(f3)

            st.markdown('</div>', unsafe_allow_html=True)

        # ---- RIGHT COLUMN: Map + Drift analysis + Environments ----
        with col_right:
            st.markdown('<div class="ow-panel">', unsafe_allow_html=True)

            if spill_det and drift_ok and drift_result is not None and lat_val is not None:
                st.markdown(
                    '<div class="ow-panel-title">\U0001f5fa\ufe0f&nbsp; Map View</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="map-container">', unsafe_allow_html=True)
                
                # Render Map with spill, trajectory, search radius, and AIS vessels
                fmap = _build_drift_map(
                    spill_lat=lat_val,
                    spill_lon=lon_val,
                    origin_lat=drift_result.probable_latitude,
                    origin_lon=drift_result.probable_longitude,
                    traj_pts=drift_result.backward_trajectory,
                    spill_area=area_km2,
                    spill_cov=area_pct,
                    spill_conf=conf_val * 100.0,
                    rel_time_str=drift_result.estimated_release_time.strftime("%Y-%m-%d %H:%M UTC"),
                    n_pts=drift_result.trajectory_points_used,
                    candidate_vessels=ais_result["candidate_vessels"] if (ais_ok and ais_result) else None
                )
                st_folium(fmap, width="100%", height=480, returned_objects=[])
                st.markdown('</div>', unsafe_allow_html=True)

            elif spill_det and lat_val is not None:
                st.markdown(
                    '<div class="ow-panel-title">\U0001f5fa\ufe0f&nbsp; Map View</div>',
                    unsafe_allow_html=True,
                )
                m2 = folium.Map(location=[lat_val, lon_val], zoom_start=10, control_scale=True)
                folium.CircleMarker(
                    location=[lat_val, lon_val], radius=14,
                    color="#ef4444", fill=True, fill_color="#ef4444", fill_opacity=0.30,
                    tooltip=f"Detected Spill ({lat_val:.4f}\u00b0, {lon_val:.4f}\u00b0)",
                ).add_to(m2)
                st.markdown('<div class="map-container">', unsafe_allow_html=True)
                st_folium(m2, width="100%", height=480, returned_objects=[])
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="ow-panel-title">\U0001f5fa\ufe0f&nbsp; Map View</div>',
                    unsafe_allow_html=True,
                )
                st.info("No oil spill detected \u2014 no location to display.")

            st.markdown('</div>', unsafe_allow_html=True)

            # Bottom metrics & Trajectory Chart & Environmental Card
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            col_bottom_left, col_bottom_right = st.columns([3.2, 1.2], gap="medium")

            with col_bottom_left:
                st.markdown('<div class="ow-panel" style="height:100%;">', unsafe_allow_html=True)
                st.markdown('<div class="ow-panel-title">DRIFT &amp; ORIGIN ANALYSIS</div>', unsafe_allow_html=True)

                if drift_ok and drift_result is not None:
                    rel_dt       = drift_result.estimated_release_time
                    rel_date     = rel_dt.strftime("%Y-%m-%d")
                    rel_time     = rel_dt.strftime("%H:%M:%S")
                    n_pts        = drift_result.trajectory_points_used
                    back_hrs     = int(n_pts * 0.5)
                    orig_lat     = drift_result.probable_latitude
                    orig_lon     = drift_result.probable_longitude

                    # Metric Row HTML
                    st.markdown(f"""
                    <div class="drift-metrics-row">
                      <div class="drift-metric-card">
                         <div class="card-icon icon-red">\u25ce</div>
                         <div class="card-title">Detected Position</div>
                         <div class="card-val">{lat_val:.6f}\u00b0 N<br>{lon_val:.6f}\u00b0 E</div>
                      </div>
                      <div class="drift-metric-card">
                         <div class="card-icon icon-green">\u25ce</div>
                         <div class="card-title">Probable Origin</div>
                         <div class="card-val">{orig_lat:.6f}\u00b0 N<br>{orig_lon:.6f}\u00b0 E</div>
                      </div>
                      <div class="drift-metric-card">
                         <div class="card-icon icon-blue">\u23f2</div>
                         <div class="card-title">Estimated Release Time</div>
                         <div class="card-val">{rel_date}<br>{rel_time} UTC</div>
                      </div>
                      <div class="drift-metric-card">
                         <div class="card-icon icon-blue">\u23f1</div>
                         <div class="card-title">Backtracking Duration</div>
                         <div class="card-val">{back_hrs} Hours<br>({n_pts-1} Steps)</div>
                      </div>
                      <div class="drift-metric-card">
                         <div class="card-icon icon-blue">\U0001f5d2\ufe0f</div>
                         <div class="card-title">Trajectory Points</div>
                         <div class="card-val">{n_pts}<br>Points Used</div>
                      </div>
                      <div class="drift-metric-card">
                         <div class="card-icon icon-green">\u2713</div>
                         <div class="card-title">Status</div>
                         <div class="card-val" style="color:#48bb78; font-weight:700;">Origin Estimated</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Trajectory line chart
                    st.markdown('<div class="drift-group-label" style="font-size:0.75rem;margin-bottom:0.4rem;">BACKTRACKED TRAJECTORY (6 HOURS)</div>', unsafe_allow_html=True)
                    fig_chart = _build_trajectory_chart(
                        drift_result.backward_trajectory,
                        spill_lat=lat_val,
                        spill_lon=lon_val,
                        origin_lat=orig_lat,
                        origin_lon=orig_lon
                    )
                    st.pyplot(fig_chart, width="stretch")
                    plt.close(fig_chart)
                    
                    # ---- Member 3: AIS Vessel Analysis UI section ----
                    if ais_ok and ais_result:
                        st.markdown('<div class="drift-divider"></div>', unsafe_allow_html=True)

                        # AIS section header with always-on live badge
                        st.markdown("""
                        <div style='display:flex;align-items:center;gap:10px;margin-bottom:0.6rem;margin-top:0.4rem;'>
                          <div class='ow-panel-title' style='color:#63b3ed;font-size:0.92rem;font-weight:700;border:none;padding:0;margin:0;'>🚢&nbsp; AIS VESSEL ANALYSIS</div>
                          <span class='live-badge'><span class='live-dot'></span>LIVE</span>
                        </div>
                        """, unsafe_allow_html=True)

                        meta = ais_result.get("metadata", {})
                        st.markdown(f"""
                        <div style="font-size:0.75rem; color:#90a4b7; margin-bottom:0.9rem; display:flex; gap:1.5rem; flex-wrap:wrap; background:rgba(255,255,255,0.01); padding:6px 10px; border-radius:4px; border:1px solid rgba(99,179,237,0.05);">
                           <span><strong>AIS Dataset:</strong> sih_demo_ais.csv (Synthetic)</span>
                           <span><strong>Search Radius:</strong> 20.0 km</span>
                           <span><strong>Time Window:</strong> &plusmn;60 min</span>
                           <span><strong>Records Examined:</strong> {meta.get('records_loaded', 0)}</span>
                           <span><strong>Candidates Found:</strong> {meta.get('unique_vessels_found', 0)}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        cands = ais_result.get("candidate_vessels", [])
                        if cands:
                            # Build Table rows dynamically
                            table_rows = ""
                            for idx, cand in enumerate(cands, 1):
                                name_str = cand.get("vessel_name") or "Unknown"
                                mmsi = cand["mmsi"]
                                v_type = cand.get("vessel_type") or "Unknown"
                                dist = cand["minimum_distance_km"]
                                time_diff = cand["time_difference_minutes"]
                                score = cand["candidate_score"]
                                
                                c_lat = cand["latitude"]
                                c_lon = cand["longitude"]

                                table_rows += f"""
                                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                                   <td style="padding: 8px; font-weight:700; color:#63b3ed;">#{idx}</td>
                                   <td style="padding: 8px;"><strong>{mmsi}</strong><br><span style="font-size:0.7rem; color:#90a4b7;">{name_str}</span></td>
                                   <td style="padding: 8px;">{v_type}</td>
                                   <td class="coord-cell" style="padding: 8px; font-family:monospace; color:#93c5fd; font-weight:600;">{c_lat:.6f}&deg; N</td>
                                   <td class="coord-cell" style="padding: 8px; font-family:monospace; color:#93c5fd; font-weight:600;">{c_lon:.6f}&deg; E</td>
                                   <td style="padding: 8px;">{dist:.2f} km</td>
                                   <td style="padding: 8px;">{time_diff:.1f} min</td>
                                   <td style="padding: 8px; text-align: right; color:#48bb78; font-weight:bold;">{score:.1f}/100</td>
                                </tr>
                                """

                            table_html = f"""
                            <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left; color: #e2e8f0;">
                              <thead>
                                <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.15); color:#63b3ed; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.02em;">
                                  <th style="padding: 6px 8px;">Rank</th>
                                  <th style="padding: 6px 8px;">MMSI / Name</th>
                                  <th style="padding: 6px 8px;">Type</th>
                                  <th style="padding: 6px 8px;">Latitude</th>
                                  <th style="padding: 6px 8px;">Longitude</th>
                                  <th style="padding: 6px 8px;">Min Distance</th>
                                  <th style="padding: 6px 8px;">Time Diff</th>
                                  <th style="padding: 6px 8px; text-align: right;">Score</th>
                                </tr>
                              </thead>
                              <tbody>
                                 {table_rows}
                              </tbody>
                            </table>
                            """
                            # Clean newlines and leading spaces so Markdown does not parse as a code block
                            clean_table_html = table_html.replace("\n", "").replace("    ", "").replace("  ", "")
                            st.markdown(clean_table_html, unsafe_allow_html=True)
                        else:
                            st.info("No candidate vessels found within thresholds.")
                    elif ais_err:
                        st.warning(f"⚠️ AIS Vessel Analysis unavailable: {ais_err[:80]}")

                else:
                    st.info("Drift analysis details unavailable.")

                st.markdown('</div>', unsafe_allow_html=True)

            with col_bottom_right:
                # Environmental conditions panel
                st.markdown('<div class="ow-panel" style="height:100%;">', unsafe_allow_html=True)
                st.markdown('<div class="ow-panel-title">ENVIRONMENTAL CONDITIONS</div>', unsafe_allow_html=True)
                
                # Fetch env conditions from the trajectory points
                if drift_ok and drift_result is not None and drift_result.backward_trajectory:
                    latest_pt = drift_result.backward_trajectory[0]
                    wind_speed = latest_pt.wind_speed_mps
                    wind_dir   = latest_pt.wind_direction_deg
                    curr_speed = latest_pt.current_speed_mps
                    curr_dir   = latest_pt.current_direction_deg
                else:
                    wind_speed = 5.2
                    wind_dir   = 135.0
                    curr_speed = 0.35
                    curr_dir   = 210.0

                wind_dir_comp = _deg_to_compass(wind_dir)
                curr_dir_comp = _deg_to_compass(curr_dir)

                st.markdown(f"""
                <div style="font-size:0.72rem; color:#90a4b7; margin-bottom:0.75rem;">
                   Environmental Data<br>
                   <strong style="color:#e2e8f0; font-size:0.8rem;">Synthetic / Prototype</strong>
                </div>
                
                <div class="ow-metric-row" style="padding:0.45rem 0.65rem; margin-bottom:0.35rem;">
                  <span class="ow-metric-key" style="font-size:0.78rem;">\U0001f4a8 Wind Speed (10m)</span>
                  <span class="ow-metric-val" style="font-size:0.78rem;">{wind_speed:.1f} m/s</span>
                </div>
                <div class="ow-metric-row" style="padding:0.45rem 0.65rem; margin-bottom:0.35rem;">
                  <span class="ow-metric-key" style="font-size:0.78rem;">\U0001f9ed Wind Direction</span>
                  <span class="ow-metric-val" style="font-size:0.78rem;">{wind_dir:.0f}\u00b0 {wind_dir_comp}</span>
                </div>
                <div class="ow-metric-row" style="padding:0.45rem 0.65rem; margin-bottom:0.35rem;">
                  <span class="ow-metric-key" style="font-size:0.78rem;">\U0001f30a Current Speed</span>
                  <span class="ow-metric-val" style="font-size:0.78rem;">{curr_speed:.2f} m/s</span>
                </div>
                <div class="ow-metric-row" style="padding:0.45rem 0.65rem; margin-bottom:0.75rem;">
                  <span class="ow-metric-key" style="font-size:0.78rem;">\U0001f9ed Current Direction</span>
                  <span class="ow-metric-val" style="font-size:0.78rem;">{curr_dir:.0f}\u00b0 {curr_dir_comp}</span>
                </div>
                
                <div style="font-size:0.65rem; color:#4a5568; line-height:1.3; font-style:italic; margin-top:1.1rem;">
                  Note: Environmental data is synthetic and for prototype demonstration only.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("""
<div style="margin-top:2.5rem;border-top:1px solid rgba(99,179,237,0.12);padding-top:1.4rem;">
  <div style="display:flex;gap:0;margin-bottom:1.3rem;">
    <div style="flex:1;display:flex;align-items:flex-start;gap:0.85rem;padding:0.9rem 1.1rem;border-right:1px solid rgba(99,179,237,0.07);padding-left:0;">
      <div style="font-size:1.9rem;opacity:0.7;">&#128752;&#65039;</div>
      <div>
        <h4 style="font-size:0.87rem;font-weight:600;color:#63b3ed !important;margin:0 0 0.22rem;">SAR Technology</h4>
        <p style="font-size:0.75rem;color:#90a4b7 !important;margin:0;line-height:1.4;">Sentinel-1 SAR<br>VV Polarization</p>
      </div>
    </div>
    <div style="flex:1;display:flex;align-items:flex-start;gap:0.85rem;padding:0.9rem 1.1rem;border-right:1px solid rgba(99,179,237,0.07);">
      <div style="font-size:1.9rem;opacity:0.7;">&#129504;</div>
      <div>
        <h4 style="font-size:0.87rem;font-weight:600;color:#63b3ed !important;margin:0 0 0.22rem;">AI Powered</h4>
        <p style="font-size:0.75rem;color:#90a4b7 !important;margin:0;line-height:1.4;">U-Net Deep Learning<br>Model</p>
      </div>
    </div>
    <div style="flex:1;display:flex;align-items:flex-start;gap:0.85rem;padding:0.9rem 1.1rem;border-right:1px solid rgba(99,179,237,0.07);">
      <div style="font-size:1.9rem;opacity:0.7;">&#129517;</div>
      <div>
        <h4 style="font-size:0.87rem;font-weight:600;color:#63b3ed !important;margin:0 0 0.22rem;">Drift Analysis</h4>
        <p style="font-size:0.75rem;color:#90a4b7 !important;margin:0;line-height:1.4;">Backward Trajectory<br>Origin Estimation</p>
      </div>
    </div>
    <div style="flex:1;display:flex;align-items:flex-start;gap:0.85rem;padding:0.9rem 1.1rem;">
      <div style="font-size:1.9rem;opacity:0.7;">&#127754;</div>
      <div>
        <h4 style="font-size:0.87rem;font-weight:600;color:#63b3ed !important;margin:0 0 0.22rem;">Marine Monitoring</h4>
        <p style="font-size:0.75rem;color:#90a4b7 !important;margin:0;line-height:1.4;">Real-time Ocean<br>Surveillance</p>
      </div>
    </div>
  </div>
  <div style="text-align:center;color:#5a7a90 !important;font-size:0.78rem;padding-bottom:1.4rem;">
    &copy; 2026 <a href="#" style="color:#63b3ed !important;text-decoration:none;">OceanWatch</a> | Protecting Our Oceans with AI Technology
  </div>
</div>
""", unsafe_allow_html=True)


# ---- Live vessel tracking autorefresh (5s interval to prevent flash) ----
import time as _time
st.session_state.sim_step = (st.session_state.get("sim_step", 0) + 1) % 100
_time.sleep(5)
st.rerun()

