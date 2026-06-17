"""
app.py — AquaVision: Interfaz Streamlit principal.

Estructura de la UI:
  ┌─────────────────────────────────────────────────────────┐
  │  🐠 AquaVision — YOLO26 vs Faster R-CNN Benchmark       │
  ├─────────────────────────────────────────────────────────┤
  │  Sidebar:  Modo | Modelo(s) | Parámetros | Upload       │
  ├──────────────────────┬──────────────────────────────────┤
  │  [YOLO26]            │  [Faster R-CNN]                  │
  │  Imagen / Video      │  Imagen / Video                  │
  │  FPS / Inferencia    │  FPS / Inferencia                │
  └──────────────────────┴──────────────────────────────────┘

Modos de operación:
  1. Imagen Estática  — compara detecciones side-by-side.
  2. Video Secuencial — reproduce un modelo a la vez con métricas.
  3. Split-Screen     — (experimental) ambos modelos en paralelo*.

* El modo split-screen en paralelo real requiere threading; se implementa
  de forma pseudo-paralela intercalando frames. Ver NOTA_THREADING.md.

Ejecutar con:
  streamlit run app.py
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# ── Configuración de la página (DEBE ser el primer comando Streamlit) ──────
st.set_page_config(
    page_title="AquaVision — Benchmark",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    APP_TITLE, APP_DESCRIPTION, CONFIDENCE_THRESHOLD,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, DEVICE, AQUARIUM_CLASSES,
)
from models.model_loader import load_yolo_model, load_frcnn_model
from inference.yolo_image   import detect_image_yolo
from inference.frcnn_image  import detect_image_frcnn
from inference.yolo_video   import detect_video_yolo
from inference.frcnn_video  import detect_video_frcnn


# ══════════════════════════════════════════════════════════════════════════
# ESTILOS CSS PERSONALIZADOS
# ══════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    /* ── OCULTAR ELEMENTOS NATIVOS DE STREAMLIT ────────────────────── */
    /* Barra blanca superior (toolbar: deploy, hamburger, etc.) */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    /* Footer de Streamlit */
    footer { display: none !important; }
    /* Menú hamburguesa flotante */
    #MainMenu { display: none !important; }
    /* Botón de deploy */
    [data-testid="stToolbar"] { display: none !important; }
    /* Padding que deja el header oculto */
    .block-container {
        padding-top: 1.5rem !important;
    }

    /* ── TIPOGRAFÍA GLOBAL ─────────────────────────────────────────── */
    html, body {
        font-family: 'Inter', sans-serif;
    }

    /* ── FONDO PRINCIPAL OSCURO ────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #091521 100%) !important;
    }
    /* Color de texto base de la app */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp div {
        color: #e2e8f0;
    }
    /* Markdown h1-h3 en la zona principal */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: #f1f5f9 !important;
    }

    /* ── SIDEBAR — FONDO ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080f1e 0%, #0a1628 100%) !important;
        border-right: 1px solid rgba(0, 180, 255, 0.2);
    }

    /* Texto base: párrafos, spans y markdown de la sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span:not([data-baseweb="tag"] span),
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }

    /* Títulos de sección */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] strong {
        color: #f1f5f9 !important;
    }

    /* Labels de todos los widgets */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* ── RADIO ──────────────────────────────────────────────────────── */
    /* Texto de cada opción del radio */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color: #e2e8f0 !important;
    }
    /* Opción seleccionada */
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div p {
        color: #67e8f9 !important;
        font-weight: 600 !important;
    }

    /* ── MULTISELECT ────────────────────────────────────────────────── */
    /* Caja de control (fondo oscuro para que el texto sea visible) */
    [data-testid="stSidebar"] [data-baseweb="select"] > div:first-child,
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="control"] {
        background-color: #0d1e35 !important;
        border-color: rgba(0, 180, 255, 0.3) !important;
    }
    /* Placeholder y texto dentro del input */
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="placeholder"],
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="single-value"] {
        color: #94a3b8 !important;
        background: transparent !important;
    }
    /* Flecha desplegable */
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #94a3b8 !important;
    }
    /* Chips/tags de opciones seleccionadas */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background: rgba(0, 180, 255, 0.18) !important;
        border: 1px solid rgba(0, 180, 255, 0.45) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #67e8f9 !important;
    }
    /* X de borrar chip */
    [data-testid="stSidebar"] [data-baseweb="tag"] [role="presentation"] svg {
        fill: #67e8f9 !important;
    }
    /* Menú desplegable del multiselect */
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="menu"] {
        background: #0d1b2a !important;
        border: 1px solid rgba(0, 180, 255, 0.25) !important;
    }
    [data-baseweb="option"] {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {
        background: rgba(0, 180, 255, 0.15) !important;
        color: #67e8f9 !important;
    }

    /* ── SLIDER ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stSlider"] p {
        color: #67e8f9 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: #00b4ff !important;
    }

    /* ── FILE UPLOADER ──────────────────────────────────────────────── */
    /* Zona de drop */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(0, 180, 255, 0.04) !important;
        border: 1.5px dashed rgba(0, 180, 255, 0.35) !important;
        border-radius: 10px !important;
    }
    /* Texto dentro del dropzone */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
        color: #94a3b8 !important;
    }
    /* Botón "Browse files" */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: rgba(0, 180, 255, 0.12) !important;
        border: 1px solid rgba(0, 180, 255, 0.4) !important;
        color: #67e8f9 !important;
        border-radius: 8px !important;
        padding: 0.35rem 1rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background: rgba(0, 180, 255, 0.22) !important;
    }

    /* ── TOGGLE ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stToggle"] p {
        color: #cbd5e1 !important;
    }

    /* ── EXPANDER ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] details summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #94a3b8 !important;
        background: rgba(255,255,255,0.03) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] details summary:hover {
        color: #67e8f9 !important;
    }

    /* ── ALERTS en sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(0, 180, 255, 0.06) !important;
        border-left-color: #00b4ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #94a3b8 !important;
    }

    /* ── HEADER DE LA APP ──────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 0.8rem 0 0.3rem 0;
        background: linear-gradient(90deg, #00b4ff, #7c3aed, #00b4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -0.5px;
    }

    /* ── TARJETAS DE MODELO ────────────────────────────────────────── */
    .model-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0, 180, 255, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
    }
    .model-card-yolo  { border-color: rgba(0, 255, 120, 0.35); }
    .model-card-frcnn { border-color: rgba(0, 120, 255, 0.35); }

    /* ── BADGES DE MÉTRICAS ────────────────────────────────────────── */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 180, 255, 0.12);
        border: 1px solid rgba(0, 180, 255, 0.3);
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #67e8f9;
        margin: 4px 4px 4px 0;
    }
    .badge-yolo  { color: #4ade80 !important; border-color: rgba(74, 222, 128, 0.4) !important; background: rgba(74, 222, 128, 0.08) !important; }
    .badge-frcnn { color: #60a5fa !important; border-color: rgba(96, 165, 250, 0.4) !important; background: rgba(96, 165, 250, 0.08) !important; }

    /* ── SEPARADOR ─────────────────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1rem 0;
    }

    /* ── IMÁGENES ──────────────────────────────────────────────────── */
    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── BOTONES PRINCIPALES ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #0066ff, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(0, 100, 255, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 100, 255, 0.45) !important;
    }

    /* ── TABS ──────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 180, 255, 0.2) !important;
        color: #67e8f9 !important;
    }

    /* ── CAJAS DE INFO ─────────────────────────────────────────────── */
    .info-box {
        background: rgba(0, 180, 255, 0.06);
        border-left: 3px solid #00b4ff;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS CON CACHÉ DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_yolo_model():
    """Carga y cachea el modelo YOLO26 para toda la sesión."""
    return load_yolo_model()


@st.cache_resource(show_spinner=False)
def get_frcnn_model():
    """Carga y cachea el modelo Faster R-CNN para toda la sesión."""
    return load_frcnn_model()


# ══════════════════════════════════════════════════════════════════════════
# COMPONENTES DE UI
# ══════════════════════════════════════════════════════════════════════════

def render_header():
    """Renderiza el header principal de la app."""
    st.markdown('<h1 class="app-header">🔬 AquaVision</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="text-align:center; color:#64748b; font-size:0.95rem; margin-top:-0.5rem;">'
        f'{APP_DESCRIPTION}</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def render_metric_badges(model_name: str, inference_ms: float, fps: Optional[float] = None, n_det: int = 0):
    """Renderiza badges de métricas debajo de un resultado."""
    css_cls = "badge-yolo" if "YOLO" in model_name else "badge-frcnn"
    badges = [
        f'<span class="metric-badge {css_cls}">⚡ {inference_ms:.1f} ms</span>',
        f'<span class="metric-badge {css_cls}">🎯 {n_det} detecciones</span>',
    ]
    if fps is not None:
        badges.append(f'<span class="metric-badge {css_cls}">📹 {fps:.1f} FPS</span>')
    st.markdown(" ".join(badges), unsafe_allow_html=True)


def render_sidebar() -> dict:
    """
    Renderiza la barra lateral y retorna la configuración seleccionada.

    Returns
    -------
    dict con claves:
        mode          : 'image' | 'video'
        models        : list de 'yolo' | 'frcnn'
        conf_threshold: float
        uploaded_file : UploadedFile | None
        split_screen  : bool
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Modo de entrada
        st.markdown("### 📂 Tipo de entrada")
        mode = st.radio(
            "Selecciona el modo:",
            options=["image", "video"],
            format_func=lambda x: "🖼️  Imagen Estática" if x == "image" else "🎬  Video Precargado",
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Selección de modelo(s)
        st.markdown("### 🤖 Modelos")
        model_options = st.multiselect(
            "Selecciona uno o ambos modelos:",
            options=["yolo", "frcnn"],
            default=["yolo", "frcnn"],
            format_func=lambda x: "🟢 YOLO26 (One-Stage)" if x == "yolo" else "🔵 Faster R-CNN (Two-Stage)",
        )

        if not model_options:
            st.warning("⚠️ Selecciona al menos un modelo.")
            model_options = ["yolo"]

        # Split-screen solo disponible si ambos modelos están activos
        split_screen = False
        if len(model_options) == 2 and mode == "video":
            split_screen = st.toggle("🔀 Split-Screen (comparación lado a lado)", value=False)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Umbral de confianza
        st.markdown("### 🎯 Parámetros de inferencia")
        conf_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.05,
            max_value=0.95,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Score mínimo para mostrar una detección. Valores bajos = más detecciones (posible ruido).",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Upload de archivo
        st.markdown("### 📤 Cargar archivo")
        if mode == "image":
            uploaded_file = st.file_uploader(
                "Sube una imagen",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Formatos soportados: JPG, PNG, BMP, WEBP",
            )
        else:
            uploaded_file = st.file_uploader(
                "Sube un video",
                type=["mp4", "avi", "mov", "mkv"],
                help="Formatos soportados: MP4, AVI, MOV, MKV",
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Info del dispositivo
        device_icon = "⚡ GPU" if DEVICE == "cuda" else "💻 CPU"
        st.markdown(
            f'<div class="info-box">Dispositivo de inferencia:<br>'
            f'<strong style="color:#67e8f9;">{device_icon} ({DEVICE.upper()})</strong></div>',
            unsafe_allow_html=True,
        )

        # Clases del dataset
        with st.expander("📋 Clases del dataset"):
            for cls in AQUARIUM_CLASSES[1:]:  # Omitir background
                st.markdown(f"• `{cls}`")

    return {
        "mode": mode,
        "models": model_options,
        "conf_threshold": conf_threshold,
        "uploaded_file": uploaded_file,
        "split_screen": split_screen,
    }


# ══════════════════════════════════════════════════════════════════════════
# LÓGICA DE IMAGEN ESTÁTICA
# ══════════════════════════════════════════════════════════════════════════

def run_image_mode(config: dict, yolo_model, frcnn_model):
    """
    Modo Imagen: ejecuta la detección y muestra resultados side-by-side.
    """
    uploaded_file = config["uploaded_file"]
    conf = config["conf_threshold"]
    models = config["models"]

    if uploaded_file is None:
        st.info("👆 Sube una imagen en el panel lateral para comenzar.")
        return

    # Guardar imagen temporal
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Mostrar imagen original
    st.markdown("#### 📷 Imagen original")
    st.image(uploaded_file, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Resultados de detección")

    # Columnas según número de modelos seleccionados
    cols = st.columns(len(models))

    for i, model_key in enumerate(models):
        with cols[i]:
            if model_key == "yolo":
                st.markdown(
                    '<div class="model-card model-card-yolo">',
                    unsafe_allow_html=True,
                )
                st.markdown("##### 🟢 YOLO26 (One-Stage)")
                with st.spinner("Procesando con YOLO26..."):
                    result = detect_image_yolo(yolo_model, tmp_path, conf)
                st.image(result.annotated_frame_rgb, use_container_width=True)
                render_metric_badges("YOLO26", result.inference_ms, n_det=result.total_detections)
                st.markdown("</div>", unsafe_allow_html=True)

                # Tabla de detecciones
                if result.detections:
                    with st.expander("📊 Ver detecciones YOLO26"):
                        import pandas as pd
                        df = pd.DataFrame(result.detections)
                        df["confidence"] = df["confidence"].map("{:.1%}".format)
                        st.dataframe(df[["class", "confidence"]], use_container_width=True)

            elif model_key == "frcnn":
                st.markdown(
                    '<div class="model-card model-card-frcnn">',
                    unsafe_allow_html=True,
                )
                st.markdown("##### 🔵 Faster R-CNN (Two-Stage)")
                with st.spinner("Procesando con Faster R-CNN..."):
                    result = detect_image_frcnn(frcnn_model, tmp_path, conf)
                st.image(result.annotated_frame_rgb, use_container_width=True)
                render_metric_badges("Faster R-CNN", result.inference_ms, n_det=result.total_detections)
                st.markdown("</div>", unsafe_allow_html=True)

                if result.detections:
                    with st.expander("📊 Ver detecciones Faster R-CNN"):
                        import pandas as pd
                        df = pd.DataFrame(result.detections)
                        df["confidence"] = df["confidence"].map("{:.1%}".format)
                        st.dataframe(df[["class", "confidence"]], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# LÓGICA DE VIDEO — UN MODELO
# ══════════════════════════════════════════════════════════════════════════

def run_single_video(config: dict, model_key: str, yolo_model, frcnn_model):
    """
    Reproduce el video con UN modelo seleccionado.
    """
    uploaded_file = config["uploaded_file"]
    conf = config["conf_threshold"]

    if uploaded_file is None:
        st.info("👆 Sube un video en el panel lateral para comenzar.")
        return

    # Guardar video temporal
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    model_label = "YOLO26 (One-Stage)" if model_key == "yolo" else "Faster R-CNN (Two-Stage)"
    css_cls = "model-card-yolo" if model_key == "yolo" else "model-card-frcnn"
    icon = "🟢" if model_key == "yolo" else "🔵"

    st.markdown(
        f'<div class="model-card {css_cls}"><h5>{icon} {model_label}</h5></div>',
        unsafe_allow_html=True,
    )

    # Controles de video
    col_start, col_stop, _ = st.columns([1, 1, 3])
    start_btn = col_start.button("▶️ Iniciar", key=f"start_{model_key}")
    stop_btn  = col_stop.button("⏹️ Detener", key=f"stop_{model_key}")

    if "video_running" not in st.session_state:
        st.session_state.video_running = False

    if start_btn:
        st.session_state.video_running = True
    if stop_btn:
        st.session_state.video_running = False

    if not st.session_state.video_running:
        st.info("Presiona ▶️ Iniciar para comenzar la detección.")
        return

    # Placeholders de UI
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Seleccionar generador
    if model_key == "yolo":
        frame_gen = detect_video_yolo(yolo_model, tmp_path, conf)
    else:
        frame_gen = detect_video_frcnn(frcnn_model, tmp_path, conf)

    for frame_result in frame_gen:
        if not st.session_state.get("video_running", True):
            break

        frame_placeholder.image(
            frame_result.frame_rgb,
            use_container_width=True,
            caption=f"Frame #{frame_result.frame_index}",
        )
        with metrics_placeholder.container():
            render_metric_badges(
                model_label,
                frame_result.inference_ms,
                fps=frame_result.fps,
                n_det=frame_result.total_detections,
            )

    st.session_state.video_running = False
    st.success("✅ Video finalizado.")


# ══════════════════════════════════════════════════════════════════════════
# LÓGICA DE VIDEO — SPLIT-SCREEN (pseudo-paralelo)
# ══════════════════════════════════════════════════════════════════════════

def run_split_screen_video(config: dict, yolo_model, frcnn_model):
    """
    Modo Split-Screen: reproduce el video con ambos modelos en columnas
    intercalando frames para simular procesamiento paralelo.

    Nota: Streamlit no soporta threading nativo para UI. Esta implementación
    intercala un frame de YOLO y luego uno de F-RCNN, lo que produce FPS
    divididos por 2 para cada modelo pero permite comparar visualmente.
    """
    uploaded_file = config["uploaded_file"]
    conf = config["conf_threshold"]

    if uploaded_file is None:
        st.info("👆 Sube un video en el panel lateral para comenzar.")
        return

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_yolo, \
         tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_frcnn:
        data = uploaded_file.getvalue()
        tmp_yolo.write(data)
        tmp_frcnn.write(data)
        yolo_path  = tmp_yolo.name
        frcnn_path = tmp_frcnn.name

    # UI
    col_start, col_stop, _ = st.columns([1, 1, 3])
    start_btn = col_start.button("▶️ Iniciar Split-Screen", key="split_start")
    stop_btn  = col_stop.button("⏹️ Detener", key="split_stop")

    if "split_running" not in st.session_state:
        st.session_state.split_running = False

    if start_btn:
        st.session_state.split_running = True
    if stop_btn:
        st.session_state.split_running = False

    if not st.session_state.split_running:
        st.info("Presiona ▶️ Iniciar Split-Screen para la comparación.")
        return

    col_y, col_f = st.columns(2)

    with col_y:
        st.markdown("#### 🟢 YOLO26")
        yolo_frame_ph   = st.empty()
        yolo_metrics_ph = st.empty()

    with col_f:
        st.markdown("#### 🔵 Faster R-CNN")
        frcnn_frame_ph   = st.empty()
        frcnn_metrics_ph = st.empty()

    yolo_gen  = detect_video_yolo(yolo_model, yolo_path, conf)
    frcnn_gen = detect_video_frcnn(frcnn_model, frcnn_path, conf)

    yolo_done  = False
    frcnn_done = False

    while not (yolo_done and frcnn_done):
        if not st.session_state.get("split_running", True):
            break

        # Frame YOLO
        if not yolo_done:
            try:
                yolo_res = next(yolo_gen)
                yolo_frame_ph.image(
                    yolo_res.frame_rgb,
                    use_container_width=True,
                    caption=f"Frame #{yolo_res.frame_index}",
                )
                with yolo_metrics_ph.container():
                    render_metric_badges("YOLO26", yolo_res.inference_ms, yolo_res.fps, yolo_res.total_detections)
            except StopIteration:
                yolo_done = True

        # Frame Faster R-CNN
        if not frcnn_done:
            try:
                frcnn_res = next(frcnn_gen)
                frcnn_frame_ph.image(
                    frcnn_res.frame_rgb,
                    use_container_width=True,
                    caption=f"Frame #{frcnn_res.frame_index}",
                )
                with frcnn_metrics_ph.container():
                    render_metric_badges("Faster R-CNN", frcnn_res.inference_ms, frcnn_res.fps, frcnn_res.total_detections)
            except StopIteration:
                frcnn_done = True

    st.session_state.split_running = False
    st.success("✅ Split-Screen finalizado.")


# ══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def main():
    inject_custom_css()
    render_header()

    # Obtener configuración del sidebar
    config = render_sidebar()

    # Cargar modelos (con spinner durante la primera carga)
    yolo_model, frcnn_model = None, None

    if "yolo" in config["models"]:
        with st.spinner("⏳ Cargando YOLO26..."):
            try:
                yolo_model = get_yolo_model()
            except Exception as e:
                st.error(f"❌ Error cargando YOLO26: {e}")

    if "frcnn" in config["models"]:
        with st.spinner("⏳ Cargando Faster R-CNN..."):
            try:
                frcnn_model = get_frcnn_model()
            except Exception as e:
                st.error(f"❌ Error cargando Faster R-CNN: {e}")

    # ── Routing por modo ──────────────────────────────────────────────────
    if config["mode"] == "image":
        run_image_mode(config, yolo_model, frcnn_model)

    elif config["mode"] == "video":
        if config["split_screen"] and len(config["models"]) == 2:
            run_split_screen_video(config, yolo_model, frcnn_model)
        elif len(config["models"]) == 1:
            run_single_video(config, config["models"][0], yolo_model, frcnn_model)
        else:
            # Dos modelos pero sin split-screen: tabs
            tab_y, tab_f = st.tabs(["🟢 YOLO26", "🔵 Faster R-CNN"])
            with tab_y:
                run_single_video({**config, "models": ["yolo"]}, "yolo", yolo_model, frcnn_model)
            with tab_f:
                run_single_video({**config, "models": ["frcnn"]}, "frcnn", yolo_model, frcnn_model)


if __name__ == "__main__":
    main()
