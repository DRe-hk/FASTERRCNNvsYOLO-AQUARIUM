"""
config.py — Configuración global de la aplicación.

Define rutas de modelos, clases del dataset Aquarium,
colores de visualización, umbrales y parámetros de inferencia.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# RUTAS BASE DEL PROYECTO
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Carpeta donde se almacenan los pesos de los modelos
MODELS_DIR = BASE_DIR / "models"

# Carpeta para guardar imágenes y videos de entrada (demos)
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Carpeta para outputs / resultados exportados
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# RUTAS DE PESOS DE LOS MODELOS
# ──────────────────────────────────────────────
YOLO_WEIGHTS_PATH = MODELS_DIR / "best.pt"
FRCNN_WEIGHTS_PATH = MODELS_DIR / "faster_rcnn_aquarium.pth"

# ──────────────────────────────────────────────
# CLASES DEL DATASET AQUARIUM (7 marinas + fondo)
# ──────────────────────────────────────────────
# Índice 0 = fondo (background), índices 1-7 = clases marinas.
# Este orden DEBE coincidir con el orden usado en el entrenamiento.
AQUARIUM_CLASSES = [
    "__background__",  # 0 — Clase de fondo (solo Faster R-CNN)
    "fish",            # 1
    "jellyfish",       # 2
    "penguin",         # 3
    "puffin",          # 4
    "shark",           # 5
    "starfish",        # 6
    "stingray",        # 7
]

# Número total de clases (incluyendo fondo para Faster R-CNN)
NUM_CLASSES_FRCNN = len(AQUARIUM_CLASSES)   # 8  (background + 7 marinas)
NUM_CLASSES_YOLO = NUM_CLASSES_FRCNN - 1    # 7  (YOLO no tiene clase background)

# ──────────────────────────────────────────────
# COLORES POR CLASE (BGR para OpenCV)
# ──────────────────────────────────────────────
CLASS_COLORS_BGR = {
    "fish":      (255, 165,   0),   # Naranja
    "jellyfish": (255,   0, 255),   # Magenta
    "penguin":   ( 50, 205,  50),   # Verde lima
    "puffin":    (  0, 191, 255),   # Azul cielo
    "shark":     (220,  20,  60),   # Rojo carmesí
    "starfish":  (255, 215,   0),   # Dorado
    "stingray":  (138,  43, 226),   # Violeta
}

# Color por defecto para clases no mapeadas
DEFAULT_COLOR_BGR = (200, 200, 200)

# ──────────────────────────────────────────────
# PARÁMETROS DE INFERENCIA
# ──────────────────────────────────────────────
# Umbral de confianza mínimo para mostrar una detección
CONFIDENCE_THRESHOLD = 0.50

# Umbral para NMS (Non-Maximum Suppression) en Faster R-CNN manual
NMS_IOU_THRESHOLD = 0.45

# Resolución de entrada para Faster R-CNN (resize previo al tensor)
FRCNN_INPUT_SIZE = (800, 600)   # (width, height) — opcional, el modelo escala internamente

# ──────────────────────────────────────────────
# PARÁMETROS DE VIDEO
# ──────────────────────────────────────────────
# Máxima resolución de frame para el preview en Streamlit
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Buffer de frames para video en modo split-screen
VIDEO_FRAME_BUFFER = 1  # Sin buffer — procesamiento frame a frame

# ──────────────────────────────────────────────
# OPCIONES DE DISPOSITIVO
# ──────────────────────────────────────────────
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# TEXTOS DE LA UI
# ──────────────────────────────────────────────
APP_TITLE = "🐠 AquaVision — YOLO26 vs Faster R-CNN Benchmark"
APP_ICON = "🔬"
APP_DESCRIPTION = (
    "Compara el rendimiento de dos arquitecturas de detección de objetos "
    "entrenadas sobre el dataset Aquarium: **YOLO26 (One-Stage)** y "
    "**Faster R-CNN ResNet50 FPN V2 (Two-Stage)**."
)
