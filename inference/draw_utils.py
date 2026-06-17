"""
inference/draw_utils.py — Utilidades de visualización compartidas.

Funciones para dibujar bounding boxes, etiquetas y métricas
sobre imágenes NumPy/OpenCV para ambos modelos.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CLASS_COLORS_BGR, DEFAULT_COLOR_BGR, AQUARIUM_CLASSES


# ── Fuente de texto OpenCV ────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.55
FONT_SCALE_FPS   = 0.75
FONT_THICKNESS   = 2
BOX_THICKNESS    = 2


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Retorna el color BGR asociado a la clase dada."""
    return CLASS_COLORS_BGR.get(class_name, DEFAULT_COLOR_BGR)


def draw_detection_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    class_name: str,
    confidence: float,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Dibuja una bounding box con etiqueta y confianza sobre un frame.

    Parameters
    ----------
    frame       : Imagen NumPy en formato BGR (H, W, 3).
    x1,y1,x2,y2: Coordenadas de la caja en píxeles.
    class_name  : Nombre de la clase detectada.
    confidence  : Score de confianza [0.0 – 1.0].
    color       : Color BGR opcional. Si None, se usa el mapeado por clase.

    Returns
    -------
    np.ndarray
        Frame con la caja y etiqueta dibujadas (in-place modificado).
    """
    color = color or get_class_color(class_name)
    label = f"{class_name}  {confidence:.1%}"

    # Caja principal
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Fondo de la etiqueta
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
    label_y1 = max(y1 - text_h - baseline - 4, 0)
    cv2.rectangle(
        frame,
        (x1, label_y1),
        (x1 + text_w + 4, y1),
        color,
        cv2.FILLED,
    )

    # Texto de etiqueta en blanco
    cv2.putText(
        frame, label,
        (x1 + 2, y1 - baseline - 2),
        FONT, FONT_SCALE_LABEL,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    return frame


def draw_fps_counter(
    frame: np.ndarray,
    fps: float,
    model_name: str = "",
    inference_ms: Optional[float] = None,
) -> np.ndarray:
    """
    Dibuja el contador de FPS (y opcionalmente el tiempo de inferencia)
    en la esquina superior izquierda del frame.

    Parameters
    ----------
    frame        : Frame NumPy BGR.
    fps          : Valor de FPS actual.
    model_name   : Nombre del modelo (ej. 'YOLO26' o 'Faster R-CNN').
    inference_ms : Tiempo de inferencia en milisegundos (opcional).

    Returns
    -------
    np.ndarray
        Frame con el overlay de métricas dibujado.
    """
    lines = [f"{model_name} | FPS: {fps:.1f}"]
    if inference_ms is not None:
        lines.append(f"Inference: {inference_ms:.1f} ms")

    padding = 8
    line_height = 28
    overlay_h = line_height * len(lines) + padding * 2
    overlay_w = 260

    # Panel semitransparente
    panel = frame[0:overlay_h, 0:overlay_w].copy()
    cv2.rectangle(panel, (0, 0), (overlay_w, overlay_h), (15, 15, 15), cv2.FILLED)
    frame[0:overlay_h, 0:overlay_w] = cv2.addWeighted(
        frame[0:overlay_h, 0:overlay_w], 0.3, panel, 0.7, 0
    )

    for i, line in enumerate(lines):
        y_pos = padding + (i + 1) * line_height - 6
        cv2.putText(
            frame, line, (padding, y_pos),
            FONT, FONT_SCALE_FPS,
            (0, 255, 120),  # Verde neón
            FONT_THICKNESS, cv2.LINE_AA,
        )

    return frame


def draw_detection_count(frame: np.ndarray, count: int) -> np.ndarray:
    """
    Dibuja el conteo total de detecciones en la esquina superior derecha.
    """
    text = f"Det: {count}"
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE_FPS, FONT_THICKNESS)
    x = w - tw - 12
    y = th + 12
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE_FPS, (255, 200, 0), FONT_THICKNESS, cv2.LINE_AA)
    return frame


def resize_for_display(frame: np.ndarray, max_width: int = 640, max_height: int = 480) -> np.ndarray:
    """
    Redimensiona un frame manteniendo el aspect ratio para visualización en UI.
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convierte un frame BGR (OpenCV) a RGB (Streamlit/PIL)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
