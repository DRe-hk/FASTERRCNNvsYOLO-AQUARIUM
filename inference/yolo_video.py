"""
inference/yolo_video.py — Función 3: Detección en video precargado con YOLO26.

Pipeline:
  1. Abrir video con cv2.VideoCapture.
  2. Leer frame a frame usando el método stream de YOLO26 (optimizado).
  3. Calcular FPS reales con timestamp entre frames.
  4. Dibujar overlay de FPS y tiempo de inferencia en cada frame.
  5. Yieldar frames procesados para consumo por la UI (Streamlit generator).

Diseño de la función como generador (yield):
  - Permite a Streamlit mostrar el video sin congelar la UI.
  - El generador puede detenerse desde la UI llamando .close().
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Generator, NamedTuple

import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIDENCE_THRESHOLD, DISPLAY_WIDTH, DISPLAY_HEIGHT
from inference.draw_utils import (
    draw_detection_box,
    draw_fps_counter,
    draw_detection_count,
    resize_for_display,
    bgr_to_rgb,
)


# ── Estructura de resultado por frame ────────────────────────────────────

class VideoFrameResult(NamedTuple):
    """Resultado de un frame de video procesado."""
    frame_rgb: np.ndarray    # Frame en RGB listo para Streamlit
    fps: float               # FPS real calculado con ventana deslizante
    inference_ms: float      # Tiempo de inferencia de este frame
    frame_index: int         # Número de frame actual
    total_detections: int    # Número de detecciones en este frame


# ── FPS Calculator ────────────────────────────────────────────────────────

class FPSCalculator:
    """Calcula FPS real usando una ventana deslizante de timestamps."""

    def __init__(self, window_size: int = 30):
        self._timestamps: deque = deque(maxlen=window_size)

    def tick(self) -> float:
        """Registra un tick y retorna el FPS promedio de la ventana."""
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ── Función Principal (generador) ─────────────────────────────────────────

def detect_video_yolo(
    yolo_model,
    video_path: str | Path,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_display_width: int = DISPLAY_WIDTH,
    max_display_height: int = DISPLAY_HEIGHT,
) -> Generator[VideoFrameResult, None, None]:
    """
    Función 3: Detecta objetos frame a frame en un video usando YOLO26.

    Implementada como generador para integración sin bloqueo con Streamlit.
    La inferencia usa el modo stream=True de Ultralytics, optimizado para
    flujos de video continuos (menor uso de memoria).

    Parameters
    ----------
    yolo_model           : Instancia ultralytics.YOLO ya cargada.
    video_path           : Ruta al archivo de video (.mp4, .avi, etc.).
    confidence_threshold : Confianza mínima para mostrar detecciones.
    max_display_width    : Ancho máximo para el frame de display.
    max_display_height   : Alto máximo para el frame de display.

    Yields
    ------
    VideoFrameResult
        Un resultado por frame con imagen RGB, FPS, inferencia y detecciones.

    Raises
    ------
    FileNotFoundError
        Si el archivo de video no existe.
    RuntimeError
        Si el video no puede abrirse con OpenCV.

    Example (Streamlit)
    -------------------
    >>> placeholder = st.empty()
    >>> for frame_result in detect_video_yolo(model, "video.mp4"):
    ...     placeholder.image(frame_result.frame_rgb)
    ...     if stop_button:
    ...         break
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"❌ Video no encontrado: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir el video: {video_path}")

    fps_calc = FPSCalculator(window_size=30)
    frame_index = 0

    try:
        # YOLO stream=True: optimizado para flujos — genera resultados lazy
        for result in yolo_model.predict(
            source=str(video_path),
            conf=confidence_threshold,
            stream=True,           # ← clave para video: lazy generation
            verbose=False,
        ):
            # Tiempo de inferencia reportado por YOLO (en ms)
            inference_ms = result.speed.get("inference", 0.0)

            # Frame anotado por YOLO (con plot() que devuelve BGR)
            annotated_bgr = result.plot(
                conf=True,
                labels=True,
                line_width=2,
                font_size=0.5,
            )

            # Calcular FPS real
            fps = fps_calc.tick()

            # Overlay de FPS
            draw_fps_counter(annotated_bgr, fps, model_name="YOLO26", inference_ms=inference_ms)

            # Conteo de detecciones
            n_detections = len(result.boxes) if result.boxes is not None else 0
            draw_detection_count(annotated_bgr, n_detections)

            # Redimensionar para display
            display_frame = resize_for_display(annotated_bgr, max_display_width, max_display_height)

            yield VideoFrameResult(
                frame_rgb=bgr_to_rgb(display_frame),
                fps=fps,
                inference_ms=inference_ms,
                frame_index=frame_index,
                total_detections=n_detections,
            )

            frame_index += 1

    finally:
        cap.release()
