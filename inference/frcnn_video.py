"""
inference/frcnn_video.py — Función 4: Detección en video precargado con Faster R-CNN.

Pipeline optimizado para medir la diferencia de FPS vs YOLO26:
  1. Abrir video con cv2.VideoCapture.
  2. Por cada frame:
     a. Convertir BGR → tensor float32 [0, 1] directamente en memoria (sin PIL).
     b. Enviar tensor al dispositivo (CPU/GPU).
     c. Inferir con torch.no_grad() en modo eval().
     d. Filtrar por umbral de confianza.
     e. Dibujar bounding boxes con OpenCV.
     f. Calcular FPS real con ventana deslizante.
  3. Yieldar frames procesados como generador compatible con Streamlit.

Optimizaciones implementadas:
  - Conversión BGR→tensor in-place con numpy (sin PIL en el loop).
  - torch.no_grad() elimina el grafo de gradientes (ahorra ~50% de memoria).
  - torch.cuda.synchronize() solo cuando hay GPU para timing preciso.
  - Resize pre-inferencia opcional para acelerar en CPU.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Generator, NamedTuple, Optional

import cv2
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIDENCE_THRESHOLD, AQUARIUM_CLASSES, DEVICE, DISPLAY_WIDTH, DISPLAY_HEIGHT
from inference.draw_utils import (
    draw_detection_box,
    draw_fps_counter,
    draw_detection_count,
    resize_for_display,
    bgr_to_rgb,
)


# ── Estructura de resultado por frame ────────────────────────────────────

class VideoFrameResult(NamedTuple):
    """Resultado de un frame de video procesado por Faster R-CNN."""
    frame_rgb: np.ndarray
    fps: float
    inference_ms: float
    frame_index: int
    total_detections: int


# ── FPS Calculator ────────────────────────────────────────────────────────

class FPSCalculator:
    """Calcula FPS real usando una ventana deslizante de timestamps."""

    def __init__(self, window_size: int = 30):
        self._timestamps: deque = deque(maxlen=window_size)

    def tick(self) -> float:
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ── Conversión de frame BGR → tensor optimizada ───────────────────────────

def frame_bgr_to_tensor(frame_bgr: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    """
    Convierte un frame BGR de OpenCV a tensor float32 [0, 1] en el dispositivo.

    Evita PIL Image.open() en el bucle de video (cuello de botella).
    Usa conversiones NumPy nativas para máxima velocidad.

    Parameters
    ----------
    frame_bgr : Frame NumPy BGR (H, W, 3) uint8.
    device    : 'cpu' o 'cuda'.

    Returns
    -------
    torch.Tensor
        Tensor [3, H, W] float32 en [0.0, 1.0] en el device especificado.
    """
    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # HWC → CHW y normalizar a [0, 1]
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


# ── Función Principal (generador) ─────────────────────────────────────────

def detect_video_frcnn(
    frcnn_model: torch.nn.Module,
    video_path: str | Path,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_display_width: int = DISPLAY_WIDTH,
    max_display_height: int = DISPLAY_HEIGHT,
    inference_resize: Optional[tuple[int, int]] = None,
) -> Generator[VideoFrameResult, None, None]:
    """
    Función 4: Detecta objetos frame a frame en un video usando Faster R-CNN.

    Diseñada para maximizar la medición de FPS real y evidenciar la diferencia
    de velocidad vs YOLO26 (arquitectura Two-Stage vs One-Stage).

    Parameters
    ----------
    frcnn_model          : Modelo Faster R-CNN en modo eval().
    video_path           : Ruta al archivo de video (.mp4, .avi, etc.).
    confidence_threshold : Score mínimo para mostrar una detección (filtra ruido).
    max_display_width    : Ancho máximo para el frame de display en Streamlit.
    max_display_height   : Alto máximo para el frame de display.
    inference_resize     : Tuple (W, H) opcional para redimensionar el frame
                           ANTES de la inferencia (acelera en CPU).
                           Si None, usa el frame en su resolución original.

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

    Performance Notes
    -----------------
    - En CPU: ~1-5 FPS típico (Faster R-CNN es computacionalmente costoso).
    - En GPU (CUDA): ~15-30 FPS dependiendo de la GPU.
    - YOLO26 en CPU: ~15-30 FPS para comparación.
    - Usar inference_resize=(640, 480) para acelerar en CPU sin GPU.

    Example (Streamlit)
    -------------------
    >>> placeholder = st.empty()
    >>> for frame_result in detect_video_frcnn(model, "video.mp4"):
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
    use_cuda = DEVICE == "cuda"

    # Asegurar modo eval (por si se pasó un modelo sin llamar a .eval())
    frcnn_model.eval()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # Fin del video

            # ── Paso a: Preparar tensor ──────────────────────────────────
            # Resize previo a inferencia (opcional, acelera en CPU)
            inference_frame = frame_bgr
            if inference_resize is not None:
                inference_frame = cv2.resize(
                    frame_bgr,
                    inference_resize,
                    interpolation=cv2.INTER_LINEAR,
                )

            tensor = frame_bgr_to_tensor(inference_frame, DEVICE)

            # ── Paso b-c: Inferencia medida con precisión ────────────────
            if use_cuda:
                torch.cuda.synchronize()   # Asegurar ops previas completadas
            t0 = time.perf_counter()

            with torch.no_grad():
                predictions = frcnn_model([tensor])

            if use_cuda:
                torch.cuda.synchronize()   # Asegurar que la inferencia terminó
            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000.0

            # ── Paso d: Filtrar por confianza ────────────────────────────
            pred = predictions[0]
            boxes  = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            mask   = scores >= confidence_threshold
            boxes  = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            # ── Paso e: Dibujar sobre el frame ORIGINAL (no el resize) ──
            annotated_frame = frame_bgr.copy()

            # Escalar cajas si se hizo resize para inferencia
            if inference_resize is not None and len(boxes) > 0:
                orig_h, orig_w = frame_bgr.shape[:2]
                inf_w, inf_h   = inference_resize
                scale_x = orig_w / inf_w
                scale_y = orig_h / inf_h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

            n_detections = 0
            for i in range(len(boxes)):
                label_idx  = int(labels[i])
                confidence = float(scores[i])

                if label_idx == 0 or label_idx >= len(AQUARIUM_CLASSES):
                    continue  # Omitir background
                class_name = AQUARIUM_CLASSES[label_idx]

                x1, y1, x2, y2 = boxes[i].astype(int)
                draw_detection_box(annotated_frame, x1, y1, x2, y2, class_name, confidence)
                n_detections += 1

            # ── Paso f: FPS y overlays ───────────────────────────────────
            fps = fps_calc.tick()
            draw_fps_counter(annotated_frame, fps, model_name="Faster R-CNN", inference_ms=inference_ms)
            draw_detection_count(annotated_frame, n_detections)

            # Redimensionar para display en Streamlit
            display_frame = resize_for_display(annotated_frame, max_display_width, max_display_height)

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
