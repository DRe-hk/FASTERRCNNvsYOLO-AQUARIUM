"""
inference/yolo_image.py — Función 1: Detección en imagen estática con YOLO26.

Pipeline:
  1. Cargar imagen desde disco.
  2. Pasar por el modelo YOLO26 y medir tiempo de inferencia.
  3. Dibujar bounding boxes, etiquetas y confianza sobre la imagen.
  4. Retornar imagen anotada + dict de métricas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIDENCE_THRESHOLD, AQUARIUM_CLASSES
from inference.draw_utils import draw_detection_box, draw_detection_count, bgr_to_rgb


# ── Estructura de resultado ───────────────────────────────────────────────

@dataclass
class ImageDetectionResult:
    """Resultado completo de una detección sobre imagen estática."""
    annotated_frame_bgr: np.ndarray           # Imagen BGR con anotaciones
    annotated_frame_rgb: np.ndarray           # Imagen RGB para Streamlit
    inference_ms: float                        # Tiempo de inferencia en ms
    detections: List[Dict[str, Any]] = field(default_factory=list)
    # Cada dict: {"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
    total_detections: int = 0
    image_path: str = ""


# ── Función Principal ─────────────────────────────────────────────────────

def detect_image_yolo(
    yolo_model,
    image_path: str | Path,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> ImageDetectionResult:
    """
    Función 1: Detecta objetos en una imagen estática usando YOLO26.

    Parameters
    ----------
    yolo_model           : Instancia de ultralytics.YOLO ya cargada.
    image_path           : Ruta absoluta o relativa a la imagen (.jpg, .png).
    confidence_threshold : Score mínimo para mostrar una detección.

    Returns
    -------
    ImageDetectionResult
        Imagen anotada en BGR y RGB, tiempo de inferencia y lista de detecciones.

    Raises
    ------
    FileNotFoundError
        Si la imagen no existe en la ruta dada.
    ValueError
        Si el archivo no es una imagen válida.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"❌ Imagen no encontrada: {image_path}")

    # 1. Cargar imagen con OpenCV (BGR)
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise ValueError(f"❌ No se pudo leer la imagen: {image_path}")

    # 2. Inferencia YOLO26 — medir tiempo exacto
    start_time = time.perf_counter()

    results = yolo_model.predict(
        source=frame_bgr,
        conf=confidence_threshold,
        verbose=False,
        stream=False,       # Para imagen única no usamos stream
    )

    end_time = time.perf_counter()
    inference_ms = (end_time - start_time) * 1000.0

    # 3. Extraer detecciones del primer (y único) resultado
    result = results[0]
    detections: List[Dict[str, Any]] = []
    annotated_frame = frame_bgr.copy()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_data = result.boxes

        for i in range(len(boxes_data)):
            # Coordenadas de la caja (formato xyxy)
            xyxy = boxes_data.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Confianza
            confidence = float(boxes_data.conf[i].cpu().numpy())

            # Clase — YOLO usa índices desde 0, sin background
            class_idx = int(boxes_data.cls[i].cpu().numpy())

            # Mapear índice YOLO a nombre de clase
            # YOLO names del modelo puede diferir; usamos el names del modelo si disponible
            if hasattr(yolo_model, "names") and class_idx in yolo_model.names:
                class_name = yolo_model.names[class_idx]
            elif class_idx + 1 < len(AQUARIUM_CLASSES):  # +1 por el background en AQUARIUM_CLASSES
                class_name = AQUARIUM_CLASSES[class_idx + 1]
            else:
                class_name = f"class_{class_idx}"

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

            # 4. Dibujar caja
            draw_detection_box(annotated_frame, x1, y1, x2, y2, class_name, confidence)

    # Dibujar conteo total
    draw_detection_count(annotated_frame, len(detections))

    # Añadir tiempo de inferencia en la imagen
    cv2.putText(
        annotated_frame,
        f"YOLO26  |  {inference_ms:.1f} ms",
        (8, annotated_frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 120), 2, cv2.LINE_AA,
    )

    return ImageDetectionResult(
        annotated_frame_bgr=annotated_frame,
        annotated_frame_rgb=bgr_to_rgb(annotated_frame),
        inference_ms=inference_ms,
        detections=detections,
        total_detections=len(detections),
        image_path=str(image_path),
    )
