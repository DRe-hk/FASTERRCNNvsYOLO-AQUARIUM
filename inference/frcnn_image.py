"""
inference/frcnn_image.py — Función 2: Detección en imagen estática con Faster R-CNN.

Pipeline:
  1. Cargar imagen desde disco con PIL.
  2. Convertir a tensor float32 normalizado en [0, 1].
  3. Pasar por el modelo en modo eval() con torch.no_grad().
  4. Filtrar detecciones por umbral de confianza.
  5. Dibujar bounding boxes, etiquetas y confianza sobre la imagen.
  6. Retornar imagen anotada + dict de métricas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIDENCE_THRESHOLD, AQUARIUM_CLASSES, DEVICE
from inference.draw_utils import draw_detection_box, draw_detection_count, bgr_to_rgb


# ── Estructura de resultado (reutiliza la misma de YOLO para coherencia) ──

@dataclass
class ImageDetectionResult:
    """Resultado de detección en imagen estática para Faster R-CNN."""
    annotated_frame_bgr: np.ndarray
    annotated_frame_rgb: np.ndarray
    inference_ms: float
    detections: List[Dict[str, Any]] = field(default_factory=list)
    total_detections: int = 0
    image_path: str = ""


# ── Transformación de imagen a tensor ────────────────────────────────────

def image_to_tensor(image_path: str | Path) -> tuple[torch.Tensor, np.ndarray]:
    """
    Carga una imagen y la convierte en tensor float32 normalizado en [0, 1].

    Parameters
    ----------
    image_path : Ruta a la imagen (.jpg, .png).

    Returns
    -------
    Tuple[torch.Tensor, np.ndarray]
        (tensor con shape [1, 3, H, W], frame BGR original para dibujar)
    """
    pil_img = Image.open(str(image_path)).convert("RGB")
    # Convertir PIL → tensor [3, H, W] con valores en [0.0, 1.0]
    tensor = TF.to_tensor(pil_img)
    # Faster R-CNN espera una lista de tensores
    tensor = tensor.to(DEVICE)
    # También retornamos el frame BGR para dibujar con OpenCV
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return tensor, frame_bgr


# ── Función Principal ─────────────────────────────────────────────────────

def detect_image_frcnn(
    frcnn_model: torch.nn.Module,
    image_path: str | Path,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> ImageDetectionResult:
    """
    Función 2: Detecta objetos en una imagen estática usando Faster R-CNN.

    Parameters
    ----------
    frcnn_model          : Instancia de Faster R-CNN ya cargada y en modo eval().
    image_path           : Ruta absoluta o relativa a la imagen (.jpg, .png).
    confidence_threshold : Score mínimo para incluir una detección (filtra ruido).

    Returns
    -------
    ImageDetectionResult
        Imagen anotada en BGR y RGB, tiempo de inferencia y lista de detecciones.

    Raises
    ------
    FileNotFoundError
        Si la imagen no existe.
    ValueError
        Si el archivo no es una imagen válida.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"❌ Imagen no encontrada: {image_path}")

    # 1. Cargar imagen → tensor float [0, 1] y frame BGR para dibujo
    tensor, frame_bgr = image_to_tensor(image_path)

    # 2. Inferencia con torch.no_grad() para eficiencia (sin gradientes)
    #    El modelo debe estar en modo .eval() (garantizado por load_frcnn_model)
    start_time = time.perf_counter()

    with torch.no_grad():
        # Faster R-CNN espera una lista de tensores [C, H, W]
        predictions = frcnn_model([tensor])

    end_time = time.perf_counter()
    inference_ms = (end_time - start_time) * 1000.0

    # 3. Extraer resultado del único elemento de la lista
    pred = predictions[0]
    boxes   = pred["boxes"].cpu().numpy()     # shape: [N, 4]  xyxy
    labels  = pred["labels"].cpu().numpy()    # shape: [N]     int
    scores  = pred["scores"].cpu().numpy()    # shape: [N]     float

    # 4. Filtrar por umbral de confianza
    mask = scores >= confidence_threshold
    boxes   = boxes[mask]
    labels  = labels[mask]
    scores  = scores[mask]

    # 5. Dibujar detecciones
    detections: List[Dict[str, Any]] = []
    annotated_frame = frame_bgr.copy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        label_idx = int(labels[i])
        confidence = float(scores[i])

        # Mapear índice de clase → nombre (índice 0 = background, omitir)
        if label_idx == 0 or label_idx >= len(AQUARIUM_CLASSES):
            continue   # Ignorar clase background o índices fuera de rango
        class_name = AQUARIUM_CLASSES[label_idx]

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

        draw_detection_box(annotated_frame, x1, y1, x2, y2, class_name, confidence)

    # Conteo total y marca del modelo
    draw_detection_count(annotated_frame, len(detections))
    cv2.putText(
        annotated_frame,
        f"Faster R-CNN  |  {inference_ms:.1f} ms",
        (8, annotated_frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 120, 255), 2, cv2.LINE_AA,
    )

    return ImageDetectionResult(
        annotated_frame_bgr=annotated_frame,
        annotated_frame_rgb=bgr_to_rgb(annotated_frame),
        inference_ms=inference_ms,
        detections=detections,
        total_detections=len(detections),
        image_path=str(image_path),
    )
