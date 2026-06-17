"""
inference/__init__.py — Paquete de inferencia.
"""

from .yolo_image  import detect_image_yolo
from .frcnn_image import detect_image_frcnn
from .yolo_video  import detect_video_yolo
from .frcnn_video import detect_video_frcnn

__all__ = [
    "detect_image_yolo",
    "detect_image_frcnn",
    "detect_video_yolo",
    "detect_video_frcnn",
]
