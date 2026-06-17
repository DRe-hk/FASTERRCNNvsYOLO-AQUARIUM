"""
models/__init__.py — Paquete de modelos.
Expone las funciones de carga para importación directa.
"""

from .model_loader import load_yolo_model, load_frcnn_model, load_all_models

__all__ = ["load_yolo_model", "load_frcnn_model", "load_all_models"]
