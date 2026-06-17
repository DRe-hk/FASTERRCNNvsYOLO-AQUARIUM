"""
models/model_loader.py — Carga centralizada de arquitecturas de modelos.

Responsabilidades:
- Cargar YOLO26 desde pesos nativos de Ultralytics (.pt).
- Instanciar y cargar Faster R-CNN ResNet50 FPN V2 desde state_dict (.pth).
- Exponer funciones de carga con manejo de errores y mensajes claros.
- Cachear los modelos en sesión (evitar recarga redundante en Streamlit).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

# ── Agregar raíz del proyecto al path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    YOLO_WEIGHTS_PATH,
    FRCNN_WEIGHTS_PATH,
    NUM_CLASSES_FRCNN,
    DEVICE,
)


# ═══════════════════════════════════════════════════════════════════════════
# CARGA DE YOLO26 (Ultralytics)
# ═══════════════════════════════════════════════════════════════════════════

def load_yolo_model():
    """
    Carga el modelo YOLO26 desde el archivo de pesos 'best.pt'.

    Returns
    -------
    ultralytics.YOLO
        Instancia del modelo lista para inferencia.

    Raises
    ------
    FileNotFoundError
        Si el archivo de pesos no existe en la ruta configurada.
    ImportError
        Si el paquete `ultralytics` no está instalado.
    """
    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"❌ No se encontró el archivo de pesos YOLO26 en:\n"
            f"   {YOLO_WEIGHTS_PATH}\n"
            f"   Asegúrate de copiar 'best.pt' desde 'yolo_produccion_final/' "
            f"a la carpeta 'models/'."
        )

    try:
        from ultralytics import YOLO  # Importación diferida para evitar error si no está instalado
    except ImportError as exc:
        raise ImportError(
            "❌ El paquete 'ultralytics' no está instalado.\n"
            "   Ejecuta: pip install ultralytics"
        ) from exc

    print(f"[YOLO26] Cargando pesos desde: {YOLO_WEIGHTS_PATH}")
    model = YOLO(str(YOLO_WEIGHTS_PATH))
    model.to(DEVICE)
    print(f"[YOLO26] ✅ Modelo cargado correctamente en dispositivo: {DEVICE.upper()}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# CARGA DE FASTER R-CNN (Torchvision)
# ═══════════════════════════════════════════════════════════════════════════

def _build_frcnn_architecture() -> nn.Module:
    """
    Instancia la arquitectura Faster R-CNN ResNet50 FPN V2 y ajusta
    el box_predictor para el número de clases del dataset Aquarium.

    La arquitectura se inicializa SIN pesos preentrenados (weights=None)
    porque se cargarán los pesos propios desde el .pth.

    Returns
    -------
    torch.nn.Module
        Modelo con la cabeza de clasificación ajustada a NUM_CLASSES_FRCNN clases.
    """
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:
        raise ImportError(
            "❌ El paquete 'torchvision' no está instalado.\n"
            "   Consulta https://pytorch.org/get-started/locally/ para instalarlo."
        ) from exc

    # 1. Instanciar arquitectura sin pesos preentrenados
    model = fasterrcnn_resnet50_fpn_v2(weights=None)

    # 2. Reemplazar el box_predictor para ajustar el número de clases
    #    in_features: dimensión de entrada del clasificador original
    in_features: int = model.roi_heads.box_predictor.cls_score.in_features

    # 3. Nuevo predictor: NUM_CLASSES_FRCNN clases (background + 7 marinas)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES_FRCNN)

    return model


def load_frcnn_model() -> nn.Module:
    """
    Carga el modelo Faster R-CNN con los pesos entrenados sobre Aquarium.

    Pasos:
    1. Construye la arquitectura ResNet50 FPN V2.
    2. Modifica el box_predictor para NUM_CLASSES_FRCNN clases.
    3. Carga el state_dict desde 'faster_rcnn_aquarium.pth'.
    4. Pone el modelo en modo .eval() y lo mueve al dispositivo correcto.

    Returns
    -------
    torch.nn.Module
        Modelo listo para inferencia (modo eval).

    Raises
    ------
    FileNotFoundError
        Si el archivo .pth no existe en la ruta configurada.
    RuntimeError
        Si el state_dict no es compatible con la arquitectura instanciada.
    """
    if not FRCNN_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"❌ No se encontró el archivo de pesos Faster R-CNN en:\n"
            f"   {FRCNN_WEIGHTS_PATH}\n"
            f"   Asegúrate de copiar 'faster_rcnn_aquarium.pth' a la carpeta 'models/'."
        )

    print(f"[Faster R-CNN] Construyendo arquitectura ResNet50 FPN V2...")
    model = _build_frcnn_architecture()

    print(f"[Faster R-CNN] Cargando pesos desde: {FRCNN_WEIGHTS_PATH}")
    state_dict = torch.load(str(FRCNN_WEIGHTS_PATH), map_location=DEVICE)

    # Compatibilidad: algunos state_dicts vienen envueltos en una clave 'model'
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()  # Modo evaluación: desactiva Dropout y BatchNorm en modo train

    print(f"[Faster R-CNN] ✅ Modelo cargado correctamente en dispositivo: {DEVICE.upper()}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# CARGA COMBINADA (para inicialización de sesión en Streamlit)
# ═══════════════════════════════════════════════════════════════════════════

def load_all_models() -> Tuple:
    """
    Carga ambos modelos de forma secuencial.
    Usar con @st.cache_resource en la app Streamlit para cachear en sesión.

    Returns
    -------
    Tuple[ultralytics.YOLO, torch.nn.Module]
        (yolo_model, frcnn_model)
    """
    yolo_model = load_yolo_model()
    frcnn_model = load_frcnn_model()
    return yolo_model, frcnn_model


# ── Bloque de prueba directa ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST DE CARGA DE MODELOS")
    print("=" * 60)
    print(f"  Dispositivo detectado: {DEVICE.upper()}")
    print("-" * 60)

    try:
        yolo = load_yolo_model()
        print(f"  YOLO26 cargado:        ✅")
    except Exception as e:
        print(f"  YOLO26 ERROR:          ❌  {e}")

    print("-" * 60)

    try:
        frcnn = load_frcnn_model()
        print(f"  Faster R-CNN cargado:  ✅")
    except Exception as e:
        print(f"  Faster R-CNN ERROR:    ❌  {e}")

    print("=" * 60)
