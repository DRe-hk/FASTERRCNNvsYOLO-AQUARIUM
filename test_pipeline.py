"""
test_pipeline.py — Script de prueba del pipeline completo (sin Streamlit).

Verifica que todos los módulos funcionen correctamente antes de lanzar la UI.
Útil para depurar en entornos sin display o antes del primer uso.

Uso:
  python test_pipeline.py --image ruta/imagen.jpg
  python test_pipeline.py --video ruta/video.mp4
  python test_pipeline.py --models-only   (solo prueba la carga de pesos)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEVICE, YOLO_WEIGHTS_PATH, FRCNN_WEIGHTS_PATH


def test_models_load():
    """Test 1: Verifica que ambos modelos cargan correctamente."""
    print("\n" + "═" * 60)
    print("  TEST 1 — Carga de modelos")
    print("═" * 60)
    print(f"  Dispositivo: {DEVICE.upper()}")
    print(f"  YOLO weights: {YOLO_WEIGHTS_PATH} {'✅' if YOLO_WEIGHTS_PATH.exists() else '❌ NO ENCONTRADO'}")
    print(f"  FRCNN weights: {FRCNN_WEIGHTS_PATH} {'✅' if FRCNN_WEIGHTS_PATH.exists() else '❌ NO ENCONTRADO'}")

    from models.model_loader import load_yolo_model, load_frcnn_model

    yolo_model = frcnn_model = None

    try:
        t0 = time.perf_counter()
        yolo_model = load_yolo_model()
        t1 = time.perf_counter()
        print(f"\n  YOLO26 cargado en {(t1-t0)*1000:.0f} ms ✅")
    except Exception as e:
        print(f"\n  YOLO26 ERROR: {e} ❌")

    try:
        t0 = time.perf_counter()
        frcnn_model = load_frcnn_model()
        t1 = time.perf_counter()
        print(f"  Faster R-CNN cargado en {(t1-t0)*1000:.0f} ms ✅")
    except Exception as e:
        print(f"  Faster R-CNN ERROR: {e} ❌")

    return yolo_model, frcnn_model


def test_image(image_path: str, yolo_model, frcnn_model):
    """Test 2: Verifica las funciones de detección en imagen."""
    print("\n" + "═" * 60)
    print("  TEST 2 — Detección en imagen estática")
    print("═" * 60)
    print(f"  Imagen: {image_path}")

    from inference.yolo_image  import detect_image_yolo
    from inference.frcnn_image import detect_image_frcnn

    if yolo_model:
        try:
            result = detect_image_yolo(yolo_model, image_path)
            print(f"\n  YOLO26:")
            print(f"    Tiempo de inferencia : {result.inference_ms:.1f} ms")
            print(f"    Detecciones totales  : {result.total_detections}")
            for d in result.detections:
                print(f"      • {d['class']:12s}  conf={d['confidence']:.2f}  bbox={d['bbox']}")
            # Guardar imagen resultado
            import cv2
            out_path = ROOT / "outputs" / "test_yolo_image.jpg"
            cv2.imwrite(str(out_path), result.annotated_frame_bgr)
            print(f"    Resultado guardado: {out_path} ✅")
        except Exception as e:
            print(f"  YOLO26 ERROR: {e} ❌")

    if frcnn_model:
        try:
            result = detect_image_frcnn(frcnn_model, image_path)
            print(f"\n  Faster R-CNN:")
            print(f"    Tiempo de inferencia : {result.inference_ms:.1f} ms")
            print(f"    Detecciones totales  : {result.total_detections}")
            for d in result.detections:
                print(f"      • {d['class']:12s}  conf={d['confidence']:.2f}  bbox={d['bbox']}")
            import cv2
            out_path = ROOT / "outputs" / "test_frcnn_image.jpg"
            cv2.imwrite(str(out_path), result.annotated_frame_bgr)
            print(f"    Resultado guardado: {out_path} ✅")
        except Exception as e:
            print(f"  Faster R-CNN ERROR: {e} ❌")


def test_video(video_path: str, yolo_model, frcnn_model, max_frames: int = 30):
    """Test 3: Verifica las funciones de detección en video (primeros N frames)."""
    print("\n" + "═" * 60)
    print(f"  TEST 3 — Detección en video (primeros {max_frames} frames)")
    print("═" * 60)
    print(f"  Video: {video_path}")

    from inference.yolo_video  import detect_video_yolo
    from inference.frcnn_video import detect_video_frcnn

    if yolo_model:
        print("\n  YOLO26 video:")
        fps_values = []
        try:
            for i, fr in enumerate(detect_video_yolo(yolo_model, video_path)):
                fps_values.append(fr.fps)
                print(f"    Frame {fr.frame_index:3d} | {fr.inference_ms:.1f} ms | {fr.fps:.1f} FPS | {fr.total_detections} det")
                if i + 1 >= max_frames:
                    break
            avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
            print(f"    → FPS promedio: {avg_fps:.1f} ✅")
        except Exception as e:
            print(f"  YOLO26 ERROR: {e} ❌")

    if frcnn_model:
        print("\n  Faster R-CNN video:")
        fps_values = []
        try:
            for i, fr in enumerate(detect_video_frcnn(frcnn_model, video_path)):
                fps_values.append(fr.fps)
                print(f"    Frame {fr.frame_index:3d} | {fr.inference_ms:.1f} ms | {fr.fps:.1f} FPS | {fr.total_detections} det")
                if i + 1 >= max_frames:
                    break
            avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
            print(f"    → FPS promedio: {avg_fps:.1f} ✅")
        except Exception as e:
            print(f"  Faster R-CNN ERROR: {e} ❌")


def main():
    parser = argparse.ArgumentParser(description="Test del pipeline AquaVision")
    parser.add_argument("--image", type=str, help="Ruta a imagen de prueba")
    parser.add_argument("--video", type=str, help="Ruta a video de prueba")
    parser.add_argument("--models-only", action="store_true", help="Solo prueba la carga de modelos")
    parser.add_argument("--max-frames", type=int, default=30, help="Máx frames a procesar en modo video")
    args = parser.parse_args()

    yolo_model, frcnn_model = test_models_load()

    if not args.models_only:
        if args.image:
            test_image(args.image, yolo_model, frcnn_model)
        if args.video:
            test_video(args.video, yolo_model, frcnn_model, args.max_frames)

    print("\n" + "═" * 60)
    print("  Tests completados.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
