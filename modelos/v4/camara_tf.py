"""
Cámara en vivo — TensorFlow v4 (MobileNetV3-Small).

Apunta la webcam a una diapositiva (en pantalla o impresa) y clasifica en vivo su
huella de IA (clases 0/1/2) mostrando el vector completo de probabilidades.

Valor narrativo: este script es también el PRIMER banco de pruebas del *domain gap*.
Entrenamos con renders limpios, pero acá la cámara ve una diapo con moiré de pantalla,
reflejos y perspectiva. Las capturas que guardes con 's' (en modelos/v4/capturas/)
alimentan el análisis de métricas (analisis_metricas.md).

Controles:
  espacio -> congela el frame y predice sobre el congelado (soltás para seguir)
  s       -> guarda el frame anotado + la predicción en modelos/v4/capturas/
  q       -> salir

CONTRASTE DE FRAMEWORKS (espejo de camara_pt.py):
  TF: frame RGB -> float32 0-255 SIN dividir por 255 y SIN normalizar -> expand_dims
      -> model(batch, training=False). La normalización ImageNet vive DENTRO del .keras.
  PT: frame RGB -> PIL -> el MISMO transform de eval del checkpoint (Resize+ToTensor+
      Normalize con mean/std leídos del .pt). La normalización vive FUERA del modelo.
"""
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

import cv2
import numpy as np
import tensorflow as tf

# --- CONFIG ---
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "TensorFlow" / "v4" / "modelotf_v4_mobilenetv3.keras"
CAPTURAS = Path(__file__).resolve().parent / "capturas"
IMG_SIZE = (224, 224)                          # MISMO tamaño que en entrenamiento (TF v4)
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
PREDICE_CADA = 5                               # predecir cada N frames para mantener fluidez

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Color por clase (BGR): 0 verde (limpio), 1 ámbar, 2 rojo (saturada).
COLORES = [(0, 180, 0), (0, 180, 255), (0, 0, 220)]


# --- Cargar modelo ---
if not MODEL_PATH.exists():
    raise SystemExit(
        f"No encontré el modelo en '{MODEL_PATH}'.\n"
        "Entrená primero TF v4:  python TensorFlow/v4/04_scripts.py"
    )
model = tf.keras.models.load_model(MODEL_PATH)


def predecir(frame_bgr) -> np.ndarray:
    """Frame BGR de OpenCV -> vector de 3 probabilidades."""
    # GOTCHA CLAVE: OpenCV entrega BGR; el modelo se entrenó con RGB. Sin este
    # cvtColor las predicciones se degradan en silencio (canales R y B cruzados).
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Aplastar el frame 16:9 a cuadrado 224x224 es CONSISTENTE con el entrenamiento
    # (image_dataset_from_directory hace exactamente lo mismo con las diapos); no es
    # un bug, es paridad de pipeline.
    img = cv2.resize(rgb, IMG_SIZE)
    # 0-255 crudo: NO /255 y NO normalizar. La normalización ImageNet vive en el .keras.
    batch = np.expand_dims(img.astype("float32"), axis=0)
    return model(batch, training=False).numpy()[0]


def dibujar_overlay(vis, probs, congelado: bool) -> None:
    """Dibuja clase ganadora, barras por clase y vector completo sobre el frame."""
    h = vis.shape[0]
    idx = int(np.argmax(probs))
    cv2.putText(vis, f"{CLASS_NAMES[idx]}  {probs[idx] * 100:4.1f}%",
                (12, 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    x0, y0, bw, bh, gap = 12, 46, 200, 18, 8
    for i, p in enumerate(probs):
        y = y0 + i * (bh + gap)
        cv2.rectangle(vis, (x0, y), (x0 + bw, y + bh), (60, 60, 60), 1)          # marco
        cv2.rectangle(vis, (x0, y), (x0 + int(bw * p), y + bh), COLORES[i], -1)  # relleno
        cv2.putText(vis, f"{CLASS_NAMES[i][2:]:11s} {p * 100:4.1f}%",
                    (x0 + bw + 10, y + bh - 3), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(vis, f"[{probs[0]:.2f} / {probs[1]:.2f} / {probs[2]:.2f}]",
                (12, y0 + 3 * (bh + gap) + 12), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if congelado:
        cv2.putText(vis, "CONGELADO (espacio = seguir)", (12, h - 40),
                    FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "espacio=congelar   s=guardar   q=salir", (12, h - 15),
                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def dibujar_guia(vis) -> None:
    """Rectángulo guía 16:9 centrado para encuadrar la diapo (solo guía visual)."""
    # Mejora futura: recortar ESTE ROI antes de predecir para aislar la diapo del
    # fondo y reducir el domain gap. Por ahora es solo ayuda de encuadre.
    h, w = vis.shape[:2]
    gw = int(w * 0.7)
    gh = int(gw * 9 / 16)
    if gh > h * 0.9:
        gh = int(h * 0.7)
        gw = int(gh * 16 / 9)
    cx, cy = w // 2, h // 2
    p1 = (cx - gw // 2, cy - gh // 2)
    p2 = (cx + gw // 2, cy + gh // 2)
    cv2.rectangle(vis, p1, p2, (255, 255, 0), 1)
    cv2.putText(vis, "encuadra la diapo (16:9)", (p1[0], p1[1] - 8),
                FONT, 0.5, (255, 255, 0), 1, cv2.LINE_AA)


def abrir_camara() -> cv2.VideoCapture:
    """VideoCapture(0); si falla, prueba el índice 1; si nada, mensaje claro y sale."""
    for idx in (0, 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Cámara abierta en el índice {idx}. Teclas: espacio / s / q.")
            return cap
        cap.release()
    raise SystemExit(
        "No se pudo abrir la cámara (probé índices 0 y 1).\n"
        "Revisá que haya una webcam conectada y permisos de cámara."
    )


def guardar(vis, probs) -> None:
    """Guarda el frame anotado + una línea con la predicción en modelos/v4/capturas/."""
    CAPTURAS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = CAPTURAS / f"captura_tf_{ts}.png"
    cv2.imwrite(str(ruta), vis)
    idx = int(np.argmax(probs))
    linea = (f"{ruta.name}\t{CLASS_NAMES[idx]}\t"
             f"[{probs[0]:.3f} / {probs[1]:.3f} / {probs[2]:.3f}]")
    with open(CAPTURAS / "capturas_tf.txt", "a", encoding="utf-8") as f:
        f.write(linea + "\n")
    print("Guardado:", linea)


def main() -> None:
    cap = abrir_camara()
    congelado = None          # frame BGR anotado y congelado, o None si está en vivo
    probs = None
    frame_idx = 0
    try:
        while True:
            if congelado is None:
                ok, frame = cap.read()
                if not ok:
                    print("No llegan frames de la cámara; salgo.")
                    break
                frame_idx += 1
                if probs is None or frame_idx % PREDICE_CADA == 0:
                    probs = predecir(frame)
                vis = frame.copy()
                dibujar_overlay(vis, probs, congelado=False)
                dibujar_guia(vis)
            else:
                vis = congelado

            cv2.imshow("Is-it-AI - camara TF v4", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                if congelado is None:
                    # Congelar: predecir sobre el frame actual y fijar el anotado.
                    probs = predecir(frame)
                    snap = frame.copy()
                    dibujar_overlay(snap, probs, congelado=True)
                    dibujar_guia(snap)
                    congelado = snap
                else:
                    congelado = None  # descongelar y volver al vivo
            elif k == ord("s"):
                guardar(vis, probs)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
