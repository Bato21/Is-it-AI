"""
Exporta el modelo entrenado a TFLite para Android — TensorFlow v2

  python exportar_tflite.py            # float32
  python exportar_tflite.py --quant    # cuantización dinámica (más liviano)

Genera modelo_diapositivas.tflite (o _quant.tflite).
Entrada: 1x224x224x3 float32, RGB 0-255 (el preprocesamiento va dentro del modelo).
En Android: org.tensorflow:tensorflow-lite.
"""
import argparse
import sys
from pathlib import Path

import tensorflow as tf

AQUI = Path(__file__).resolve().parent
MODELO = AQUI / "modelo_diapositivas.keras"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", action="store_true", help="Cuantización dinámica (int8).")
    args = ap.parse_args()

    if not MODELO.exists():
        sys.exit(f"No existe {MODELO}. Entrena primero: python entrenar.py")

    modelo = tf.keras.models.load_model(MODELO)
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    if args.quant:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite = conv.convert()
    nombre = "modelo_diapositivas_quant.tflite" if args.quant else "modelo_diapositivas.tflite"
    out = AQUI / nombre
    out.write_bytes(tflite)
    print(f"Guardado {out}  ({len(tflite)/1024:.0f} KB)")
    print("Android: org.tensorflow:tensorflow-lite . Entrada 1x224x224x3 float32, RGB 0-255.")


if __name__ == "__main__":
    main()
