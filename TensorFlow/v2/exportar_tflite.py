"""Exporta el modelo entrenado a TFLite para Android — TensorFlow v2.

  python exportar_tflite.py            # float32
  python exportar_tflite.py --quant    # cuantización dinámica (más liviano)

Genera ``modelo_diapositivas.tflite`` (o ``_quant.tflite``).
Entrada: 1x224x224x3 float32, RGB 0-255 (el preprocesamiento va dentro del modelo).
En Android: ``org.tensorflow:tensorflow-lite``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import config
import tensorflow as tf

logger = logging.getLogger("tensorflow.exportar")


def main() -> None:
    config.configurar_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant", action="store_true", help="Cuantización dinámica (int8).")
    args = ap.parse_args()

    if not config.MODELO_PATH.exists():
        sys.exit(f"No existe {config.MODELO_PATH}. Entrena primero: python entrenar.py")

    logger.info("Cargando modelo: %s", config.MODELO_PATH)
    modelo = tf.keras.models.load_model(config.MODELO_PATH)
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    if args.quant:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite = conv.convert()
    nombre = "modelo_diapositivas_quant.tflite" if args.quant else "modelo_diapositivas.tflite"
    out = config.AQUI / nombre
    out.write_bytes(tflite)

    print(f"Guardado {out}  ({len(tflite) / 1024:.0f} KB)")
    print("Android: org.tensorflow:tensorflow-lite . Entrada 1x224x224x3 float32, RGB 0-255.")


if __name__ == "__main__":
    main()
