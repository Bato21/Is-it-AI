"""Consumidor del modelo de diapositivas — TensorFlow v2.

Carga ``modelo_diapositivas.keras`` (generado por entrenar.py) y predice la
clase de una o varias imágenes. Independiente del entrenamiento: solo necesita
el archivo ``.keras``.

  python consumidor.py imagen.png
  python consumidor.py carpeta_con_imagenes/
  python consumidor.py img1.png img2.png ...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

import config
import datos
import tensorflow as tf

logger = logging.getLogger("tensorflow.consumidor")


def imprimir_prediccion(ruta: Path, probs: np.ndarray) -> None:
    """Imprime la predicción de una imagen (salida orientada al usuario)."""
    idx = int(np.argmax(probs))
    barra = "█" * int(probs[idx] * 20)
    print(f"\n  {ruta.name}")
    print(f"  -> {config.CLASES[idx]}  ({probs[idx]:.1%})  {barra}")
    for i, clase in enumerate(config.CLASES):
        print(f"     {clase:<14} {probs[i]:.1%}")


def expandir(args: list[str]) -> list[Path]:
    """Expande archivos y carpetas a una lista ordenada de imágenes válidas."""
    rutas: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            rutas += [f for f in sorted(p.iterdir()) if f.suffix.lower() in config.EXTS]
        elif p.suffix.lower() in config.EXTS:
            rutas.append(p)
    return rutas


def main() -> None:
    config.configurar_logging()
    if len(sys.argv) < 2:
        sys.exit("Uso: python consumidor.py <imagen|carpeta> [...]")
    if not config.MODELO_PATH.exists():
        sys.exit(f"No existe {config.MODELO_PATH}. Entrena primero: python entrenar.py")

    rutas = expandir(sys.argv[1:])
    if not rutas:
        sys.exit("No se encontraron imágenes válidas.")

    logger.info("Cargando modelo: %s", config.MODELO_PATH)
    modelo = tf.keras.models.load_model(config.MODELO_PATH)
    for ruta in rutas:
        probs = modelo.predict(datos.cargar_imagen(ruta), verbose=0)[0]
        imprimir_prediccion(ruta, probs)


if __name__ == "__main__":
    main()
