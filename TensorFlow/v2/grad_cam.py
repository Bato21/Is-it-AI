"""Grad-CAM para el modelo de diapositivas — TensorFlow v2.

Muestra DÓNDE mira la red dentro de la imagen al decidir la clase. Es la misma
idea del ejemplo de clase (Grad-CAM con Xception), adaptada a nuestro modelo con
MobileNetV3Large como base.

  python grad_cam.py una_diapositiva.png
  -> genera grad_cam_output.png (mapa de calor superpuesto)

Útil para explicar el modelo: ¿se fija en el layout/plantilla, en el texto, en
las imágenes generadas? Eso respalda el criterio de las 3 clases.
"""

from __future__ import annotations

import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import tensorflow as tf
from modelo import encontrar_base

logger = logging.getLogger("tensorflow.grad_cam")


def construir_modelo_inferencia(modelo: tf.keras.Model, base: tf.keras.Model) -> tf.keras.Model:
    """Modelo plano base→cabeza que expone el mapa de características y la salida.

    No se parte de ``modelo.inputs`` porque la capa 'augmentation' es un
    sub-modelo con su propio Input y desconecta el grafo ("Graph disconnected").
    En inferencia el augmentation es identidad, así que reusamos los tensores ya
    construidos del base (``base.input`` / ``base.output``) y reaplicamos la
    cabeza (GAP → Dense → Dropout → Softmax). Devuelve ``[mapa, predicciones]``.
    """
    idx = modelo.layers.index(base)
    cabeza = modelo.layers[idx + 1 :]
    feat = base.output  # mapa de características (None, 7, 7, 960)
    x = feat
    for capa in cabeza:
        x = capa(x)
    return tf.keras.Model(base.input, [feat, x])


def heatmap_gradcam(
    img_array: np.ndarray,
    grad_model: tf.keras.Model,
    pred_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """Calcula el mapa de calor Grad-CAM para una imagen (1, H, W, 3) en 0-255."""
    with tf.GradientTape() as tape:
        feat, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        canal = preds[:, pred_index]
    grads = tape.gradient(canal, feat)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    feat = feat[0]
    heat = tf.squeeze(feat @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + tf.keras.backend.epsilon())
    return heat.numpy(), int(pred_index)


def guardar(img_path: str, heat: np.ndarray, pred: int, alpha: float = 0.4) -> None:
    """Superpone el mapa de calor sobre la imagen original y lo guarda."""
    img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(img_path))
    heat = np.uint8(255 * heat)
    jet = plt.colormaps["jet"](np.arange(256))[:, :3]
    jet_heat = jet[heat]
    jet_heat = tf.keras.utils.array_to_img(jet_heat).resize((img.shape[1], img.shape[0]))
    jet_heat = tf.keras.utils.img_to_array(jet_heat)
    superp = tf.keras.utils.array_to_img(jet_heat * alpha + img)

    plt.imshow(superp)
    plt.axis("off")
    plt.title(f"Grad-CAM — predicción: {config.CLASES[pred]}")
    plt.savefig(config.GRADCAM_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    config.configurar_logging()
    if len(sys.argv) < 2:
        sys.exit("Uso: python grad_cam.py <imagen.png>")
    if not config.MODELO_PATH.exists():
        sys.exit(f"No existe {config.MODELO_PATH}. Entrena primero: python entrenar.py")
    img_path = sys.argv[1]

    logger.info("Cargando modelo: %s", config.MODELO_PATH)
    modelo = tf.keras.models.load_model(config.MODELO_PATH)
    base = encontrar_base(modelo)
    grad_model = construir_modelo_inferencia(modelo, base)

    img = tf.keras.utils.load_img(img_path, target_size=config.IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), 0)  # 0-255
    heat, pred = heatmap_gradcam(arr, grad_model)
    guardar(img_path, heat, pred)
    print(f"Grad-CAM guardado en '{config.GRADCAM_PATH}'  (predicción: {config.CLASES[pred]})")


if __name__ == "__main__":
    main()
