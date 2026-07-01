"""Definición del modelo TensorFlow v2 (única fuente de verdad).

Antes la arquitectura estaba repetida en ``entrenar.py``, ``consumidor.py`` y
``exportar_tflite.py``. Vive aquí una sola vez; los demás scripts la importan.
"""

from __future__ import annotations

import config
import tensorflow as tf


def construir_augmentation() -> tf.keras.Sequential:
    """Capas de data augmentation (activas solo en entrenamiento)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def construir_modelo() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Construye el modelo completo y devuelve ``(modelo, base)``.

    Topología: augmentation → MobileNetV3Large (congelado) → GAP → Dense →
    Dropout → Softmax.

    ``include_preprocessing=True`` hace que el modelo espere imágenes en 0-255
    (el escalado a [-1, 1] ocurre dentro de la red), lo que simplifica el
    despliegue móvil. Se devuelve también ``base`` para que ``entrenar.py``
    pueda descongelarlo en la fase 2.
    """
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=(*config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )
    base.trainable = False  # Fase 1: base congelada

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = construir_augmentation()(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT)(x)
    outputs = tf.keras.layers.Dense(config.N_CLASES, activation="softmax")(x)

    modelo = tf.keras.Model(inputs, outputs, name="mobilenetv3_diapositivas")
    return modelo, base


def encontrar_base(modelo: tf.keras.Model) -> tf.keras.Model:
    """Devuelve el sub-modelo MobileNetV3Large dentro de un modelo cargado.

    Lo usa ``grad_cam.py`` para reconstruir el modelo de inferencia a partir de
    un ``.keras`` ya entrenado.
    """
    for capa in modelo.layers:
        if isinstance(capa, tf.keras.Model):
            return capa
    raise RuntimeError("No se encontró el sub-modelo base en el modelo cargado.")
