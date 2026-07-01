"""Carga y validación del dataset para TensorFlow v2.

Separa la lógica de datos de la de entrenamiento e incorpora chequeos de casos
borde con mensajes claros: carpeta inexistente, dataset vacío, alguna clase sin
imágenes y desbalance fuerte entre clases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config
import tensorflow as tf

logger = logging.getLogger(__name__)

# Si la clase más poblada supera a la menos poblada por este factor, avisamos.
FACTOR_DESBALANCE = 10


def _contar_por_clase() -> dict[str, int]:
    """Cuenta imágenes válidas por clase y valida la estructura de carpetas."""
    raiz = config.DATASET_DIR
    if not raiz.exists():
        raise SystemExit(
            f"No existe la carpeta de dataset: {raiz}\n"
            f"Crea las subcarpetas {config.CLASES} con imágenes, o genera datos "
            f"de prueba: python herramientas/crear_datos_prueba.py"
        )

    conteo: dict[str, int] = {}
    vacias: list[str] = []
    for clase in config.CLASES:
        carpeta = raiz / clase
        if not carpeta.exists():
            raise SystemExit(
                f"Falta la carpeta de la clase '{clase}' en {raiz}. "
                f"Se esperan exactamente estas clases: {config.CLASES}"
            )
        n = sum(1 for f in carpeta.glob("*") if f.suffix.lower() in config.EXTS)
        conteo[clase] = n
        if n == 0:
            vacias.append(clase)

    if vacias:
        raise SystemExit(
            f"Estas clases no tienen imágenes: {vacias}. "
            f"Clasifica imágenes en dataset/<clase>/ antes de entrenar."
        )
    return conteo


def _avisar_desbalance(conteo: dict[str, int]) -> None:
    """Registra una advertencia si el dataset está muy desbalanceado."""
    menor, mayor = min(conteo.values()), max(conteo.values())
    if menor and mayor >= menor * FACTOR_DESBALANCE:
        logger.warning(
            "Dataset desbalanceado (de %d a %d por clase): %s. "
            "Las métricas por clase pueden ser engañosas.",
            menor,
            mayor,
            conteo,
        )


def cargar_dataset() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Carga el dataset y lo parte 70% train · 15% val · 15% test.

    El orden de clases queda fijo en ``config.CLASES`` (no alfabético sorpresa)
    y todo se siembra con ``config.SEED`` para reproducibilidad.
    """
    conteo = _contar_por_clase()
    total = sum(conteo.values())
    logger.info("Total imágenes: %d  |  por clase: %s", total, conteo)
    _avisar_desbalance(conteo)

    kwargs = dict(
        directory=str(config.DATASET_DIR),
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_names=config.CLASES,  # fija el orden 0,1,2
        label_mode="categorical",  # one-hot => necesario para ROC y CE
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=config.VALIDATION_SPLIT, subset="training", **kwargs
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=config.VALIDATION_SPLIT, subset="validation", **kwargs
    )

    # Partir val_test en 50/50 a NIVEL DE EJEMPLO (no de batch). Con datasets
    # pequeños puede haber un solo batch; un skip() por batch dejaría test vacío.
    val_test_unb = val_test_ds.unbatch()
    n_vt = sum(int(tf.shape(y)[0]) for _, y in val_test_ds)
    n_val = max(1, min(n_vt - 1, n_vt // 2))  # garantiza >=1 ejemplo en test
    val_ds = val_test_unb.take(n_val).batch(config.BATCH_SIZE)
    test_ds = val_test_unb.skip(n_val).batch(config.BATCH_SIZE)
    logger.info("Particiones val/test (ejemplos): val=%d  test=%d", n_val, n_vt - n_val)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=config.SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds


def cargar_imagen(ruta: Path) -> tf.Tensor:
    """Carga una imagen como tensor (1, H, W, 3) en 0-255 (preproc va en el modelo)."""
    img = tf.keras.utils.load_img(ruta, target_size=config.IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return tf.expand_dims(arr, 0)
