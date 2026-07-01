"""Entrenamiento del clasificador de huella de IA — TensorFlow v2.

Transfer learning con MobileNetV3Large en dos fases (cabeza congelada →
fine-tuning), respondiendo al sobreajuste de la v1 (CNN desde cero):
  - Transfer learning (MobileNetV3Large preentrenada en ImageNet)
  - Data augmentation como capas del modelo
  - Dropout en la cabeza
  - Dos fases: base congelada → fine-tuning de las últimas capas

Clases (orden fijo):
  0_sin_ia      — sin huella de IA visible
  1_rastro_ia   — IA con clara intervención humana (rastro)
  2_saturada_ia — predominantemente generada por IA

Genera, en esta carpeta:
  modelo_diapositivas.keras  — modelo entrenado (lo usa consumidor.py)
  training_curves.png · confusion_matrix.png · roc_curves.png

Configuración y arquitectura: config.py · modelo.py · datos.py.
Métricas y figuras: metricas.py.
"""

from __future__ import annotations

import logging

import numpy as np

import config
import datos
import metricas
import tensorflow as tf
from modelo import construir_modelo

logger = logging.getLogger("tensorflow.entrenar")


def _callbacks_fase(con_reduce_lr: bool) -> list[tf.keras.callbacks.Callback]:
    """EarlyStopping (+ ReduceLROnPlateau en la fase 1) sobre val_accuracy."""
    cbs: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        )
    ]
    if con_reduce_lr:
        cbs.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
            )
        )
    return cbs


def entrenar(
    modelo: tf.keras.Model,
    base: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
) -> metricas.Historial:
    """Entrena en dos fases y devuelve el historial en formato común."""
    logger.info("FASE 1 — Cabeza (%d épocas), base congelada", config.EPOCHS_FASE1)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(config.LR_FASE1),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    h1 = modelo.fit(
        train_ds,
        epochs=config.EPOCHS_FASE1,
        validation_data=val_ds,
        callbacks=_callbacks_fase(con_reduce_lr=True),
    ).history

    logger.info("FASE 2 — Fine-tuning (desde índice %d)", config.FINE_TUNE_AT)
    base.trainable = True
    for capa in base.layers[: config.FINE_TUNE_AT]:
        capa.trainable = False
    logger.info(
        "Capas descongeladas: %d  |  LR %g",
        sum(c.trainable for c in base.layers),
        config.LR_FASE2,
    )
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(config.LR_FASE2),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    h2 = modelo.fit(
        train_ds,
        epochs=config.EPOCHS_FASE2,
        validation_data=val_ds,
        callbacks=_callbacks_fase(con_reduce_lr=False),
    ).history

    return metricas.Historial(
        acc=h1["accuracy"] + h2["accuracy"],
        val_acc=h1["val_accuracy"] + h2["val_accuracy"],
        loss=h1["loss"] + h2["loss"],
        val_loss=h1["val_loss"] + h2["val_loss"],
        corte=len(h1["accuracy"]),
    )


def predecir_test(
    modelo: tf.keras.Model, test_ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (y_true, y_pred, y_prob) sobre el conjunto de test."""
    y_true_lotes, y_prob_lotes = [], []
    for imgs, labels in test_ds:
        probs = modelo(imgs, training=False).numpy()
        y_true_lotes.append(np.argmax(labels.numpy(), axis=1))
        y_prob_lotes.append(probs)
    y_true = np.concatenate(y_true_lotes)
    y_prob = np.concatenate(y_prob_lotes)
    return y_true, np.argmax(y_prob, axis=1), y_prob


def main() -> None:
    config.configurar_logging()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    train_ds, val_ds, test_ds = datos.cargar_dataset()

    logger.info("Construyendo modelo...")
    modelo, base = construir_modelo()
    logger.info("Parámetros totales: %d", modelo.count_params())

    historial = entrenar(modelo, base, train_ds, val_ds)

    logger.info("Evaluando en test...")
    y_true, y_pred, y_prob = predecir_test(modelo, test_ds)
    cm = metricas.matriz_confusion(y_true, y_pred, config.N_CLASES)
    met = metricas.metricas_por_clase(cm)
    acc = metricas.accuracy(cm)

    # Salida orientada al usuario => print (no logging).
    metricas.imprimir_tabla(met, config.CLASES, acc)

    logger.info("Generando visualizaciones...")
    metricas.plot_curvas(
        historial, config.TITULO_CURVAS, config.ETIQUETA_LOSS, str(config.CURVAS_PATH)
    )
    metricas.plot_matriz_confusion(
        cm, met, acc, config.CLASES, config.TITULO_MATRIZ, str(config.MATRIZ_PATH)
    )
    metricas.plot_roc(
        y_true, y_prob, config.CLASES, config.COLORES, config.TITULO_ROC, str(config.ROC_PATH)
    )

    modelo.save(config.MODELO_PATH)
    logger.info("Modelo guardado: %s", config.MODELO_PATH)
    print(f"\nListo. Modelo en {config.MODELO_PATH}")


if __name__ == "__main__":
    main()
