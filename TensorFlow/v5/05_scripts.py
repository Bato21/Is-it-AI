"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 5  (clases DESBALANCEADAS + corrección con pesos de clase)

Propósito experimental:
  La v4 entrenó sobre un dataset balanceado (~100/100/100). La v5 hace UN solo cambio
  conceptual: los datos. Entrena sobre dataset_desbalanceado/ (100/50/25 en las clases
  0/1/2), una distribución más realista de producción (la mayoría de las presentaciones
  son humanas), y analiza el efecto del desbalance y su corrección.

  MISMO modelo que la v4 (MobileNetV3-Small congelada + cabeza nueva), MISMO seed/split
  80-20, MISMO early stopping. Lo único que cambia es el dataset y su corrección.

  En UNA sola corrida se entrenan DOS variantes para el A/B:
    (a) SIN corrección  -> línea base sobre datos desbalanceados.
    (b) CON pesos de clase -> se compensa el desbalance en la pérdida.
  Fórmula estándar de pesos:  w_c = n_total / (n_clases * n_c)   (n_c = casos de la
  clase c en TRAIN). La clase minoritaria recibe más peso.

Nuevo respecto a la v4:
  - Dataset desbalanceado (dataset_desbalanceado/, generado por
    documentacion/crear_subset_desbalanceado.py).
  - Dos variantes en la misma corrida (sin/con pesos) + reporte por clase de cada una.
  - Soporte por clase en validación impreso con la advertencia de leer el recall de la
    clase 2 en NÚMEROS ABSOLUTOS (queda con ~5 casos en val).

CONTRASTE DE FRAMEWORKS (espejo de PyTorch/v5/05_scripts.py):
  En Keras la corrección es un argumento de fit(): class_weight={0:w0,1:w1,2:w2}.
  En PyTorch se elige entre pesar la pérdida (CrossEntropyLoss(weight=...)) o
  re-muestrear (WeightedRandomSampler); no hay un argumento de "fit".
"""

import collections
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
# v5: dataset DESBALANCEADO (no dataset/). Se genera aparte y no toca el original.
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset_desbalanceado")
if not Path(DATA_DIR).is_dir():
    raise SystemExit(
        f"No existe la carpeta de datos: {DATA_DIR}\n"
        "Generá el subset desbalanceado primero:\n"
        "    python documentacion/crear_subset_desbalanceado.py"
    )
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 60
PATIENCE = 8

# --- 2. Carga de datos (split 80/20, mismo seed que v4) ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Orden de clases:", class_names)
num_classes = len(class_names)


def contar_labels(ds) -> collections.Counter:
    c = collections.Counter()
    for _, y in ds.unbatch():
        c[int(y)] += 1
    return c


# Soporte por clase (train y val) ANTES de prefetch.
train_counts = contar_labels(train_ds)
val_counts = contar_labels(val_ds)

print("\nSoporte por clase:")
print(f"{'clase':<16}{'train':>7}{'val':>6}")
for i, nombre in enumerate(class_names):
    print(f"{nombre:<16}{train_counts[i]:>7}{val_counts[i]:>6}")
print(
    "\nAVISO: con 80/20 sobre ~175 imágenes, la clase 2 queda con pocos casos en val\n"
    f"       ({val_counts[2]} imágenes). Leé su recall en NÚMEROS ABSOLUTOS, no solo en %."
)

# --- Pesos de clase:  w_c = n_total / (n_clases * n_c)  ---
n_total = sum(train_counts.values())
pesos = {i: n_total / (num_classes * train_counts[i]) for i in range(num_classes)}
print("\nPesos de clase (para la variante con corrección):")
for i, nombre in enumerate(class_names):
    print(f"  {nombre:<16} w={pesos[i]:.3f}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# --- 3. Reporte por clase (calculado A MANO desde la matriz de confusión) ---
# Duplicado a propósito en cada lado (TF y PT) para no acoplar entornos; misma
# función chica aparece en reporte_metricas.py.  cm[i, j] = real i, predicho j.
def reporte_por_clase(cm: np.ndarray, nombres: list[str]) -> None:
    total = cm.sum()
    print(f"{'clase':<16}{'precision':>10}{'recall':>9}{'f1':>7}{'soporte':>9}")
    print("-" * 51)
    precs, recs, f1s = [], [], []
    for i in range(len(nombres)):
        soporte = cm[i, :].sum()             # casos reales de la clase i
        predichos = cm[:, i].sum()           # veces que se predijo la clase i
        recall = cm[i, i] / soporte if soporte else 0.0
        precision = cm[i, i] / predichos if predichos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
        print(f"{nombres[i]:<16}{precision:>10.3f}{recall:>9.3f}{f1:>7.3f}{int(soporte):>9}")
    print("-" * 51)
    print(f"{'macro avg':<16}{np.mean(precs):>10.3f}{np.mean(recs):>9.3f}{np.mean(f1s):>7.3f}{int(total):>9}")
    acc = np.trace(cm) / total if total else 0.0
    print(f"accuracy: {acc:.3f}  ({int(np.trace(cm))}/{int(total)})")


# --- 4. Modelo: MobileNetV3-Small congelada + cabeza (idéntico a la v4) ---
def construir_modelo() -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False  # feature extraction; BN en modo inferencia
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def matriz_confusion(model) -> np.ndarray:
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    return tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()


def entrenar_variante(usar_pesos: bool):
    """Entrena una variante (sin/con pesos) y devuelve (history, cm)."""
    model = construir_modelo()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, min_delta=0.0, restore_best_weights=True,
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early],
        class_weight=(pesos if usar_pesos else None),
        verbose=2,
    )
    return history, matriz_confusion(model)


# --- 5. Correr las DOS variantes en la misma corrida ---
variantes = [("sin_pesos", False), ("con_pesos", True)]
resultados = {}
for etiqueta, usar in variantes:
    titulo = "CON pesos de clase" if usar else "SIN corrección (línea base)"
    print("\n" + "=" * 70)
    print(f"VARIANTE: {titulo}")
    print("=" * 70)
    history, cm = entrenar_variante(usar)
    mejor = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nMejor época {mejor}/{len(history.history['val_loss'])}  "
          f"val_loss={min(history.history['val_loss']):.4f}")
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    print("\nReporte por clase:")
    reporte_por_clase(cm, class_names)
    resultados[etiqueta] = {"history": history, "cm": cm}

# --- 6. Figuras: matrices lado a lado + curvas de las dos variantes ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (etiqueta, _) in zip(axes, variantes):
    cm = resultados[etiqueta]["cm"]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {etiqueta}")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_matrices_v5.png", dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for fila, (etiqueta, _) in enumerate(variantes):
    h = resultados[etiqueta]["history"].history
    rng = range(1, len(h["loss"]) + 1)
    axes[fila, 0].plot(rng, h["accuracy"], label="train")
    axes[fila, 0].plot(rng, h["val_accuracy"], label="val")
    axes[fila, 0].set_title(f"Accuracy — {etiqueta}"); axes[fila, 0].legend()
    axes[fila, 0].set_xlabel("época"); axes[fila, 0].set_ylabel("accuracy")
    axes[fila, 1].plot(rng, h["loss"], label="train")
    axes[fila, 1].plot(rng, h["val_loss"], label="val")
    axes[fila, 1].set_title(f"Loss — {etiqueta}"); axes[fila, 1].legend()
    axes[fila, 1].set_xlabel("época"); axes[fila, 1].set_ylabel("loss")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_curvas_v5.png", dpi=150, bbox_inches="tight")
print("\nGuardado Figure_matrices_v5.png y Figure_curvas_v5.png")
