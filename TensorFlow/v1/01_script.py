"""
Clasificador de huella de IA en diapositivas — TensorFlow
Versión 1

Objetivo de ESTA versión: que entrene de punta a punta y, a propósito,
que sobreajuste. Ver ese sobreajuste es lo que motiva la v2.
Sin augmentation, sin dropout, sin regularización, sin transfer learning.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. Parámetros ---
# DATA_DIR se resuelve a la carpeta data/ en la RAÍZ del repo, sin importar
# desde dónde se ejecute el script (raiz_repo/data). Antes era "../data".
DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123

# --- 2. Carga de datos ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# --- 3. Verificación del orden de clases ---
class_names = train_ds.class_names
print("Orden de clases:", class_names)
# Debe imprimir: ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']

num_classes = len(class_names)

# Prefetch para que el entrenamiento no se quede esperando al disco.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# --- 4. Modelo baseline ---
model = tf.keras.Sequential([
    tf.keras.Input(shape=(*IMG_SIZE, 3)),
    tf.keras.layers.Rescaling(1. / 255),       # normaliza píxeles 0-255 -> 0-1
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

# --- 5. Compilación ---
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- 6. Entrenamiento ---
EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# --- 7. Curvas train vs val (acá se ve el sobreajuste) ---
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 4))
plt.plot(epochs_range, acc, label="train")
plt.plot(epochs_range, val_acc, label="val")
plt.legend()
plt.title("Accuracy: train vs val")
plt.xlabel("época")
plt.ylabel("accuracy")
plt.show()
