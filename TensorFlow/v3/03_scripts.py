"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 3  (early stopping / convergencia)

Propósito experimental:
  La v2 quedó SUB-ENTRENADA. Con 10 épocas fijas las curvas de train y val
  todavía venían subiendo, o sea que cortamos el entrenamiento antes de tiempo.
  La v3 NO agrega ninguna técnica nueva: mismo modelo, misma data augmentation
  domain-aware, mismo split (seed 123), mismo IMG_SIZE 180x180 y mismo batch 32
  que la v2. El único cambio conceptual es dejar de cortar arbitrariamente a 10
  épocas y entrenar hasta CONVERGENCIA con un freno automático (early stopping).
  Así el A/B contra la v2 es limpio: si algo cambia en la matriz de confusión, es
  por tiempo de entrenamiento, no por otra cosa.

  La pregunta que la v3 tiene que responder: la confusión residual de la v2 —en
  particular el caso 2->0 (una diapo 'saturada' clasificada como 'sin rastro')—
  ¿era ruido de no-convergencia o algo real del etiquetado? Si al converger
  desaparece, era falta de entrenamiento; si persiste, hay un problema de fondo.

Nuevo respecto a la v2:
  - Early stopping sobre val_loss (patience=8, restore_best_weights=True).
  - Techo de 60 épocas en vez de 10 fijas.
  - Curvas y matriz calculadas sobre las épocas realmente corridas y los mejores
    pesos restaurados.

CONTRASTE DE FRAMEWORKS (clave para el curso):
  En Keras el early stopping es UNA línea de callback que se pasa a model.fit():
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
  En PyTorch (ver PyTorch/v3/03_scripts.py) no existe ese callback: hay que
  implementarlo a mano en el bucle de entrenamiento (guardar deepcopy de los
  mejores pesos, contador de paciencia, break y restaurar al final).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
DATA_DIR = "../../dataset"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123

# NUEVO EN v3: techo alto de épocas. Ya no cortamos a 10; el early stopping decide.
EPOCHS = 60
# NUEVO EN v3: hiperparámetros del early stopping (espejo de PyTorch/v3).
PATIENCE = 8

# --- 2. Carga de datos (split 80/20 IDÉNTICO a la v2, mismo seed) ---
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

# --- Data augmentation domain-aware (IDÉNTICA a la v2) ---
# Se define como bloque de capas. Keras las aplica solo en entrenamiento
# y las apaga solas en validación/inferencia — no hay que hacer nada manual.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),     # foto un poco torcida
    tf.keras.layers.RandomZoom(0.1),          # distancia variable a la diapo
    tf.keras.layers.RandomBrightness(0.2),    # iluminación / reflejo de proyector
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# --- 4. Modelo (mismo baseline de la v2) ---
model = tf.keras.Sequential([
    tf.keras.Input(shape=(*IMG_SIZE, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1. / 255),        # normaliza píxeles 0-255 -> 0-1
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

# --- 6. Entrenamiento con EARLY STOPPING ---
# NUEVO EN v3: el callback reemplaza el corte fijo a 10 épocas. Entrena hasta que
# val_loss deja de mejorar durante PATIENCE épocas y restaura los mejores pesos.
# Esto es TODO el early stopping en Keras; comparar con el bucle manual de PyTorch.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    min_delta=0.0,
    restore_best_weights=True,
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

# --- 7. Curvas train vs val: accuracy Y loss ---
# Usamos las épocas EFECTIVAMENTE corridas (len del history), no range(EPOCHS),
# porque el early stopping pudo cortar antes de 60. Línea vertical en la mejor época.
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epocas_corridas = len(loss)
epochs_range = range(1, epocas_corridas + 1)
# Mejor época = la de menor val_loss (los pesos restaurados por el callback).
mejor_epoca = int(np.argmin(val_loss)) + 1
print(
    f"\nMejor modelo: época {mejor_epoca}/{epocas_corridas}  "
    f"val_loss={val_loss[mejor_epoca - 1]:.4f}  val_acc={val_acc[mejor_epoca - 1]:.4f}  "
    f"(train loss={loss[mejor_epoca - 1]:.4f} acc={acc[mejor_epoca - 1]:.4f})"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, acc, label="train")
ax1.plot(epochs_range, val_acc, label="val")
ax1.axvline(mejor_epoca, ls="--", color="gray", label=f"mejor época ({mejor_epoca})")
ax1.legend()
ax1.set_title("Accuracy: train vs val")
ax1.set_xlabel("época")
ax1.set_ylabel("accuracy")

ax2.plot(epochs_range, loss, label="train")
ax2.plot(epochs_range, val_loss, label="val")
ax2.axvline(mejor_epoca, ls="--", color="gray", label=f"mejor época ({mejor_epoca})")
ax2.legend()
ax2.set_title("Loss: train vs val")
ax2.set_xlabel("época")
ax2.set_ylabel("loss")

plt.tight_layout()
plt.savefig("Figure_1.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 8. Matriz de confusión sobre validación (con los MEJORES pesos restaurados) ---
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(cm)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title("Matriz de confusión (val)")
# Anoto el conteo en cada celda.
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("Figure_2_matriz.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 9. Guardar el modelo entrenado (mismo formato que la v2) ---
model.save("modelotf_v3_convergencia.keras")
