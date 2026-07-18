"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 4  (transfer learning: MobileNetV3-Small, feature extraction)

Propósito experimental:
  Hasta la v3 el modelo era una CNN mínima entrenada desde cero (~3.79M parámetros,
  todos aprendidos con nuestras 300 imágenes). La v4 hace el cambio conceptual que
  el roadmap del proyecto tenía reservado: reemplazar esa CNN por MobileNetV3-Small
  preentrenada en ImageNet, congelar el backbone y entrenar SOLO una cabeza nueva.
  Esto es "feature extraction": aprovechamos filtros visuales ya aprendidos sobre
  millones de imágenes y solo ajustamos el clasificador final a nuestras 3 clases.

  El early stopping de la v3 se mantiene: ya es parte de la receta base, no un
  cambio de esta versión.

Nuevo respecto a la v3:
  - Backbone MobileNetV3-Small preentrenado (ImageNet), congelado.
  - IMG_SIZE 224x224 (requisito de entrada de MobileNetV3).
  - Preprocesamiento ImageNet integrado DENTRO del modelo (ver bloque 4).
  - Cabeza nueva: GlobalAveragePooling + Dropout(0.2) + Dense(3).

Nota: la primera corrida DESCARGA los pesos (~10 MB) a ~/.keras; requiere internet
una única vez. Las corridas siguientes usan la caché local.

CONTRASTE DE FRAMEWORKS (espejo de PyTorch/v4/04_scripts.py):
  - NORMALIZACIÓN: en Keras la normalización ImageNet vive DENTRO del modelo. Con
    include_preprocessing=True (default) MobileNetV3Small trae su propia capa de
    preprocesamiento, así que le pasamos imágenes 0-255 crudas y la normalización
    viaja con el .keras — la inferencia la aplica sola. En PyTorch la normalización
    vive en el TRANSFORM (fuera del modelo) y la inferencia tiene que reproducirla
    a mano con transforms.Normalize(mean, std), o se des-sincroniza.
  - BATCHNORM: en Keras trainable=False pone AUTOMÁTICAMENTE las capas BatchNorm en
    modo inferencia (usan sus estadísticas de ImageNet congeladas). En PyTorch esto
    NO pasa: requires_grad=False congela los pesos pero BatchNorm sigue actualizando
    running_mean/running_var en modo train(); hay que forzar el backbone a eval() a
    mano. Acá lo reforzamos igual pasando training=False al llamar al backbone.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
# T0: ruta canónica ÚNICA para TF y PyTorch -> <repo>/dataset (absoluta vía __file__,
# robusta al directorio desde el que se corra). Antes era el relativo "../../dataset".
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset")
if not Path(DATA_DIR).is_dir():
    raise SystemExit(
        f"No existe la carpeta de datos: {DATA_DIR}\n"
        "Colocá el dataset ahí (subcarpetas 0_sin_ia/ 1_rastro_ia/ 2_saturada_ia/), o\n"
        "generá datos sintéticos para probar el flujo de punta a punta:\n"
        "    python documentacion/crear_datos_prueba.py --por-clase 30"
    )
# CAMBIO: 224x224 en vez de 180x180. MobileNetV3 fue entrenada con este tamaño;
# usar otro degrada las features preentrenadas. Es un requisito de la arquitectura,
# no una elección libre como en las v1-v3.
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# Early stopping (idéntico a la v3).
EPOCHS = 60  # techo alto; el early stopping decide (heredado de la v3).
PATIENCE = 8

# --- 2. Carga de datos (split 80/20 IDÉNTICO a v2/v3, mismo seed) ---
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

# --- Data augmentation domain-aware (IDÉNTICA a v2/v3) ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),     # foto un poco torcida
    tf.keras.layers.RandomZoom(0.1),          # distancia variable a la diapo
    tf.keras.layers.RandomBrightness(0.2),    # iluminación / reflejo de proyector
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# --- 4. Modelo: MobileNetV3-Small preentrenado, backbone congelado ---
# weights="imagenet" descarga los pesos recomendados la 1a vez.
# include_top=False descarta el clasificador de 1000 clases de ImageNet.
# include_preprocessing=True (default) trae la normalización ImageNet DENTRO del
# modelo, así que el backbone espera píxeles 0-255 crudos (no hay que normalizar
# a mano como en PyTorch). NO agregamos capa Rescaling: la haría dos veces.
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
# Congelar el backbone (extractor de features): con 300 imágenes, re-entrenar el
# backbone entero sería sobreajuste garantizado. Al poner trainable=False, Keras
# también pone las BatchNorm en modo inferencia automáticamente.
base_model.trainable = False

# Cabeza nueva acorde a nuestras 3 clases. MobileNetV3 original ya usa Dropout(0.2)
# en su cabeza; lo replicamos y no amontonamos regularización extra.
inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)  # training=False => BatchNorm en modo inferencia
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Conteo de parámetros: contrastar TOTALES vs ENTRENABLES es parte del mensaje de la
# presentación. En v1-v3 los 3.79M eran TODOS entrenables; acá casi todo está
# congelado y solo entrenamos la cabeza.
total_params = model.count_params()
train_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
print(f"Parámetros totales:     {total_params:,}")
print(f"Parámetros entrenables: {train_params:,}  (solo la cabeza)")

# --- 5. Compilación ---
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- 6. Entrenamiento con EARLY STOPPING (heredado de la v3) ---
# Mismo callback que la v3: entrena hasta que val_loss deja de mejorar durante
# PATIENCE épocas y restaura los mejores pesos. En PyTorch esto se implementa a
# mano en el bucle; acá es una línea.
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
    verbose=2,  # una línea por época: log legible para Resultado_4.txt
)

# --- 7. Curvas train vs val: accuracy Y loss (épocas realmente corridas) ---
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

# --- 8. Matriz de confusión sobre validación (con los mejores pesos restaurados) ---
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

# --- 9. Guardar el modelo entrenado ---
# El .keras incluye la normalización ImageNet dentro del modelo (bloque 4), así que
# la inferencia NO tiene que reproducirla a mano — contraste con el checkpoint de
# PyTorch, que guarda normalize_mean/std aparte para replicarlos en la inferencia.
model.save("modelotf_v4_mobilenetv3.keras")
