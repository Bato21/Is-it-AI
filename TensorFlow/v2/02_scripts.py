import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
DATA_DIR = "../../data"
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

# --- NUEVO EN v2: Data augmentation domain-aware ---
# Se define como bloque de capas. Keras las aplica solo en entrenamiento
# y las apaga solas en validación/inferencia — no hay que hacer nada manual.
# (En PyTorch esto iría en el transform del loader de train únicamente.)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),     # foto un poco torcida
    tf.keras.layers.RandomZoom(0.1),          # distancia variable a la diapo
    tf.keras.layers.RandomBrightness(0.2),    # iluminación / reflejo de proyector
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# --- 4. Modelo  ---
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

# --- 6. Entrenamiento ---
# Mantenenmos EPOCHS=10 igual que v1 para poder comparar A/B el efecto de la
# augmentation con todo lo demás igual. Con augmentation el modelo tarda más
# en memorizar, así que quizás en v3 convenga subirlo.
EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# --- 7. Curvas train vs val: accuracy Y loss ---
# NUEVO en v2: agreganos loss
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(EPOCHS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, acc, label="train")
ax1.plot(epochs_range, val_acc, label="val")
ax1.legend()
ax1.set_title("Accuracy: train vs val")
ax1.set_xlabel("época")
ax1.set_ylabel("accuracy")

ax2.plot(epochs_range, loss, label="train")
ax2.plot(epochs_range, val_loss, label="val")
ax2.legend()
ax2.set_title("Loss: train vs val")
ax2.set_xlabel("época")
ax2.set_ylabel("loss")

plt.tight_layout()
plt.show()

# --- NUEVO EN v2: Matriz de confusión sobre validación ---
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
plt.show()

model.save("modelotf_v2_augmentation.keras")