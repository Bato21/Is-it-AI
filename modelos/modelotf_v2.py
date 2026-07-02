"""
Clasificar un lote de imágenes con el modelo ya entrenado.
"""
import tensorflow as tf
import numpy as np
from pathlib import Path

# --- CONFIG ---
MODEL_PATH  = "modelos/02_augmentation.keras"     # ruta a tu modelo guardado
IMG_DIR     = "fotos_a_clasificar"                # carpeta con las imágenes a clasificar
IMG_SIZE    = (224, 224)                          # MISMO tamaño que en entrenamiento
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]  # MISMO orden que train_ds.class_names

# --- Cargar modelo ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Reunir rutas de imágenes ---
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
rutas = sorted(p for p in Path(IMG_DIR).iterdir() if p.suffix.lower() in EXTS)
if not rutas:
    raise SystemExit(f"No encontré imágenes en '{IMG_DIR}'")

# --- Cargar y apilar en un solo batch ---
imgs = [
    tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(r, target_size=IMG_SIZE)
    )
    for r in rutas
]
batch = tf.stack(imgs)                             # (N, 224, 224, 3)

# --- Predecir ---
probs = model.predict(batch, verbose=0)           

# --- Mostrar resultados ---
print(f"\n{'archivo':35s}    predicción        conf     [p0 / p1 / p2]")
print("-" * 80)
for r, p in zip(rutas, probs):
    idx = int(np.argmax(p))
    print(
        f"{r.name:35s} -> {CLASS_NAMES[idx]:15s} ({p[idx]:5.1%})   "
        f"[{p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}]"
    )