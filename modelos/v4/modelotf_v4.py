"""
Inferencia con el modelo TensorFlow v4 (MobileNetV3-Small, transfer learning).

Espejo de modelos/v2/modelotf_v2.py (mismo criterio y formato de salida) y del lado
PyTorch modelos/v4/modelopt_v4.py: vector completo de probabilidades por imagen +
grilla matplotlib con la clasificación en el título, para que TF y PyTorch queden
parejos en todas las versiones.

Diferencia clave respecto a la v2: el modelo es MobileNetV3-Small y la normalización
ImageNet vive DENTRO del .keras (include_preprocessing=True). Por eso NO normalizamos
a mano: le pasamos la imagen 0-255 cruda y el modelo hace el preprocesamiento solo.
CONTRASTE DE FRAMEWORKS: en el lado PyTorch (modelopt_v4.py) la normalización vive en
el TRANSFORM, fuera del modelo, y la inferencia tiene que replicar mean/std del
checkpoint o se des-sincroniza en silencio; acá viaja dentro del modelo.
"""
from pathlib import Path
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# --- CONFIG ---
# Raíz del repo = parents[2] (este archivo está en modelos/v4/).
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "TensorFlow" / "v4" / "modelotf_v4_mobilenetv3.keras"
IMG_DIR = ROOT / "imagenes_a_probar"
IMG_SIZE = (224, 224)                          # MISMO tamaño que en entrenamiento (TF v4)
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]  # orden de train_ds.class_names

# --- Cargar modelo ---
if not MODEL_PATH.exists():
    raise SystemExit(
        f"No encontré el modelo en '{MODEL_PATH}'.\n"
        "Entrená primero TF v4:  python TensorFlow/v4/04_scripts.py"
    )
model = tf.keras.models.load_model(MODEL_PATH)

# --- Reunir rutas de imágenes ---
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
rutas = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
if not rutas:
    raise SystemExit(f"No encontré imágenes en '{IMG_DIR}'")

# Mostrar solo un subconjunto aleatorio para que la grilla sea legible.
rutas = sample(rutas, k=min(6, len(rutas)))

# --- Cargar y apilar en un solo batch ---
# SIN normalizar / SIN dividir por 255: el .keras de la v4 trae el preprocesamiento
# ImageNet adentro (a diferencia de PyTorch, que lo hace en el transform).
imgs = [
    tf.keras.utils.img_to_array(tf.keras.utils.load_img(r, target_size=IMG_SIZE))
    for r in rutas
]
batch = tf.stack(imgs)                         # (N, 224, 224, 3), valores 0-255

# --- Predecir ---
probs = model.predict(batch, verbose=0)

# --- Mostrar resultados ---
print(f"\n{'archivo':40s}    predicción        conf     [p0 / p1 / p2]")
print("-" * 85)
for r, p in zip(rutas, probs):
    idx = int(np.argmax(p))
    print(
        f"{r.name:40s} -> {CLASS_NAMES[idx]:15s} ({p[idx]:5.1%})   "
        f"[{p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}]"
    )

# --- Popup: cada imagen con su clasificación en el título ---
# Una sola ventana con una grilla pequeña y espaciada (máximo 6 imágenes).
n = len(rutas)
cols = min(3, n)
filas = (n + cols - 1) // cols
fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 5 * filas))
axes = np.atleast_1d(axes).ravel()             # normaliza a lista aunque sea 1 sola imagen

for ax, r, p in zip(axes, rutas, probs):
    idx = int(np.argmax(p))
    ax.imshow(tf.keras.utils.load_img(r))      # imagen original (sin resize) para verla bien
    ax.set_title(f"{CLASS_NAMES[idx]}  ({p[idx]:.1%})", fontsize=11)
    ax.axis("off")

# Apaga celdas sobrantes si la grilla no queda exacta.
for ax in axes[n:]:
    ax.axis("off")

fig.suptitle("Clasificación TF v4 (MobileNetV3-Small)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
