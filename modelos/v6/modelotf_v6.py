"""
Inferencia con el modelo TensorFlow v6 (MobileNetV3-Small + cabeza optimizada con Optuna).

Espejo de modelos/v4/modelotf_v4.py y del lado PyTorch modelos/v6/modelopt_v6.py:
vector completo de probabilidades por imagen + grilla matplotlib con la clasificación
en el título. Consumo INDEPENDIENTE del entrenamiento (solo lee el .keras).

El .keras de la v6 (modelotf_v6_optuna.keras) es un modelo de punta a punta: backbone
congelado con su preprocesamiento ImageNet + la cabeza que Optuna eligió. Igual que en la
v4, se le pasan imágenes 0-255 CRUDAS y el modelo hace el preprocesamiento solo.
CONTRASTE DE FRAMEWORKS: en PyTorch (modelopt_v6.py) la normalización vive en el transform
y la inferencia la replica a mano; acá viaja dentro del modelo.
"""
from pathlib import Path
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# --- CONFIG ---
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "TensorFlow" / "v6" / "modelotf_v6_optuna.keras"
IMG_DIR = ROOT / "imagenes_a_probar"
IMG_SIZE = (224, 224)                          # MISMO tamaño que en entrenamiento (TF v6)
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]

# --- Cargar modelo ---
if not MODEL_PATH.exists():
    raise SystemExit(
        f"No encontré el modelo en '{MODEL_PATH}'.\n"
        "Entrená primero TF v6:  python TensorFlow/v6/06_scripts.py"
    )
model = tf.keras.models.load_model(MODEL_PATH)

# --- Reunir rutas de imágenes ---
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
rutas = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
if not rutas:
    raise SystemExit(f"No encontré imágenes en '{IMG_DIR}'")
rutas = sample(rutas, k=min(6, len(rutas)))

# --- Cargar y apilar (SIN normalizar: el .keras trae el preprocesamiento adentro) ---
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
n = len(rutas)
cols = min(3, n)
filas = (n + cols - 1) // cols
fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 5 * filas))
axes = np.atleast_1d(axes).ravel()

for ax, r, p in zip(axes, rutas, probs):
    idx = int(np.argmax(p))
    ax.imshow(tf.keras.utils.load_img(r))
    ax.set_title(f"{CLASS_NAMES[idx]}  ({p[idx]:.1%})", fontsize=11)
    ax.axis("off")
for ax in axes[n:]:
    ax.axis("off")

fig.suptitle("Clasificación TF v6 (MobileNetV3-Small + cabeza Optuna)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
