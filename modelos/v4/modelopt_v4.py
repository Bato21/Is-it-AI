"""
Inferencia con el modelo PyTorch v4 (MobileNetV3-Small, transfer learning).

Espejo de modelos/v2/modelopt_v2.py: mismo criterio, mismo formato de salida
(vector completo de probabilidades por imagen + grilla matplotlib con títulos),
para que el lado PyTorch quede parejo con el resto de las versiones.

Diferencia clave respecto a la v2: acá el modelo es MobileNetV3-Small y la
normalización ImageNet vive en el TRANSFORM (no dentro del modelo, como en TF).
Por eso el checkpoint guarda mean/std y acá los replicamos EXACTO: si el Normalize
no coincide con el del entrenamiento, los pesos preentrenados se des-sincronizan y
las predicciones se degradan en silencio.
"""
from pathlib import Path
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# --- CONFIG ---
# Raíz del repo = carpeta padre de 'modelos/'.
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "PyTorch" / "v4" / "modelopt_v4_mobilenetv3.pt"
IMG_DIR = ROOT / "imagenes_a_probar"
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]  # orden de base.classes

# --- Cargar modelo ---
if not MODEL_PATH.exists():
    raise SystemExit(f"No encontré el modelo en '{MODEL_PATH}'. Entrená primero PyTorch v4.")
ckpt = torch.load(MODEL_PATH, map_location="cpu")
class_names = ckpt.get("class_names", CLASS_NAMES)
img_size = tuple(ckpt.get("img_size", (224, 224)))  # MISMO tamaño que en entrenamiento (PT v4)
normalize_mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
normalize_std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])

# Reconstruir MobileNetV3-Small con la MISMA cabeza reemplazada del entrenamiento.
# weights=None: no descargamos ImageNet, cargamos nuestros pesos entrenados abajo.
model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Transform de EVAL EXACTO al del entrenamiento: Resize 224 + ToTensor + Normalize
# con la media/desvío leídos del checkpoint. Sin augmentation (es inferencia).
transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ]
)

# --- Reunir rutas de imágenes ---
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
rutas = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
if not rutas:
    raise SystemExit(f"No encontré imágenes en '{IMG_DIR}'")

# Mostrar solo un subconjunto aleatorio para que la grilla sea legible.
rutas = sample(rutas, k=min(6, len(rutas)))

# --- Cargar y apilar en un solo batch ---
batch = torch.stack([transform(Image.open(r).convert("RGB")) for r in rutas])  # (N,3,224,224)

# --- Predecir ---
with torch.no_grad():
    probs = F.softmax(model(batch), dim=1).numpy()

# --- Mostrar resultados ---
print(f"\n{'archivo':40s}    predicción        conf     [p0 / p1 / p2]")
print("-" * 85)
for r, p in zip(rutas, probs):
    idx = int(p.argmax())
    print(
        f"{r.name:40s} -> {class_names[idx]:15s} ({p[idx]:5.1%})   "
        f"[{p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}]"
    )

# --- Popup: cada imagen con su clasificación en el título ---
# Una sola ventana con una grilla pequeña y espaciada (máximo 6 imágenes).
n = len(rutas)
cols = min(3, n)
filas = (n + cols - 1) // cols
fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 5 * filas))
axes = np.atleast_1d(axes).ravel()  # normaliza a lista aunque sea 1 sola imagen

for ax, r, p in zip(axes, rutas, probs):
    idx = int(p.argmax())
    ax.imshow(Image.open(r).convert("RGB"))  # imagen original (sin resize) para verla bien
    ax.set_title(f"{class_names[idx]}  ({p[idx]:.1%})", fontsize=11)
    ax.axis("off")

# Apaga celdas sobrantes si la grilla no queda exacta.
for ax in axes[n:]:
    ax.axis("off")

fig.suptitle("Clasificación PyTorch v4 (MobileNetV3-Small)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
