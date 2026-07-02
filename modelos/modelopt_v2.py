"""
Inferencia con el modelo PyTorch v2 (data augmentation) ya entrenado.

Espejo de modelotf_v2.py: mismo criterio, mismo formato de salida, para que el
lado PyTorch y el lado TensorFlow queden parejos.
"""
from pathlib import Path
from random import sample

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- CONFIG ---
# Raíz del repo = carpeta padre de 'modelos/'.
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "PyTorch" / "v2" / "modelopt_v2_augmentation.pt"
IMG_DIR = ROOT / "imagenes_a_probar"
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]  # orden de base.classes


# --- Arquitectura (misma BaselineCNN de PyTorch/v2) ---
class BaselineCNN(nn.Module):
    def __init__(self, num_clases: int):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 43 * 43, 64),
            nn.ReLU(),
            nn.Linear(64, num_clases),
        )

    def forward(self, x):
        return self.red(x)


# --- Cargar modelo ---
if not MODEL_PATH.exists():
    raise SystemExit(f"No encontré el modelo en '{MODEL_PATH}'. Entrená primero PyTorch v2.")
ckpt = torch.load(MODEL_PATH, map_location="cpu")
class_names = ckpt.get("class_names", CLASS_NAMES)
img_size = tuple(ckpt.get("img_size", (180, 180)))  # MISMO tamaño que en entrenamiento (PT v2)
model = BaselineCNN(len(class_names))
model.load_state_dict(ckpt["state_dict"])
model.eval()

transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

# --- Reunir rutas de imágenes ---
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
rutas = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
if not rutas:
    raise SystemExit(f"No encontré imágenes en '{IMG_DIR}'")

# Mostrar solo un subconjunto aleatorio para que la grilla sea legible.
rutas = sample(rutas, k=min(6, len(rutas)))

# --- Cargar y apilar en un solo batch ---
batch = torch.stack([transform(Image.open(r).convert("RGB")) for r in rutas])  # (N,3,180,180)

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
import numpy as np

axes = np.atleast_1d(axes).ravel()  # normaliza a lista aunque sea 1 sola imagen

for ax, r, p in zip(axes, rutas, probs):
    idx = int(p.argmax())
    ax.imshow(Image.open(r).convert("RGB"))  # imagen original (sin resize) para verla bien
    ax.set_title(f"{class_names[idx]}  ({p[idx]:.1%})", fontsize=11)
    ax.axis("off")

# Apaga celdas sobrantes si la grilla no queda exacta.
for ax in axes[n:]:
    ax.axis("off")

fig.suptitle("Clasificación PyTorch v2", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
