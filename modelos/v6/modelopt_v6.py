"""
Inferencia con el modelo PyTorch v6 (MobileNetV3-Small + cabeza optimizada con Optuna).

Espejo de modelos/v4/modelopt_v4.py y del lado TF modelos/v6/modelotf_v6.py: vector
completo de probabilidades por imagen + grilla matplotlib con títulos. Consumo
INDEPENDIENTE del entrenamiento (solo lee el checkpoint .pt).

El checkpoint de la v6 guarda la cabeza óptima (cabeza_state_dict), sus best_params (para
reconstruir la MISMA arquitectura que eligió Optuna) y la normalización ImageNet. Acá se
rearma: backbone congelado (features+avgpool+flatten) -> cabeza. La normalización se
replica EXACTO desde el checkpoint (contraste con TF, que la lleva dentro del modelo).
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
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "PyTorch" / "v6" / "modelopt_v6_optuna.pt"
IMG_DIR = ROOT / "imagenes_a_probar"
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]

# --- Cargar checkpoint ---
if not MODEL_PATH.exists():
    raise SystemExit(f"No encontré el modelo en '{MODEL_PATH}'. Entrená primero PyTorch v6.")
ckpt = torch.load(MODEL_PATH, map_location="cpu")
class_names = ckpt.get("class_names", CLASS_NAMES)
img_size = tuple(ckpt.get("img_size", (224, 224)))
normalize_mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
normalize_std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])
best_params = ckpt["best_params"]
emb_dim = ckpt.get("emb_dim", 576)


# --- Reconstruir extractor (backbone congelado) + cabeza óptima ---
# weights=None: no descargamos ImageNet; el extractor solo aporta la arquitectura y sus
# pesos vienen en el state_dict guardado por el entrenamiento (features del backbone).
# OJO: acá reconstruimos la MISMA cabeza que construir_cabeza() del script de entrenamiento
# a partir de best_params, o el state_dict no calza.
backbone = mobilenet_v3_small(weights=None)
extractor = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten()).eval()


def construir_cabeza(params) -> nn.Module:
    capas, entrada = [], emb_dim
    for _ in range(params["n_capas"]):
        capas += [nn.Linear(entrada, params["units"]), nn.ReLU(), nn.Dropout(params["dropout"])]
        entrada = params["units"]
    capas.append(nn.Linear(entrada, len(class_names)))
    return nn.Sequential(*capas)


cabeza = construir_cabeza(best_params)
cabeza.load_state_dict(ckpt["cabeza_state_dict"])
cabeza.eval()

# Transform de EVAL EXACTO al del entrenamiento (Resize + ToTensor + Normalize ImageNet).
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std),
])

# --- Reunir rutas de imágenes ---
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
rutas = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
if not rutas:
    raise SystemExit(f"No encontré imágenes en '{IMG_DIR}'")
rutas = sample(rutas, k=min(6, len(rutas)))

# --- Cargar, apilar y predecir (backbone -> cabeza -> softmax) ---
batch = torch.stack([transform(Image.open(r).convert("RGB")) for r in rutas])
with torch.no_grad():
    logits = cabeza(extractor(batch))
    probs = F.softmax(logits, dim=1).numpy()

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
n = len(rutas)
cols = min(3, n)
filas = (n + cols - 1) // cols
fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 5 * filas))
axes = np.atleast_1d(axes).ravel()

for ax, r, p in zip(axes, rutas, probs):
    idx = int(p.argmax())
    ax.imshow(Image.open(r).convert("RGB"))
    ax.set_title(f"{class_names[idx]}  ({p[idx]:.1%})", fontsize=11)
    ax.axis("off")
for ax in axes[n:]:
    ax.axis("off")

fig.suptitle("Clasificación PyTorch v6 (MobileNetV3-Small + cabeza Optuna)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
