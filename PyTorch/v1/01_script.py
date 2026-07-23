"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 1  (espejo de TensorFlow/v1/01_script.py)

Mismo objetivo que la v1 de TensorFlow: entrenar de punta a punta y, a propósito,
sobreajustar. Ver ese sobreajuste motiva la v2 (transfer learning).
Sin augmentation, sin dropout, sin regularización, sin transfer learning.

La arquitectura replica exactamente la de TensorFlow v1:
  Conv(16) -> Pool -> Conv(32) -> Pool -> Flatten -> Dense(64) -> Dense(3)
Con entrada 180x180 el flatten da 32*43*43 = 59168 (igual que en Keras).
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
# T0: ruta canónica ÚNICA para TF y PyTorch -> <repo>/dataset. Antes había 'data' y
# 'dataset' mezclados y cada framework podía entrenar sobre carpetas distintas.
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset")
if not Path(DATA_DIR).is_dir():
    raise SystemExit(
        f"No existe la carpeta de datos: {DATA_DIR}\n"
        "Colocá el dataset ahí (subcarpetas 0_sin_ia/ 1_rastro_ia/ 2_saturada_ia/), o\n"
        "generá datos sintéticos para probar el flujo de punta a punta:\n"
        "    python documentacion/crear_datos_prueba.py --por-clase 30"
    )
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Carga de datos ---
torch.manual_seed(SEED)
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),  # 0-255 -> 0-1 (equivale al Rescaling 1/255 de Keras)
    ]
)

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print("Orden de clases:", dataset.classes)
# Debe imprimir: ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']
num_classes = len(dataset.classes)

n_val = max(1, int(len(dataset) * 0.2))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)


# --- 3. Modelo baseline ---
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


modelo = BaselineCNN(num_classes).to(DEVICE)
print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()):,}")

# --- 4. Compilación (optimizador + pérdida) ---
criterion = nn.CrossEntropyLoss()  # incluye softmax internamente
optimizer = torch.optim.Adam(modelo.parameters())


def correr_epoca(dl, entrenar: bool) -> tuple[float, float]:
    modelo.train(entrenar)
    total, correctos, suma_loss = 0, 0, 0.0
    torch.set_grad_enabled(entrenar)
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if entrenar:
            optimizer.zero_grad()
        out = modelo(x)
        loss = criterion(out, y)
        if entrenar:
            loss.backward()
            optimizer.step()
        suma_loss += loss.item() * x.size(0)
        correctos += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return suma_loss / total, correctos / total


# --- 5. Entrenamiento ---
hist_acc, hist_val_acc = [], []
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = correr_epoca(train_dl, True)
    va_loss, va_acc = correr_epoca(val_dl, False)
    hist_acc.append(tr_acc)
    hist_val_acc.append(va_acc)
    print(
        f"Época {epoch:2d}/{EPOCHS}  loss={tr_loss:.4f} acc={tr_acc:.4f}  "
        f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
    )

# --- 6. Curvas train vs val (acá se ve el sobreajuste) ---
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), hist_acc, label="train")
plt.plot(range(1, EPOCHS + 1), hist_val_acc, label="val")
plt.legend()
plt.title("Accuracy: train vs val")
plt.xlabel("época")
plt.ylabel("accuracy")
plt.savefig(Path(__file__).resolve().parent / "Figure_1.png", dpi=150, bbox_inches="tight")

# --- 7. ROC-AUC One-vs-Rest, calculado A MANO ---
# ROC es binaria por naturaleza. Con 3 clases usamos One-vs-Rest: para cada clase i se arma
# el problema "i contra el resto" y se barre el umbral sobre su score softmax. En la v1
# (baseline que sobreajusta a propósito) sirve como línea base contra la que comparan v2+.
class_names = dataset.classes


def roc_binaria(y_bin: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Curva ROC de un problema binario, a mano: ordena por score descendente y acumula
    verdaderos/falsos positivos umbral a umbral. AUC por regla trapezoidal.
    Los scores softmax son continuos, así que los empates son despreciables."""
    orden = np.argsort(-scores, kind="mergesort")
    y = y_bin[orden].astype(float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    tpr = np.concatenate([[0.0], np.cumsum(y) / n_pos]) if n_pos else np.zeros(len(y) + 1)
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / n_neg]) if n_neg else np.zeros(len(y) + 1)
    # AUC por regla trapezoidal, a mano (compatible con numpy 1.x y 2.x).
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def curvas_roc_ovr(y_true_arr: np.ndarray, y_prob_arr: np.ndarray):
    """Devuelve: dict {clase -> (fpr, tpr, auc)}, la curva micro-promedio y el AUC macro."""
    curvas = {}
    for i in range(num_classes):
        curvas[i] = roc_binaria((y_true_arr == i).astype(int), y_prob_arr[:, i])
    macro = float(np.mean([curvas[i][2] for i in range(num_classes)]))
    onehot = np.eye(num_classes)[y_true_arr].ravel()
    micro = roc_binaria(onehot.astype(int), y_prob_arr.ravel())
    return curvas, micro, macro


# Pase de evaluación sobre validación: recolectamos las probabilidades softmax por clase.
y_true, y_prob = [], []
modelo.eval()
with torch.no_grad():
    for x, y in val_dl:
        out = modelo(x.to(DEVICE))
        y_true.extend(y.numpy())
        y_prob.append(torch.softmax(out, dim=1).cpu().numpy())
y_prob = np.concatenate(y_prob)
y_true_arr = np.array(y_true)

curvas_roc, micro_roc, macro_roc = curvas_roc_ovr(y_true_arr, y_prob)
print("\nROC-AUC One-vs-Rest (val):")
for i in range(num_classes):
    print(f"  {class_names[i]:<16} AUC = {curvas_roc[i][2]:.4f}")
print(f"  {'macro-promedio':<16} AUC = {macro_roc:.4f}")
print(f"  {'micro-promedio':<16} AUC = {micro_roc[2]:.4f}")

colores = ["#1F3864", "#B45309", "#2E7D32", "#7B1FA2"]
fig, ax = plt.subplots(figsize=(7, 7))
for i, (fpr, tpr, auc) in curvas_roc.items():
    ax.plot(fpr, tpr, lw=2, color=colores[i % len(colores)],
            label=f"{class_names[i]} (AUC={auc:.3f})")
ax.plot(micro_roc[0], micro_roc[1], lw=2, ls=":", color="gray",
        label=f"micro-promedio (AUC={micro_roc[2]:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="azar (AUC=0.500)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("Tasa de falsos positivos (FPR)")
ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
ax.set_title(f"Curvas ROC One-vs-Rest — v1 (val)\nAUC macro = {macro_roc:.3f}",
             fontweight="bold", color="#1F3864")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_roc_v1.png", dpi=150, bbox_inches="tight")
print("\nGuardado Figure_1.png y Figure_roc_v1.png")
