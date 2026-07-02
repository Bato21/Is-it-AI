"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 2  (espejo de TensorFlow/v2/02_scripts.py)

Mismo objetivo que la v2 de TensorFlow: partir del baseline de la v1 y agregar
data augmentation domain-aware para atacar el sobreajuste. Todo lo demás se deja
igual que la v1 (misma CNN desde cero, 10 épocas) para poder comparar A/B el
efecto de la augmentation.

Nuevo respecto a la v1:
  - Bloque de data augmentation (solo en entrenamiento).
  - Curvas de accuracy Y loss (train vs val).
  - Matriz de confusión sobre validación.
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
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset")
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NUEVO EN v2: Data augmentation domain-aware ---
# En PyTorch la augmentation va en el transform del loader de train únicamente
# (en val/inferencia se usa el transform sin augmentation). Equivale al bloque
# 'data_augmentation' de Keras en TensorFlow/v2/02_scripts.py:
#   RandomRotation(0.05)=±18° · RandomZoom(0.1) · RandomBrightness(0.2) · RandomContrast(0.2)
# Sin flip a propósito: la diapositiva lleva texto y voltearla no es realista.
transform_train = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.RandomAffine(degrees=18, scale=(0.9, 1.1)),  # rotación + zoom
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # brillo + contraste
        transforms.ToTensor(),  # 0-255 -> 0-1 (equivale al Rescaling 1/255 de Keras)
    ]
)
transform_eval = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ]
)

# --- 2. Carga de datos (split 80/20 igual que la v1) ---
torch.manual_seed(SEED)
base = datasets.ImageFolder(DATA_DIR)
class_names = base.classes
print("Orden de clases:", class_names)
# Debe imprimir: ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']
num_classes = len(class_names)

n_val = max(1, int(len(base) * 0.2))
n_train = len(base) - n_val
idx_train, idx_val = random_split(
    range(len(base)), [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)

# Dos ImageFolder (mismo orden de archivos) para dar augmentation solo a train.
ds_train = Subset(datasets.ImageFolder(DATA_DIR, transform=transform_train), list(idx_train))
ds_val = Subset(datasets.ImageFolder(DATA_DIR, transform=transform_eval), list(idx_val))
train_dl = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(ds_val, batch_size=BATCH_SIZE)


# --- 3. Modelo (mismo baseline de la v1) ---
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
acc, val_acc, loss_hist, val_loss_hist = [], [], [], []
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = correr_epoca(train_dl, True)
    va_loss, va_acc = correr_epoca(val_dl, False)
    acc.append(tr_acc)
    val_acc.append(va_acc)
    loss_hist.append(tr_loss)
    val_loss_hist.append(va_loss)
    print(
        f"Época {epoch:2d}/{EPOCHS}  loss={tr_loss:.4f} acc={tr_acc:.4f}  "
        f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
    )

# --- 6. Curvas train vs val: accuracy Y loss ---
epochs_range = range(EPOCHS)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs_range, acc, label="train")
ax1.plot(epochs_range, val_acc, label="val")
ax1.legend()
ax1.set_title("Accuracy: train vs val")
ax1.set_xlabel("época")
ax1.set_ylabel("accuracy")
ax2.plot(epochs_range, loss_hist, label="train")
ax2.plot(epochs_range, val_loss_hist, label="val")
ax2.legend()
ax2.set_title("Loss: train vs val")
ax2.set_xlabel("época")
ax2.set_ylabel("loss")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_1.png", dpi=150, bbox_inches="tight")

# --- 7. Matriz de confusión sobre validación ---
y_true, y_pred = [], []
modelo.eval()
with torch.no_grad():
    for x, y in val_dl:
        out = modelo(x.to(DEVICE))
        y_true.extend(y.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(cm)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title("Matriz de confusión (val)")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j, i, cm[i, j], ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_2_matriz.png", dpi=150, bbox_inches="tight")
print("\nGuardado Figure_1.png y Figure_2_matriz.png")
