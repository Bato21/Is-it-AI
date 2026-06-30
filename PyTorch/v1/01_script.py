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
    sys.stdout.reconfigure(encoding="utf-8")   # consola UTF-8 en Windows
except Exception:
    pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset")
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Carga de datos ---
torch.manual_seed(SEED)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),          # 0-255 -> 0-1 (equivale al Rescaling 1/255 de Keras)
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print("Orden de clases:", dataset.classes)
# Debe imprimir: ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']
num_classes = len(dataset.classes)

n_val = max(1, int(len(dataset) * 0.2))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- 3. Modelo baseline ---
class BaselineCNN(nn.Module):
    def __init__(self, num_clases: int):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 43 * 43, 64), nn.ReLU(),
            nn.Linear(64, num_clases),
        )

    def forward(self, x):
        return self.red(x)

modelo = BaselineCNN(num_classes).to(DEVICE)
print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()):,}")

# --- 4. Compilación (optimizador + pérdida) ---
criterion = nn.CrossEntropyLoss()       # incluye softmax internamente
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
    hist_acc.append(tr_acc); hist_val_acc.append(va_acc)
    print(f"Época {epoch:2d}/{EPOCHS}  loss={tr_loss:.4f} acc={tr_acc:.4f}  "
          f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

# --- 6. Curvas train vs val (acá se ve el sobreajuste) ---
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), hist_acc, label="train")
plt.plot(range(1, EPOCHS + 1), hist_val_acc, label="val")
plt.legend(); plt.title("Accuracy: train vs val")
plt.xlabel("época"); plt.ylabel("accuracy")
plt.savefig(Path(__file__).resolve().parent / "curvas_v1.png", dpi=150, bbox_inches="tight")
print("Guardado curvas_v1.png")
