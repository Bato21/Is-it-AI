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

# --- 6b. ROC-AUC One-vs-Rest, calculado A MANO (mismo criterio que la matriz de confusión) ---
# ROC es binaria por naturaleza. Con 3 clases usamos One-vs-Rest: para cada clase i se arma
# el problema "i contra el resto" y se barre el umbral sobre su score softmax.
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


# --- 7. Matriz de confusión sobre validación ---
# La matriz usa la predicción DURA (argmax); ROC-AUC necesita el score CONTINUO (softmax),
# así que en el mismo recorrido recolectamos ambas cosas.
y_true, y_pred, y_prob = [], [], []
modelo.eval()
with torch.no_grad():
    for x, y in val_dl:
        out = modelo(x.to(DEVICE))
        y_true.extend(y.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_prob.append(torch.softmax(out, dim=1).cpu().numpy())
y_prob = np.concatenate(y_prob)
y_true_arr = np.array(y_true)

cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(cm)

# ROC-AUC One-vs-Rest sobre las probabilidades softmax de validación.
curvas_roc, micro_roc, macro_roc = curvas_roc_ovr(y_true_arr, y_prob)
print("\nROC-AUC One-vs-Rest (val):")
for i in range(num_classes):
    print(f"  {class_names[i]:<16} AUC = {curvas_roc[i][2]:.4f}")
print(f"  {'macro-promedio':<16} AUC = {macro_roc:.4f}")
print(f"  {'micro-promedio':<16} AUC = {micro_roc[2]:.4f}")

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

# --- 7b. Curvas ROC One-vs-Rest (val) ---
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
ax.set_title(f"Curvas ROC One-vs-Rest — v2 (val)\nAUC macro = {macro_roc:.3f}",
             fontweight="bold", color="#1F3864")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_roc_v2.png", dpi=150, bbox_inches="tight")
print("\nGuardado Figure_1.png, Figure_2_matriz.png y Figure_roc_v2.png")

# --- 8. Guardar el modelo entrenado (espejo del model.save de TensorFlow/v2) ---
# Queda junto al script, igual que modelotf_v2_augmentation.keras en TensorFlow/v2/.
# Lo consume modelos/modelopt_v2.py para que el profe pueda probarlo.
MODEL_PATH = Path(__file__).resolve().parent / "modelopt_v2_augmentation.pt"
torch.save(
    {"state_dict": modelo.state_dict(), "class_names": class_names, "img_size": IMG_SIZE},
    MODEL_PATH,
)
print(f"Modelo guardado en {MODEL_PATH.name}")
