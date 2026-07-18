"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 3  (early stopping / convergencia)

Propósito experimental:
  La v2 quedó SUB-ENTRENADA. Con 10 épocas fijas las curvas de train y val
  todavía venían subiendo (train apenas 0.77, sin aplanarse), o sea que cortamos
  el entrenamiento antes de tiempo. La v3 NO agrega ninguna técnica nueva: mismo
  BaselineCNN, misma data augmentation domain-aware, mismo split (seed 123), mismo
  IMG_SIZE 180x180 y mismo batch 32 que la v2. El único cambio conceptual es dejar
  de cortar arbitrariamente a 10 épocas y entrenar hasta CONVERGENCIA con un freno
  automático (early stopping). Así el A/B contra la v2 es limpio: si algo cambia en
  la matriz de confusión, es por tiempo de entrenamiento, no por otra cosa.

  La pregunta que la v3 tiene que responder: la confusión residual de la v2 —en
  particular el caso 2->0 (una diapo 'saturada' clasificada como 'sin rastro')—
  ¿era ruido de no-convergencia o algo real del etiquetado? Si al converger
  desaparece, era falta de entrenamiento; si persiste, hay un problema de fondo.

Nuevo respecto a la v2:
  - Early stopping manual sobre val_loss (patience=8, restaurar mejores pesos).
  - Techo de 60 épocas en vez de 10 fijas.
  - Curvas y matriz calculadas sobre las épocas realmente corridas y los mejores
    pesos restaurados.
"""

import copy
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
# T0: ruta canónica ÚNICA para TF y PyTorch -> <repo>/dataset. Antes esta v3 apuntaba
# a 'data' y el lado TF a 'dataset': riesgo de entrenar cada framework sobre otra carpeta.
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
EPOCHS = 60  # NUEVO: techo alto. Ya no cortamos a 10; el early stopping decide.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NUEVO EN v3: hiperparámetros del early stopping.
# En Keras esto sería UNA línea de callback:
#   EarlyStopping(monitor="val_loss", patience=8, min_delta=0, restore_best_weights=True)
# En PyTorch no existe ese callback: hay que implementarlo a mano (bloque 5).
PATIENCE = 8
MIN_DELTA = 0.0

# --- Data augmentation domain-aware (IDÉNTICA a la v2) ---
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

# --- 2. Carga de datos (split 80/20 IDÉNTICO a la v2, mismo seed) ---
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


# --- 3. Modelo (mismo baseline de la v1/v2) ---
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


# --- 5. Entrenamiento con EARLY STOPPING manual ---
# CONTRASTE DE FRAMEWORKS (clave para el curso):
#   En Keras el early stopping es una línea que se pasa a model.fit():
#     EarlyStopping(monitor="val_loss", patience=8, min_delta=0, restore_best_weights=True)
#   En PyTorch el bucle de entrenamiento es nuestro, así que el callback también:
#     - cuando val_loss mejora (baja más que min_delta): guardamos una COPIA PROFUNDA
#       de los pesos (copy.deepcopy del state_dict) y reseteamos el contador;
#     - cuando no mejora: incrementamos el contador de paciencia;
#     - al llegar a patience: cortamos con break;
#     - al final: restauramos los mejores pesos ANTES de evaluar y guardar.
#   El deepcopy es imprescindible: state_dict() devuelve referencias a los tensores
#   vivos del modelo; sin copiar, "los mejores pesos" se pisarían en la época siguiente.
acc, val_acc, loss_hist, val_loss_hist = [], [], [], []
mejor_val_loss = float("inf")
mejor_epoca = 0
mejor_state = copy.deepcopy(modelo.state_dict())
epocas_sin_mejora = 0

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

    # ¿Mejoró la val_loss más que min_delta? -> guardar pesos y resetear paciencia.
    if va_loss < mejor_val_loss - MIN_DELTA:
        mejor_val_loss = va_loss
        mejor_epoca = epoch
        mejor_state = copy.deepcopy(modelo.state_dict())
        epocas_sin_mejora = 0
    else:
        epocas_sin_mejora += 1
        if epocas_sin_mejora >= PATIENCE:
            print(
                f"\nEarly stopping en la época {epoch}: "
                f"val_loss no mejora hace {PATIENCE} épocas."
            )
            break

# Restaurar los MEJORES pesos antes de evaluar y guardar (restore_best_weights=True).
modelo.load_state_dict(mejor_state)
epocas_corridas = len(loss_hist)
print(
    f"\nMejor modelo: época {mejor_epoca}/{epocas_corridas}  "
    f"val_loss={mejor_val_loss:.4f}  val_acc={val_acc[mejor_epoca - 1]:.4f}  "
    f"(train loss={loss_hist[mejor_epoca - 1]:.4f} acc={acc[mejor_epoca - 1]:.4f})"
)

# --- 6. Curvas train vs val: accuracy Y loss ---
# Usamos las épocas EFECTIVAMENTE corridas (no range(EPOCHS) hardcodeado), porque
# el early stopping pudo cortar antes de 60. Línea vertical en la mejor época.
epochs_range = range(1, epocas_corridas + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs_range, acc, label="train")
ax1.plot(epochs_range, val_acc, label="val")
ax1.axvline(mejor_epoca, ls="--", color="gray", label=f"mejor época ({mejor_epoca})")
ax1.legend()
ax1.set_title("Accuracy: train vs val")
ax1.set_xlabel("época")
ax1.set_ylabel("accuracy")
ax2.plot(epochs_range, loss_hist, label="train")
ax2.plot(epochs_range, val_loss_hist, label="val")
ax2.axvline(mejor_epoca, ls="--", color="gray", label=f"mejor época ({mejor_epoca})")
ax2.legend()
ax2.set_title("Loss: train vs val")
ax2.set_xlabel("época")
ax2.set_ylabel("loss")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_1.png", dpi=150, bbox_inches="tight")

# --- 7. Matriz de confusión sobre validación (con los MEJORES pesos restaurados) ---
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

# --- 8. Guardar el modelo entrenado (mismo formato de dict que la v2) ---
MODEL_PATH = Path(__file__).resolve().parent / "modelopt_v3_convergencia.pt"
torch.save(
    {"state_dict": modelo.state_dict(), "class_names": class_names, "img_size": IMG_SIZE},
    MODEL_PATH,
)
print(f"Modelo guardado en {MODEL_PATH.name}")
