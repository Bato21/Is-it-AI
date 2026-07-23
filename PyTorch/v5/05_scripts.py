"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 5  (clases DESBALANCEADAS + corrección con pesos de clase)

Propósito experimental:
  La v4 entrenó sobre un dataset balanceado (~100/100/100). La v5 hace UN solo cambio
  conceptual: los datos. Entrena sobre dataset_desbalanceado/ (100/50/25 en las clases
  0/1/2), una distribución más realista de producción (la mayoría de las presentaciones
  son humanas), y analiza el efecto del desbalance y su corrección.

  MISMO modelo que la v4 (MobileNetV3-Small congelada + cabeza nueva), MISMO seed/split
  80-20, MISMO early stopping. Lo único que cambia es el dataset y su corrección.

  En UNA sola corrida se entrenan DOS variantes para el A/B:
    (a) SIN corrección  -> línea base sobre datos desbalanceados.
    (b) CON pesos de clase -> se compensa el desbalance en la pérdida.
  Fórmula estándar de pesos:  w_c = n_total / (n_clases * n_c)   (n_c = casos de la
  clase c en TRAIN). La clase minoritaria recibe más peso.

Nuevo respecto a la v4:
  - Dataset desbalanceado (dataset_desbalanceado/, generado por
    documentacion/crear_subset_desbalanceado.py).
  - Dos variantes en la misma corrida (sin/con pesos) + reporte por clase de cada una.
  - Soporte por clase en validación impreso con la advertencia de leer el recall de la
    clase 2 en NÚMEROS ABSOLUTOS (queda con ~5 casos en val).

CONTRASTE DE FRAMEWORKS (espejo de TensorFlow/v5/05_scripts.py):
  En Keras la corrección es un argumento de fit(): class_weight={...}. En PyTorch se
  elige entre PESAR la pérdida (nn.CrossEntropyLoss(weight=...)) o RE-MUESTREAR con un
  WeightedRandomSampler en el DataLoader. Acá usamos el peso en la pérdida (más directo);
  el sampler queda mencionado como alternativa equivalente.
"""

import collections
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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
# v5: dataset DESBALANCEADO (no dataset/). Se genera aparte y no toca el original.
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset_desbalanceado")
if not Path(DATA_DIR).is_dir():
    raise SystemExit(
        f"No existe la carpeta de datos: {DATA_DIR}\n"
        "Generá el subset desbalanceado primero:\n"
        "    python documentacion/crear_subset_desbalanceado.py"
    )
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 60
PATIENCE = 8
MIN_DELTA = 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomAffine(degrees=18, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])
transform_eval = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

# --- 2. Carga de datos (split 80/20 IDÉNTICO a la v4, mismo seed) ---
torch.manual_seed(SEED)
base = datasets.ImageFolder(DATA_DIR)
class_names = base.classes
print("Orden de clases:", class_names)
num_classes = len(class_names)

n_val = max(1, int(len(base) * 0.2))
n_train = len(base) - n_val
idx_train, idx_val = random_split(
    range(len(base)), [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)
idx_train, idx_val = list(idx_train), list(idx_val)

ds_train = Subset(datasets.ImageFolder(DATA_DIR, transform=transform_train), idx_train)
ds_val = Subset(datasets.ImageFolder(DATA_DIR, transform=transform_eval), idx_val)
train_dl = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(ds_val, batch_size=BATCH_SIZE)

# Soporte por clase (sin cargar imágenes: se lee de base.targets).
train_counts = collections.Counter(base.targets[i] for i in idx_train)
val_counts = collections.Counter(base.targets[i] for i in idx_val)
print("\nSoporte por clase:")
print(f"{'clase':<16}{'train':>7}{'val':>6}")
for i, nombre in enumerate(class_names):
    print(f"{nombre:<16}{train_counts[i]:>7}{val_counts[i]:>6}")
print(
    "\nAVISO: con 80/20 sobre ~175 imágenes, la clase 2 queda con pocos casos en val\n"
    f"       ({val_counts[2]} imágenes). Leé su recall en NÚMEROS ABSOLUTOS, no solo en %."
)

# --- Pesos de clase:  w_c = n_total / (n_clases * n_c)  ---
n_total = sum(train_counts.values())
pesos_lista = [n_total / (num_classes * train_counts[i]) for i in range(num_classes)]
pesos = torch.tensor(pesos_lista, dtype=torch.float32, device=DEVICE)
print("\nPesos de clase (para la variante con corrección):")
for i, nombre in enumerate(class_names):
    print(f"  {nombre:<16} w={pesos_lista[i]:.3f}")


# --- 3. Reporte por clase (a mano desde la matriz de confusión) ---
# Duplicado a propósito en cada lado (TF y PT) para no acoplar entornos.
def reporte_por_clase(cm: np.ndarray, nombres: list[str]) -> None:
    total = cm.sum()
    print(f"{'clase':<16}{'precision':>10}{'recall':>9}{'f1':>7}{'soporte':>9}")
    print("-" * 51)
    precs, recs, f1s = [], [], []
    for i in range(len(nombres)):
        soporte = cm[i, :].sum()
        predichos = cm[:, i].sum()
        recall = cm[i, i] / soporte if soporte else 0.0
        precision = cm[i, i] / predichos if predichos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
        print(f"{nombres[i]:<16}{precision:>10.3f}{recall:>9.3f}{f1:>7.3f}{int(soporte):>9}")
    print("-" * 51)
    print(f"{'macro avg':<16}{np.mean(precs):>10.3f}{np.mean(recs):>9.3f}{np.mean(f1s):>7.3f}{int(total):>9}")
    acc = np.trace(cm) / total if total else 0.0
    print(f"accuracy: {acc:.3f}  ({int(np.trace(cm))}/{int(total)})")


# --- 3b. ROC-AUC One-vs-Rest, calculado A MANO (mismo criterio que reporte_por_clase) ---
# ROC es binaria por naturaleza. Con 3 clases usamos One-vs-Rest: para cada clase i se arma
# el problema "i contra el resto" y se barre el umbral sobre su score softmax. En la v5 esto
# se calcula para las DOS variantes (sin/con pesos) para comparar el efecto del desbalance:
# la corrección debería mejorar sobre todo el AUC de la clase minoritaria (2_saturada_ia).
# AVISO: la clase 2 tiene pocos casos en val, así que su curva ROC sale escalonada (gruesa).
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


def curvas_roc_ovr(y_true: np.ndarray, y_prob: np.ndarray):
    """Devuelve: dict {clase -> (fpr, tpr, auc)}, la curva micro-promedio y el AUC macro."""
    curvas = {}
    for i in range(num_classes):
        curvas[i] = roc_binaria((y_true == i).astype(int), y_prob[:, i])
    macro = float(np.mean([curvas[i][2] for i in range(num_classes)]))
    onehot = np.eye(num_classes)[y_true].ravel()
    micro = roc_binaria(onehot.astype(int), y_prob.ravel())
    return curvas, micro, macro


# --- 4. Modelo: MobileNetV3-Small congelada + cabeza (idéntico a la v4) ---
def construir_modelo() -> nn.Module:
    modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for p in modelo.features.parameters():
        p.requires_grad = False
    modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, num_classes)
    return modelo.to(DEVICE)


def correr_epoca(modelo, optimizer, criterion, dl, entrenar: bool):
    modelo.train(entrenar)
    if entrenar:
        modelo.features.eval()  # BatchNorm congelada (gotcha del transfer learning, ver v4)
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


def matriz_confusion(modelo) -> np.ndarray:
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
    return cm


# La matriz de confusión usa la predicción DURA (argmax); ROC-AUC necesita el score
# CONTINUO (probabilidad softmax) de cada clase, así que lo recolectamos aparte.
def probabilidades_val(modelo) -> tuple[np.ndarray, np.ndarray]:
    ys, probs = [], []
    modelo.eval()
    with torch.no_grad():
        for x, y in val_dl:
            p = torch.softmax(modelo(x.to(DEVICE)), dim=1).cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def entrenar_variante(usar_pesos: bool):
    """Entrena una variante (sin/con pesos) con early stopping; devuelve (hist, cm)."""
    modelo = construir_modelo()
    criterion = nn.CrossEntropyLoss(weight=pesos if usar_pesos else None)
    optimizer = torch.optim.Adam(modelo.classifier.parameters())

    hist = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}
    mejor_val_loss = float("inf")
    mejor_state = copy.deepcopy(modelo.state_dict())
    epocas_sin_mejora = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = correr_epoca(modelo, optimizer, criterion, train_dl, True)
        va_loss, va_acc = correr_epoca(modelo, optimizer, criterion, val_dl, False)
        hist["acc"].append(tr_acc); hist["val_acc"].append(va_acc)
        hist["loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        print(f"Época {epoch:2d}/{EPOCHS}  loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_loss < mejor_val_loss - MIN_DELTA:
            mejor_val_loss = va_loss
            mejor_state = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PATIENCE:
                print(f"Early stopping en la época {epoch}.")
                break
    modelo.load_state_dict(mejor_state)
    y_true, y_prob = probabilidades_val(modelo)
    return hist, matriz_confusion(modelo), y_true, y_prob


# --- 5. Correr las DOS variantes en la misma corrida ---
variantes = [("sin_pesos", False), ("con_pesos", True)]
resultados = {}
for etiqueta, usar in variantes:
    titulo = "CON pesos de clase" if usar else "SIN corrección (línea base)"
    print("\n" + "=" * 70)
    print(f"VARIANTE: {titulo}")
    print("=" * 70)
    hist, cm, y_true, y_prob = entrenar_variante(usar)
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    print("\nReporte por clase:")
    reporte_por_clase(cm, class_names)
    curvas, micro, macro = curvas_roc_ovr(y_true, y_prob)
    print("\nROC-AUC One-vs-Rest (val):")
    for i in range(num_classes):
        print(f"  {class_names[i]:<16} AUC = {curvas[i][2]:.4f}")
    print(f"  {'macro-promedio':<16} AUC = {macro:.4f}")
    print(f"  {'micro-promedio':<16} AUC = {micro[2]:.4f}")
    resultados[etiqueta] = {"hist": hist, "cm": cm, "roc": (curvas, micro, macro)}

# --- 6. Figuras: matrices lado a lado + curvas de las dos variantes ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (etiqueta, _) in zip(axes, variantes):
    cm = resultados[etiqueta]["cm"]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {etiqueta}")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_matrices_v5.png", dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for fila, (etiqueta, _) in enumerate(variantes):
    h = resultados[etiqueta]["hist"]
    rng = range(1, len(h["loss"]) + 1)
    axes[fila, 0].plot(rng, h["acc"], label="train")
    axes[fila, 0].plot(rng, h["val_acc"], label="val")
    axes[fila, 0].set_title(f"Accuracy — {etiqueta}"); axes[fila, 0].legend()
    axes[fila, 0].set_xlabel("época"); axes[fila, 0].set_ylabel("accuracy")
    axes[fila, 1].plot(rng, h["loss"], label="train")
    axes[fila, 1].plot(rng, h["val_loss"], label="val")
    axes[fila, 1].set_title(f"Loss — {etiqueta}"); axes[fila, 1].legend()
    axes[fila, 1].set_xlabel("época"); axes[fila, 1].set_ylabel("loss")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_curvas_v5.png", dpi=150, bbox_inches="tight")

# Curvas ROC One-vs-Rest de las DOS variantes, lado a lado (mismo layout que las matrices):
# permite ver si la corrección por pesos levanta el AUC de la clase minoritaria.
colores = ["#1F3864", "#B45309", "#2E7D32", "#7B1FA2"]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (etiqueta, _) in zip(axes, variantes):
    curvas, micro, macro = resultados[etiqueta]["roc"]
    for i, (fpr, tpr, auc) in curvas.items():
        ax.plot(fpr, tpr, lw=2, color=colores[i % len(colores)],
                label=f"{class_names[i]} (AUC={auc:.3f})")
    ax.plot(micro[0], micro[1], lw=2, ls=":", color="gray",
            label=f"micro-promedio (AUC={micro[2]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="azar (AUC=0.500)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title(f"ROC One-vs-Rest — {etiqueta}\nAUC macro = {macro:.3f}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Curvas ROC v5 (val) — efecto de la corrección por pesos de clase",
             fontweight="bold", color="#1F3864")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Figure_roc_v5.png", dpi=150, bbox_inches="tight")
print("\nGuardado Figure_matrices_v5.png, Figure_curvas_v5.png y Figure_roc_v5.png")
