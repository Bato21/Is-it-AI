"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 4  (transfer learning: MobileNetV3-Small, feature extraction)

Propósito experimental:
  Hasta la v3 el modelo era una CNN mínima entrenada desde cero (~3.79M parámetros,
  todos aprendidos con nuestras 300 imágenes). La v4 hace el cambio conceptual que
  el roadmap del proyecto tenía reservado: reemplazar esa CNN por MobileNetV3-Small
  preentrenada en ImageNet, congelar el backbone y entrenar SOLO una cabeza nueva.
  Esto es "feature extraction": aprovechamos filtros visuales ya aprendidos sobre
  millones de imágenes y solo ajustamos el clasificador final a nuestras 3 clases.

  El early stopping de la v3 se mantiene: ya es parte de la receta base, no un
  cambio de esta versión.

Nuevo respecto a la v3:
  - Backbone MobileNetV3-Small preentrenado (ImageNet), congelado.
  - IMG_SIZE 224x224 (requisito de entrada de MobileNetV3).
  - Normalización ImageNet OBLIGATORIA en train y eval.
  - Gotcha de BatchNorm en modo eval con backbone congelado (ver bloque 5).

Nota: la primera corrida DESCARGA los pesos (~10 MB) a ~/.cache/torch; requiere
internet una única vez. Las corridas siguientes usan la caché local.
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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")  # carpeta real del dataset
# CAMBIO: 224x224 en vez de 180x180. MobileNetV3 fue entrenada con este tamaño;
# usar otro degrada las features preentrenadas. Es un requisito de la arquitectura,
# no una elección libre como en las v1-v3.
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 60  # techo alto; el early stopping decide (heredado de la v3).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping (idéntico a la v3).
PATIENCE = 8
MIN_DELTA = 0.0

# Normalización ImageNet: media y desvío por canal (RGB) del set con el que se
# preentrenó MobileNetV3. Los guardamos aparte porque la INFERENCIA (modelopt_v4.py)
# tiene que replicarlos EXACTAMENTE; van también dentro del checkpoint.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# --- Transforms ---
# NORMALIZACIÓN IMAGENET OBLIGATORIA (train Y eval), siempre DESPUÉS de ToTensor().
# Los pesos preentrenados esperan entradas normalizadas con esta media/desvío; si se
# omite, el modelo NO tira error, simplemente se degrada EN SILENCIO (las activaciones
# quedan fuera del rango que el backbone aprendió a manejar).
# CONTRASTE CON TENSORFLOW: en el lado TF la normalización vive DENTRO del modelo
# (capa Rescaling/Normalization), así que viaja con el .keras y la inferencia la
# aplica sola. Acá vive en el TRANSFORM, fuera del modelo -> la inferencia tiene que
# reproducir el mismo Normalize a mano, o se des-sincroniza.
# Augmentation domain-aware: la MISMA de v2/v3 (RandomAffine 18°, scale 0.9-1.1;
# ColorJitter brillo/contraste 0.2), adaptada al nuevo tamaño. Sin flip, como siempre.
transform_train = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.RandomAffine(degrees=18, scale=(0.9, 1.1)),  # rotación + zoom
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # brillo + contraste
        transforms.ToTensor(),  # 0-255 -> 0-1
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),  # ImageNet
    ]
)
transform_eval = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),  # ImageNet
    ]
)

# --- 2. Carga de datos (split 80/20 IDÉNTICO a v2/v3, mismo seed) ---
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

ds_train = Subset(datasets.ImageFolder(DATA_DIR, transform=transform_train), list(idx_train))
ds_val = Subset(datasets.ImageFolder(DATA_DIR, transform=transform_eval), list(idx_val))
train_dl = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(ds_val, batch_size=BATCH_SIZE)


# --- 3. Modelo: MobileNetV3-Small preentrenado, backbone congelado ---
# weights=DEFAULT trae los pesos recomendados de ImageNet (descarga la 1a vez).
modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

# Congelar el backbone (extractor de features): no queremos re-aprender los filtros,
# solo usarlos. Con 300 imágenes, re-entrenar 3.79M+ parámetros sería sobreajuste
# garantizado; congelando, solo entrenamos la cabeza.
for p in modelo.features.parameters():
    p.requires_grad = False

# Reemplazar la última capa por una acorde a nuestras 3 clases. En MobileNetV3-Small
# el clasificador es Sequential y la capa final es classifier[3] (un Linear).
# La cabeza original de MobileNetV3 ya trae Dropout(0.2), así que NO agregamos
# regularización extra: la receta del proyecto es responder al diagnóstico, no
# amontonar técnicas (y acá todavía no vimos sobreajuste con transfer learning).
modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, num_classes)
modelo = modelo.to(DEVICE)

# Conteo de parámetros: contrastar TOTALES vs ENTRENABLES es parte del mensaje de la
# presentación. En v1-v3 los 3.79M eran TODOS entrenables; acá casi todo está congelado
# y solo entrenamos la cabeza.
total_params = sum(p.numel() for p in modelo.parameters())
train_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"Parámetros totales:     {total_params:,}")
print(f"Parámetros entrenables: {train_params:,}  (solo la cabeza)")

# --- 4. Compilación (optimizador + pérdida) ---
criterion = nn.CrossEntropyLoss()  # incluye softmax internamente
# Adam SOLO sobre la cabeza: no tiene sentido pasarle los parámetros congelados.
optimizer = torch.optim.Adam(modelo.classifier.parameters())


def correr_epoca(dl, entrenar: bool) -> tuple[float, float]:
    modelo.train(entrenar)
    # GOTCHA DE BATCHNORM (contraste de frameworks de oro para el curso):
    # requires_grad=False congela los PESOS del backbone, pero NO congela las
    # estadísticas de las capas BatchNorm. En modo train(), BatchNorm sigue
    # actualizando su running_mean/running_var con NUESTROS datos, corrompiendo en
    # silencio las estadísticas de ImageNet que vienen con los pesos preentrenados.
    # Solución: después de modelo.train(), forzar el backbone a modo eval() para que
    # BatchNorm use sus estadísticas congeladas.
    # En KERAS esto NO pasa: trainable=False pone automáticamente las BN en modo
    # inferencia. Es una diferencia real y famosa entre los dos frameworks.
    if entrenar:
        modelo.features.eval()
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


# --- 5. Entrenamiento con EARLY STOPPING manual (heredado de la v3) ---
# En Keras esto sería EarlyStopping(monitor="val_loss", patience=8, min_delta=0,
# restore_best_weights=True). En PyTorch lo implementamos a mano: guardamos una
# copia profunda del state_dict cuando val_loss mejora, contamos épocas sin mejora,
# cortamos al llegar a patience y restauramos los mejores pesos antes de evaluar.
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

# Restaurar los mejores pesos antes de evaluar y guardar.
modelo.load_state_dict(mejor_state)
epocas_corridas = len(loss_hist)
print(
    f"\nMejor modelo: época {mejor_epoca}/{epocas_corridas}  "
    f"val_loss={mejor_val_loss:.4f}  val_acc={val_acc[mejor_epoca - 1]:.4f}  "
    f"(train loss={loss_hist[mejor_epoca - 1]:.4f} acc={acc[mejor_epoca - 1]:.4f})"
)

# --- 6. Curvas train vs val: accuracy Y loss (épocas realmente corridas) ---
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

# --- 7. Matriz de confusión sobre validación (con los mejores pesos restaurados) ---
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

# --- 8. Guardar el modelo entrenado ---
# Además del state_dict, guardamos arch y la normalización: la inferencia necesita
# reconstruir la MISMA arquitectura y aplicar el MISMO Normalize (si no, se
# des-sincroniza con los pesos, ver contraste con TF en el bloque de transforms).
MODEL_PATH = Path(__file__).resolve().parent / "modelopt_v4_mobilenetv3.pt"
torch.save(
    {
        "state_dict": modelo.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "arch": "mobilenet_v3_small",
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
    },
    MODEL_PATH,
)
print(f"Modelo guardado en {MODEL_PATH.name}")
