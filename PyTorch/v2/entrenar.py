"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 2 — Transfer Learning con MobileNetV3Large

Equivalente en PyTorch a TensorFlow/v2/entrenar.py. Mismo flujo de trabajo:
  - Transfer learning (MobileNetV3Large preentrenada en ImageNet, torchvision)
  - Data augmentation en el pipeline de entrenamiento
  - Dropout en la cabeza
  - Dos fases: cabeza congelada -> fine-tuning de las últimas capas
  - Métricas manuales (sin sklearn) y mismas figuras que el lado TF

Clases:
  0_sin_ia · 1_rastro_ia · 2_saturada_ia

Genera, en esta carpeta:
  modelo_diapositivas.pt  — checkpoint {model_state, config} (lo usa consumidor.py)
  training_curves.png · confusion_matrix.png · roc_curves.png
"""
import math
import sys
from pathlib import Path

# Consola UTF-8 (Windows usa cp1252 por defecto y rompe con ─, acentos, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. CONFIGURACIÓN ──────────────────────────────────────────────────
RAIZ          = Path(__file__).resolve().parents[2]
DATA_DIR      = str(RAIZ / "dataset")
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS_FASE1  = 15
EPOCHS_FASE2  = 10
LR_FASE1      = 1e-3
LR_FASE2      = 1e-4
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASES   = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
N_CLASES = len(CLASES)
COLORES  = ["#1E7B45", "#B45309", "#991B1B"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

AQUI = Path(__file__).resolve().parent
MODELO_OUT = AQUI / "modelo_diapositivas.pt"


# ── 2. DATASET (70/15/15) ─────────────────────────────────────────────
def _tf_train():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def _tf_eval():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def cargar_dataset():
    base = datasets.ImageFolder(DATA_DIR)
    if len(base) == 0:
        raise SystemExit(f"No hay imágenes en {DATA_DIR}. Convierte y clasifica diapositivas primero.")
    assert base.classes == CLASES, f"Orden de clases {base.classes} != {CLASES}"

    n_total = len(base)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val
    g = torch.Generator().manual_seed(SEED)
    tr, va, te = random_split(base, [n_train, n_val, n_test], generator=g)

    # Transform por split (train con augmentation, val/test sin)
    full_train = datasets.ImageFolder(DATA_DIR, transform=_tf_train())
    full_eval  = datasets.ImageFolder(DATA_DIR, transform=_tf_eval())
    tr.dataset = full_train
    va.dataset = full_eval
    te.dataset = full_eval

    print(f"Total: {n_total}  |  Train: {n_train}  Val: {n_val}  Test: {n_test}")
    return (
        DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(va, batch_size=BATCH_SIZE),
        DataLoader(te, batch_size=BATCH_SIZE),
    )


# ── 3. MODELO ─────────────────────────────────────────────────────────
def construir_modelo() -> nn.Module:
    net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    in_f = net.classifier[0].in_features  # 960
    net.classifier = nn.Sequential(
        nn.Linear(in_f, 256), nn.Hardswish(),
        nn.Dropout(0.3), nn.Linear(256, N_CLASES),
    )
    return net


def set_features_trainable(net: nn.Module, trainable: bool, ultimos_bloques: int | None = None):
    """Congela/descongela el extractor. Si ultimos_bloques se da, solo descongela
    los últimos N bloques (fine-tuning parcial, como FINE_TUNE_AT en Keras)."""
    bloques = list(net.features.children())
    for b in bloques:
        for p in b.parameters():
            p.requires_grad = False
    if trainable:
        objetivo = bloques if ultimos_bloques is None else bloques[-ultimos_bloques:]
        for b in objetivo:
            for p in b.parameters():
                p.requires_grad = True


# ── 4. ENTRENAR ───────────────────────────────────────────────────────
def correr_epoca(net, dl, criterion, optimizer=None):
    entrenar = optimizer is not None
    net.train(entrenar)
    total, correctos, suma = 0, 0, 0.0
    torch.set_grad_enabled(entrenar)
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if entrenar:
            optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        if entrenar:
            loss.backward(); optimizer.step()
        suma += loss.item() * x.size(0)
        correctos += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return suma / total, correctos / total


def entrenar(net, train_dl, val_dl):
    criterion = nn.CrossEntropyLoss()
    hist = {"acc": [], "val_acc": [], "loss": [], "val_loss": [], "corte": 0}

    print(f"\n{'─'*55}\nFASE 1 — Cabeza ({EPOCHS_FASE1} épocas), base congelada\n{'─'*55}")
    set_features_trainable(net, False)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=LR_FASE1)
    for e in range(1, EPOCHS_FASE1 + 1):
        tl, ta = correr_epoca(net, train_dl, criterion, opt)
        vl, va = correr_epoca(net, val_dl, criterion)
        hist["acc"].append(ta); hist["val_acc"].append(va)
        hist["loss"].append(tl); hist["val_loss"].append(vl)
        print(f"  [cabeza] {e:2d}/{EPOCHS_FASE1}  acc={ta:.3f} val_acc={va:.3f}  loss={tl:.3f} val_loss={vl:.3f}")
    hist["corte"] = len(hist["acc"])

    print(f"\n{'─'*55}\nFASE 2 — Fine-tuning (últimos bloques)\n{'─'*55}")
    set_features_trainable(net, True, ultimos_bloques=4)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=LR_FASE2)
    for e in range(1, EPOCHS_FASE2 + 1):
        tl, ta = correr_epoca(net, train_dl, criterion, opt)
        vl, va = correr_epoca(net, val_dl, criterion)
        hist["acc"].append(ta); hist["val_acc"].append(va)
        hist["loss"].append(tl); hist["val_loss"].append(vl)
        print(f"  [fine]   {e:2d}/{EPOCHS_FASE2}  acc={ta:.3f} val_acc={va:.3f}  loss={tl:.3f} val_loss={vl:.3f}")
    return hist


# ── 5. MÉTRICAS MANUALES ──────────────────────────────────────────────
def calcular_cm(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def metricas_por_clase(cm):
    m = {}
    for i in range(len(cm)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        m[i] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}
    return m


def roc_manual(y_true_bin, y_score):
    thr = np.sort(np.unique(y_score))[::-1]
    P = int(y_true_bin.sum()); N = len(y_true_bin) - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    fprs, tprs = [0.0], [0.0]
    for t in thr:
        pred = (y_score >= t).astype(int)
        fprs.append(int(((pred == 1) & (y_true_bin == 0)).sum()) / N)
        tprs.append(int(((pred == 1) & (y_true_bin == 1)).sum()) / P)
    fprs.append(1.0); tprs.append(1.0)
    fpr, tpr = np.array(fprs), np.array(tprs)
    return fpr, tpr, abs(float(np.trapz(tpr, fpr)))


def predecir_test(net, test_dl):
    net.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in test_dl:
            probs = F.softmax(net(x.to(DEVICE)), dim=1).cpu().numpy()
            yp.append(probs); yt.append(y.numpy())
    y_true = np.concatenate(yt); y_prob = np.concatenate(yp)
    return y_true, np.argmax(y_prob, axis=1), y_prob


# ── 6. PLOTS (mismos que el lado TensorFlow) ──────────────────────────
def plot_curvas(h, out="training_curves.png"):
    ep = range(1, len(h["acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, h["acc"], "b-o", ms=4, label="Train Acc"); ax1.plot(ep, h["val_acc"], "b--s", ms=4, label="Val Acc")
    ax1.axvline(h["corte"] + 0.5, color="gray", ls=":"); ax1.set_title("Accuracy por época", fontweight="bold")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.05)
    ax2.plot(ep, h["loss"], "r-o", ms=4, label="Train Loss"); ax2.plot(ep, h["val_loss"], "r--s", ms=4, label="Val Loss")
    ax2.axvline(h["corte"] + 0.5, color="gray", ls=":"); ax2.set_title("Loss por época", fontweight="bold")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Cross-Entropy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle("MobileNetV3Large (PyTorch) — Curvas de entrenamiento\nFase 1: cabeza · Fase 2: fine-tuning",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")


def plot_cm(cm, met, acc, out="confusion_matrix.png"):
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], figure=fig)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N_CLASES)); ax.set_yticks(range(N_CLASES))
    ax.set_xticklabels(CLASES, rotation=30, ha="right"); ax.set_yticklabels(CLASES)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de Confusión — Test\nAccuracy: {acc:.1%}", pad=12)
    thr = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(N_CLASES):
        for j in range(N_CLASES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold",
                    color="white" if cm[i, j] > thr else "black")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1E7B45", lw=2.5))
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    filas = [[CLASES[i], f"{met[i]['precision']:.3f}", f"{met[i]['recall']:.3f}",
              f"{met[i]['f1']:.3f}", str(met[i]['support'])] for i in range(N_CLASES)]
    macro = [np.mean([met[i][k] for i in range(N_CLASES)]) for k in ("precision", "recall", "f1")]
    filas.append(["macro avg", f"{macro[0]:.3f}", f"{macro[1]:.3f}", f"{macro[2]:.3f}", ""])
    tbl = ax2.table(cellText=filas, colLabels=["Clase", "Precision", "Recall", "F1", "N"],
                    loc="center", bbox=[0.0, 0.3, 1.0, 0.6])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F3864"); cell.set_text_props(color="white", fontweight="bold")
        elif r == len(filas):
            cell.set_facecolor("#D6E4F0"); cell.set_text_props(fontweight="bold")
    plt.suptitle("MobileNetV3Large (PyTorch) — Evaluación en test", fontweight="bold", color="#1F3864")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")


def plot_roc(y_true, y_prob, out="roc_curves.png"):
    y_bin = np.eye(N_CLASES)[y_true]
    paneles = N_CLASES + 1; cols = 2; rows = math.ceil(paneles / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5 * rows))
    axes_flat = list(np.array(axes).flat); aucs = {}
    for i, (clase, color) in enumerate(zip(CLASES, COLORES)):
        ax = axes_flat[i]
        fpr, tpr, auc = roc_manual(y_bin[:, i], y_prob[:, i]); aucs[clase] = auc
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {clase}", fontweight="bold", color=color); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax_all = axes_flat[N_CLASES]
    for clase, color in zip(CLASES, COLORES):
        idx = CLASES.index(clase)
        fpr, tpr, auc = roc_manual(y_bin[:, idx], y_prob[:, idx])
        ax_all.plot(fpr, tpr, color=color, lw=2, label=f"{clase} (AUC={auc:.3f})")
    ax_all.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_all.set_title(f"Todas las clases\nMacro-avg AUC = {np.mean(list(aucs.values())):.3f}", fontweight="bold")
    ax_all.set_xlabel("FPR"); ax_all.set_ylabel("TPR"); ax_all.legend(loc="lower right", fontsize=9); ax_all.grid(alpha=0.3)
    for k in range(paneles, len(axes_flat)):
        axes_flat[k].axis("off")
    plt.suptitle("MobileNetV3Large (PyTorch) — Curvas ROC (One-vs-Rest)", fontweight="bold", color="#1F3864")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")
    return aucs


# ── 7. MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    train_dl, val_dl, test_dl = cargar_dataset()
    net = construir_modelo().to(DEVICE)
    print(f"Parámetros totales: {sum(p.numel() for p in net.parameters()):,}")

    hist = entrenar(net, train_dl, val_dl)

    print("\nEvaluando en test...")
    y_true, y_pred, y_prob = predecir_test(net, test_dl)
    cm  = calcular_cm(y_true, y_pred, N_CLASES)
    met = metricas_por_clase(cm)
    acc = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0.0

    print(f"\n{'─'*58}\n{'Clase':<14}{'Precision':>10}{'Recall':>8}{'F1':>8}{'N':>6}\n{'─'*58}")
    for i, clase in enumerate(CLASES):
        m = met[i]
        print(f"{clase:<14}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>6}")
    print(f"{'─'*58}\nAccuracy: {acc:.4f}  ({acc:.1%})")

    print("\nGenerando visualizaciones...")
    plot_curvas(hist); plot_cm(cm, met, acc); plot_roc(y_true, y_prob)

    # Checkpoint con todo lo necesario para reconstruir el modelo sin este script.
    CONFIG = {"clases": CLASES, "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
              "arquitectura": "mobilenet_v3_large + cabeza Linear(960->256)->Dropout->Linear(256->3)"}
    torch.save({"model_state": net.state_dict(), "config": CONFIG}, MODELO_OUT)
    print(f"\nModelo guardado: {MODELO_OUT}")
    print("Completado")
