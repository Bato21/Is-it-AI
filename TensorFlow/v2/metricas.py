"""Métricas manuales y figuras de evaluación del lado TensorFlow.

Cálculo en NumPy puro (sin scikit-learn, para no chocar con las versiones del
stack): matriz de confusión, precision/recall/F1, ROC one-vs-rest y sus figuras.
Reciben clases/colores/título como argumentos. Las ``plot_*`` registran la ruta
con ``logging``; ``imprimir_tabla`` usa ``print`` (salida para el usuario).
"""

from __future__ import annotations

import logging
import math
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: guardamos PNG, no mostramos
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Paleta institucional reutilizada en títulos y resaltados de las figuras.
AZUL_TITULO = "#1F3864"
VERDE_DIAGONAL = "#1E7B45"


class MetricaClase(TypedDict):
    """Métricas de una sola clase dentro del reporte."""

    precision: float
    recall: float
    f1: float
    support: int


class Historial(TypedDict):
    """Historial de entrenamiento en formato común a ambos frameworks.

    ``corte`` es el número de épocas de la fase 1 (cabeza); sirve para dibujar
    la línea divisoria entre fase 1 y fase 2 en las curvas.
    """

    acc: list[float]
    val_acc: list[float]
    loss: list[float]
    val_loss: list[float]
    corte: int


# ── Cálculo ───────────────────────────────────────────────────────────
def matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_clases: int) -> np.ndarray:
    """Matriz de confusión NxN: filas = etiqueta real, columnas = predicción."""
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t)][int(p)] += 1
    return cm


def metricas_por_clase(cm: np.ndarray) -> dict[int, MetricaClase]:
    """Precision, recall, F1 y soporte por clase a partir de la matriz."""
    metricas: dict[int, MetricaClase] = {}
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metricas[i] = MetricaClase(
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            support=int(cm[i, :].sum()),
        )
    return metricas


def curva_roc(y_true_bin: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC one-vs-rest por barrido de umbrales. Devuelve (fpr, tpr, auc).

    Si la clase no tiene positivos o negativos en este split, el AUC no está
    definido; devolvemos la diagonal y AUC 0.5 (azar) en vez de fallar.
    """
    umbrales = np.sort(np.unique(y_score))[::-1]
    positivos = int(y_true_bin.sum())
    negativos = len(y_true_bin) - positivos
    if positivos == 0 or negativos == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    fprs, tprs = [0.0], [0.0]
    for umbral in umbrales:
        pred = (y_score >= umbral).astype(int)
        tp = int(((pred == 1) & (y_true_bin == 1)).sum())
        fp = int(((pred == 1) & (y_true_bin == 0)).sum())
        fprs.append(fp / negativos)
        tprs.append(tp / positivos)
    fprs.append(1.0)
    tprs.append(1.0)
    fpr, tpr = np.array(fprs), np.array(tprs)
    return fpr, tpr, abs(float(np.trapz(tpr, fpr)))


def accuracy(cm: np.ndarray) -> float:
    """Accuracy global = traza / total. Devuelve 0.0 si la matriz está vacía."""
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


# ── Salida a consola (orientada al usuario) ───────────────────────────
def imprimir_tabla(met: dict[int, MetricaClase], clases: list[str], acc: float) -> None:
    """Imprime la tabla precision/recall/F1 por clase y la accuracy global."""
    ancho = 58
    print(
        f"\n{'─' * ancho}\n"
        f"{'Clase':<14}{'Precision':>10}{'Recall':>8}{'F1':>8}{'N':>6}\n"
        f"{'─' * ancho}"
    )
    for i, clase in enumerate(clases):
        m = met[i]
        print(
            f"{clase:<14}{m['precision']:>10.3f}"
            f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>6}"
        )
    print(f"{'─' * ancho}\nAccuracy: {acc:.4f}  ({acc:.1%})")


# ── Figuras ───────────────────────────────────────────────────────────
def plot_curvas(
    hist: Historial,
    titulo: str,
    etiqueta_loss: str,
    out: str = "training_curves.png",
) -> None:
    """Dibuja accuracy y loss por época, con la divisoria fase 1 / fase 2."""
    ep = range(1, len(hist["acc"]) + 1)
    corte = hist["corte"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, hist["acc"], "b-o", ms=4, label="Train Acc")
    ax1.plot(ep, hist["val_acc"], "b--s", ms=4, label="Val Acc")
    ax1.axvline(corte + 0.5, color="gray", ls=":")
    ax1.set_title("Accuracy por época", fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(ep, hist["loss"], "r-o", ms=4, label="Train Loss")
    ax2.plot(ep, hist["val_loss"], "r--s", ms=4, label="Val Loss")
    ax2.axvline(corte + 0.5, color="gray", ls=":")
    ax2.set_title("Loss por época", fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel(etiqueta_loss)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"{titulo}\nFase 1: cabeza · Fase 2: fine-tuning",
        fontweight="bold",
        color=AZUL_TITULO,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figura guardada: %s", out)


def plot_matriz_confusion(
    cm: np.ndarray,
    met: dict[int, MetricaClase],
    acc: float,
    clases: list[str],
    titulo: str,
    out: str = "confusion_matrix.png",
) -> None:
    """Dibuja la matriz de confusión junto a la tabla de métricas por clase."""
    n = len(clases)
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], figure=fig)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(clases, rotation=30, ha="right")
    ax.set_yticklabels(clases)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de Confusión — Test\nAccuracy: {acc:.1%}", pad=12)
    umbral = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if cm[i, j] > umbral else "black",
            )
        ax.add_patch(
            plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=VERDE_DIAGONAL, lw=2.5)
        )

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    filas = [
        [
            clases[i],
            f"{met[i]['precision']:.3f}",
            f"{met[i]['recall']:.3f}",
            f"{met[i]['f1']:.3f}",
            str(met[i]["support"]),
        ]
        for i in range(n)
    ]
    macro = [np.mean([met[i][k] for i in range(n)]) for k in ("precision", "recall", "f1")]
    filas.append(["macro avg", f"{macro[0]:.3f}", f"{macro[1]:.3f}", f"{macro[2]:.3f}", ""])
    tbl = ax2.table(
        cellText=filas,
        colLabels=["Clase", "Precision", "Recall", "F1", "N"],
        loc="center",
        bbox=[0.0, 0.3, 1.0, 0.6],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(AZUL_TITULO)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == len(filas):
            cell.set_facecolor("#D6E4F0")
            cell.set_text_props(fontweight="bold")

    plt.suptitle(titulo, fontweight="bold", color=AZUL_TITULO)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figura guardada: %s", out)


def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clases: list[str],
    colores: list[str],
    titulo: str,
    out: str = "roc_curves.png",
) -> dict[str, float]:
    """Dibuja una ROC por clase (one-vs-rest) más un panel resumen.

    Devuelve el AUC por clase para que el llamador lo registre o reporte.
    """
    n = len(clases)
    y_bin = np.eye(n)[y_true]
    paneles = n + 1
    cols = 2
    rows = math.ceil(paneles / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5 * rows))
    axes_flat = list(np.array(axes).flat)

    aucs: dict[str, float] = {}
    for i, (clase, color) in enumerate(zip(clases, colores, strict=True)):
        ax = axes_flat[i]
        fpr, tpr, auc = curva_roc(y_bin[:, i], y_prob[:, i])
        aucs[clase] = auc
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {clase}", fontweight="bold", color=color)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    ax_all = axes_flat[n]
    for i, (clase, color) in enumerate(zip(clases, colores, strict=True)):
        fpr, tpr, auc = curva_roc(y_bin[:, i], y_prob[:, i])
        ax_all.plot(fpr, tpr, color=color, lw=2, label=f"{clase} (AUC={auc:.3f})")
    ax_all.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_all.set_title(
        f"Todas las clases\nMacro-avg AUC = {np.mean(list(aucs.values())):.3f}",
        fontweight="bold",
    )
    ax_all.set_xlabel("FPR")
    ax_all.set_ylabel("TPR")
    ax_all.legend(loc="lower right", fontsize=9)
    ax_all.grid(alpha=0.3)

    for k in range(paneles, len(axes_flat)):
        axes_flat[k].axis("off")

    plt.suptitle(titulo, fontweight="bold", color=AZUL_TITULO)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figura guardada: %s", out)
    return aucs
