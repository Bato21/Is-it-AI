"""Genera los diagramas flat (tema oscuro) para la presentación .pptx.

Corre con el venv de PyTorch (tiene matplotlib):
  PyTorch/.venv/Scripts/python.exe documentacion/deck_assets/generar_assets.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

AQUI = Path(__file__).resolve().parent

BG = "#0B0E14"
FG = "#E8EAF2"
MUT = "#8A91A6"
VIO = "#7C5CFF"
CYA = "#19D3C5"
TF_C = "#FF8A3D"
PT_C = "#EE4C2C"
C0, C1, C2 = "#34D399", "#FBBF24", "#F87171"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "font.family": "DejaVu Sans",
    }
)

grad = LinearSegmentedColormap.from_list("vc", [VIO, CYA])


def _caja(ax, x, y, w, h, color, texto, sub=None, fs=13, fc=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.6, edgecolor=color, facecolor=fc or BG,
        )
    )
    cy = y + h / 2 + (0.035 if sub else 0)
    ax.text(x + w / 2, cy, texto, ha="center", va="center", fontsize=fs,
            color=FG, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.055, sub, ha="center", va="center",
                fontsize=fs - 4, color=MUT)


def _flecha(ax, x0, x1, y, color=MUT):
    ax.add_patch(
        FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=22,
                        linewidth=2, color=color)
    )


def gradiente_clases():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Barra degradada
    n = 600
    for i in range(n):
        ax.add_patch(plt.Rectangle((0.05 + 0.9 * i / n, 0.52), 0.9 / n + 0.002, 0.13,
                                   color=grad(i / n), linewidth=0))
    ax.add_patch(FancyBboxPatch((0.05, 0.52), 0.9, 0.13,
                                boxstyle="round,pad=0.001,rounding_size=0.02",
                                fill=False, edgecolor="#2A3042", linewidth=1))
    nodos = [
        (0.05, "0", C0, "0_sin_ia", "sin artefactos visibles"),
        (0.50, "1", C1, "1_rastro_ia", "densidad intermedia"),
        (0.95, "2", C2, "2_saturada_ia", "firma de generador end-to-end"),
    ]
    for x, num, c, nombre, sub in nodos:
        ax.scatter([x], [0.585], s=560, color=BG, edgecolor=c, linewidth=2.4, zorder=5)
        ax.text(x, 0.585, num, ha="center", va="center", fontsize=15,
                color=c, fontweight="bold", zorder=6)
        ax.text(x, 0.34, nombre, ha="center", va="center", fontsize=14,
                color=FG, fontweight="bold", family="monospace")
        ax.text(x, 0.20, sub, ha="center", va="center", fontsize=11, color=MUT)
    ax.text(0.5, 0.86, "Densidad de artefactos visuales de IA →",
            ha="center", va="center", fontsize=14, color=MUT, style="italic")
    fig.savefig(AQUI / "gradiente_clases.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pipeline_dataset():
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    pasos = [
        ("Fuentes por clase", "Zenodo10K · Gamma\nCopilot · Canva", VIO),
        ("Render", "LibreOffice headless\n.pptx → PNG por diapo", VIO),
        ("Etiquetado", "carpetas 0_/1_/2_\n(label = procedencia)", CYA),
        ("Split 80/20", "300 imgs · ~100/clase\nseed fija", CYA),
    ]
    w, h, gap = 0.20, 0.56, 0.045
    x = 0.02
    for i, (t, s, c) in enumerate(pasos):
        _caja(ax, x, 0.22, w, h, c, t, s, fs=13)
        if i < 3:
            _flecha(ax, x + w + 0.004, x + w + gap - 0.004, 0.5)
        x += w + gap
    fig.savefig(AQUI / "pipeline_dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def arquitectura_cnn():
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    capas = [
        ("Input", "180×180×3", MUT, 0.105),
        ("Conv 16", "3×3 + ReLU", VIO, 0.105),
        ("MaxPool", "2×2", "#5A4FCF", 0.09),
        ("Conv 32", "3×3 + ReLU", VIO, 0.105),
        ("MaxPool", "2×2", "#5A4FCF", 0.09),
        ("Flatten", "59.168", CYA, 0.095),
        ("Dense 64", "ReLU", CYA, 0.10),
        ("Dense 3", "softmax", C1, 0.095),
    ]
    x = 0.012
    for i, (t, s, c, w) in enumerate(capas):
        _caja(ax, x, 0.26, w, 0.50, c, t, s, fs=11)
        if i < len(capas) - 1:
            _flecha(ax, x + w + 0.003, x + w + 0.022, 0.51)
        x += w + 0.025
    ax.text(0.5, 0.06, "3.792.099 parámetros — 99% en la primera capa densa",
            ha="center", fontsize=12, color=MUT, style="italic")
    fig.savefig(AQUI / "arquitectura_cnn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def augmentation():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ops = [
        ("Rotación", "±18°\nfoto un poco torcida"),
        ("Zoom", "±10%\ndistancia a la diapo"),
        ("Brillo", "±0.2\nreflejo de proyector"),
        ("Contraste", "±0.2\niluminación variable"),
    ]
    w, gap = 0.185, 0.032
    x = 0.02
    for t, s in ops:
        _caja(ax, x, 0.34, w, 0.48, CYA, t, s, fs=13)
        x += w + gap
    # Tarjeta tachada: sin flip
    _caja(ax, x, 0.34, w, 0.48, "#4A4F63", "Flip", "texto en espejo\nno existe en el mundo real", fs=13)
    ax.plot([x + 0.015, x + w - 0.015], [0.37, 0.79], color=C2, linewidth=3, zorder=8)
    ax.text(0.5, 0.12, "Domain-aware: solo transformaciones realistas para fotos de diapositivas",
            ha="center", fontsize=12.5, color=MUT, style="italic")
    fig.savefig(AQUI / "augmentation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def frameworks():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _caja(ax, 0.05, 0.18, 0.40, 0.62, TF_C, "TensorFlow / Keras",
          "augmentation = capas del modelo\nse apagan solas en inferencia", fs=15)
    _caja(ax, 0.55, 0.18, 0.40, 0.62, PT_C, "PyTorch",
          "augmentation = transform del loader\nsolo en el split de train", fs=15)
    ax.text(0.5, 0.49, "vs", ha="center", va="center", fontsize=16, color=MUT,
            fontweight="bold")
    ax.text(0.5, 0.045, "Mismos datos · mismo modelo · mismas 10 épocas · scripts espejo v1/v2",
            ha="center", fontsize=12.5, color=MUT, style="italic")
    fig.savefig(AQUI / "frameworks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def roadmap():
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    hitos = [
        ("1 · Cerrar v2", "EarlyStopping\nentrenar a convergencia", CYA),
        ("2 · v3 según diagnóstico", "domain gap o\nprotocolo 1↔2", CYA),
        ("3 · MobileNetV3", "transfer learning\n(reemplaza CNN)", VIO),
        ("4 · Móvil", "TFLite / PyTorch Lite\napp Android", VIO),
    ]
    w, h, gap = 0.205, 0.56, 0.04
    x = 0.015
    for i, (t, s, c) in enumerate(hitos):
        _caja(ax, x, 0.24, w, h, c, t, s, fs=12.5)
        if i < 3:
            _flecha(ax, x + w + 0.004, x + w + gap - 0.004, 0.52)
        x += w + gap
    fig.savefig(AQUI / "roadmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def overfit_loss():
    """Mini-gráfico conceptual v1: la brecha está en la loss, no en la accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, titulo in ((ax1, "Accuracy (TF v1)"), (ax2, "Loss (TF v1)")):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUT, labelsize=9)
        for s in ax.spines.values():
            s.set_color("#2A3042")
        ax.set_title(titulo, color=FG, fontsize=13, fontweight="bold")
        ax.set_xlabel("época", color=MUT, fontsize=10)
    ep = range(1, 11)
    acc_t = [0.44, 0.62, 0.75, 0.85, 0.92, 0.96, 0.98, 0.99, 1.00, 1.00]
    acc_v = [0.43, 0.55, 0.70, 0.80, 0.88, 0.92, 0.94, 0.95, 0.96, 0.967]
    los_t = [1.10, 0.72, 0.48, 0.30, 0.18, 0.10, 0.05, 0.03, 0.015, 0.0099]
    los_v = [1.05, 0.80, 0.60, 0.45, 0.32, 0.24, 0.18, 0.14, 0.115, 0.1026]
    ax1.plot(ep, acc_t, color=VIO, linewidth=2.4, label="train")
    ax1.plot(ep, acc_v, color=CYA, linewidth=2.4, label="val")
    ax1.annotate("casi sin brecha", xy=(9, 0.98), xytext=(5.2, 0.62), color=FG,
                 fontsize=10.5, arrowprops=dict(arrowstyle="->", color=MUT))
    ax2.plot(ep, los_t, color=VIO, linewidth=2.4, label="train")
    ax2.plot(ep, los_v, color=CYA, linewidth=2.4, label="val")
    ax2.annotate("brecha ~10×\n(0.010 vs 0.103)", xy=(9.6, 0.06), xytext=(5.6, 0.55),
                 color=C2, fontsize=10.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C2))
    for ax in (ax1, ax2):
        leg = ax.legend(facecolor=BG, edgecolor="#2A3042", labelcolor=FG, fontsize=10)
    fig.suptitle("Curvas reconstruidas desde los logs de la corrida 2 (300 imágenes)",
                 color=MUT, fontsize=10, style="italic", y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(AQUI / "overfit_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    gradiente_clases()
    pipeline_dataset()
    arquitectura_cnn()
    augmentation()
    frameworks()
    roadmap()
    overfit_loss()
    print("Assets generados en", AQUI)
