"""Genera los diagramas flat (tema CLARO) para la presentación .pptx v2.

Corre con el venv de PyTorch (tiene matplotlib):
  PyTorch/.venv/Scripts/python.exe documentacion/deck_assets/generar_assets_light.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AQUI = Path(__file__).resolve().parent

BG = "#F7F8FC"
FG = "#1B2130"
MUT = "#5C6577"
VIO = "#6D4AFF"
TEA = "#0EA8A0"
ROJO = "#D93025"
BORDE = "#E2E8F0"
C0, C1, C2 = "#16A34A", "#D97706", "#DC2626"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "font.family": "DejaVu Sans",
    }
)

grad = LinearSegmentedColormap.from_list("vt", [VIO, TEA])


def _caja(ax, x, y, w, h, color, texto, sub=None, fs=13):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.8, edgecolor=color, facecolor="#FFFFFF",
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
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = 600
    for i in range(n):
        ax.add_patch(plt.Rectangle((0.05 + 0.9 * i / n, 0.54), 0.9 / n + 0.002, 0.14,
                                   color=grad(i / n), linewidth=0))
    nodos = [
        (0.05, "0", C0, "0_sin_ia", "sin artefactos visibles"),
        (0.50, "1", C1, "1_rastro_ia", "densidad intermedia"),
        (0.95, "2", C2, "2_saturada_ia", "firma de generador"),
    ]
    for x, num, c, nombre, sub in nodos:
        ax.scatter([x], [0.61], s=560, color="#FFFFFF", edgecolor=c, linewidth=2.6, zorder=5)
        ax.text(x, 0.61, num, ha="center", va="center", fontsize=15,
                color=c, fontweight="bold", zorder=6)
        ax.text(x, 0.34, nombre, ha="center", va="center", fontsize=14,
                color=FG, fontweight="bold", family="monospace")
        ax.text(x, 0.19, sub, ha="center", va="center", fontsize=11, color=MUT)
    ax.text(0.5, 0.90, "Densidad de artefactos visuales de IA →",
            ha="center", va="center", fontsize=13.5, color=MUT, style="italic")
    fig.savefig(AQUI / "light_gradiente_clases.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def augmentation():
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ops = [
        ("Rotación ±18°", "foto torcida"),
        ("Zoom ±10%", "distancia variable"),
        ("Brillo ±0.2", "reflejo proyector"),
        ("Contraste ±0.2", "iluminación"),
    ]
    w, gap = 0.185, 0.032
    x = 0.02
    for t, s in ops:
        _caja(ax, x, 0.34, w, 0.46, TEA, t, s, fs=12.5)
        x += w + gap
    _caja(ax, x, 0.34, w, 0.46, BORDE, "Flip", "texto en espejo:\nno existe", fs=12.5)
    ax.plot([x + 0.015, x + w - 0.015], [0.37, 0.77], color=ROJO, linewidth=3, zorder=8)
    ax.text(0.5, 0.10, "Domain-aware: solo transformaciones realistas para fotos de diapositivas — sin flip",
            ha="center", fontsize=12.5, color=MUT, style="italic")
    fig.savefig(AQUI / "light_augmentation.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def overfit_loss():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, titulo in ((ax1, "Accuracy"), (ax2, "Loss")):
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors=MUT, labelsize=9)
        for s in ax.spines.values():
            s.set_color(BORDE)
        ax.set_title(titulo, color=FG, fontsize=13, fontweight="bold")
        ax.set_xlabel("época", color=MUT, fontsize=10)
        ax.grid(color=BORDE, linewidth=0.6, alpha=0.6)
    ep = range(1, 11)
    acc_t = [0.44, 0.62, 0.75, 0.85, 0.92, 0.96, 0.98, 0.99, 1.00, 1.00]
    acc_v = [0.43, 0.55, 0.70, 0.80, 0.88, 0.92, 0.94, 0.95, 0.96, 0.967]
    los_t = [1.10, 0.72, 0.48, 0.30, 0.18, 0.10, 0.05, 0.03, 0.015, 0.0099]
    los_v = [1.05, 0.80, 0.60, 0.45, 0.32, 0.24, 0.18, 0.14, 0.115, 0.1026]
    ax1.plot(ep, acc_t, color=VIO, linewidth=2.6, label="train")
    ax1.plot(ep, acc_v, color=TEA, linewidth=2.6, label="val")
    ax1.annotate("casi sin brecha", xy=(9, 0.98), xytext=(5.0, 0.62), color=FG,
                 fontsize=10.5, arrowprops=dict(arrowstyle="->", color=MUT))
    ax2.plot(ep, los_t, color=VIO, linewidth=2.6, label="train")
    ax2.plot(ep, los_v, color=TEA, linewidth=2.6, label="val")
    ax2.annotate("brecha ~10×\n(0.010 vs 0.103)", xy=(9.6, 0.06), xytext=(5.4, 0.55),
                 color=ROJO, fontsize=10.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ROJO))
    for ax in (ax1, ax2):
        ax.legend(facecolor="#FFFFFF", edgecolor=BORDE, labelcolor=FG, fontsize=10)
    fig.suptitle("TensorFlow v1 — curvas reconstruidas desde los logs (300 imágenes)",
                 color=MUT, fontsize=10, style="italic", y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(AQUI / "light_overfit_loss.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def roadmap():
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    hitos = [
        ("1 · Converger v2", "EarlyStopping\n+ más épocas", TEA),
        ("2 · Re-leer la matriz", "¿desaparece el\nerror 0↔2?", TEA),
        ("3 · MobileNetV3", "transfer\nlearning", VIO),
        ("4 · Móvil", "TFLite / Lite\napp Android", VIO),
    ]
    w, h, gap = 0.205, 0.62, 0.04
    x = 0.015
    for i, (t, s, c) in enumerate(hitos):
        _caja(ax, x, 0.20, w, h, c, t, s, fs=12.5)
        if i < 3:
            _flecha(ax, x + w + 0.004, x + w + gap - 0.004, 0.51)
        x += w + gap
    fig.savefig(AQUI / "light_roadmap.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def pipeline_dataset():
    fig, ax = plt.subplots(figsize=(12, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    pasos = [
        ("Fuentes por clase", "Zenodo pre-2022 (clase 0)\nGamma·Copilot·Canva (clase 2)", VIO),
        ("Render", "LibreOffice headless\n.pptx → PNG por diapo", VIO),
        ("Etiquetado", "carpetas 0_/1_/2_\nlabel = procedencia", TEA),
        ("Split 80/20", "300 imgs · ~100/clase\nseed fija", TEA),
    ]
    w, h, gap = 0.205, 0.60, 0.04
    x = 0.015
    for i, (t, s, c) in enumerate(pasos):
        _caja(ax, x, 0.21, w, h, c, t, s, fs=12)
        if i < 3:
            _flecha(ax, x + w + 0.004, x + w + gap - 0.004, 0.51)
        x += w + gap
    fig.savefig(AQUI / "light_pipeline.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    gradiente_clases()
    augmentation()
    overfit_loss()
    roadmap()
    pipeline_dataset()
    print("Assets light generados en", AQUI)
