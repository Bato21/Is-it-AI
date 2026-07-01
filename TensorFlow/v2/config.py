"""Configuración central del lado TensorFlow v2: hiperparámetros, rutas y clases.

Incluye también la inicialización de logging y consola (UTF-8), para que cada
script solo tenga que llamar a :func:`configurar_logging` al arrancar.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

DATASET_DIR = RAIZ / "dataset"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALIDATION_SPLIT = 0.30  # 70% train; el resto se parte 50/50 en val/test

# Orden de clases fijo (0,1,2), NO alfabético sorpresa.
CLASES: list[str] = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
N_CLASES = len(CLASES)
COLORES: list[str] = ["#1E7B45", "#B45309", "#991B1B"]

IMG_SIZE: tuple[int, int] = (224, 224)  # entrada nativa de MobileNetV3Large
BATCH_SIZE = 32
DENSE_UNITS = 256
DROPOUT = 0.3
SEED = 42

# Entrenamiento en dos fases: cabeza congelada -> fine-tuning.
EPOCHS_FASE1 = 15
EPOCHS_FASE2 = 10
FINE_TUNE_AT = 200  # descongelar capas del base desde este índice
LR_FASE1 = 1e-3
LR_FASE2 = 1e-4  # LR más bajo para no romper los pesos preentrenados

MODELO_PATH = AQUI / "modelo_diapositivas.keras"
CURVAS_PATH = AQUI / "training_curves.png"
MATRIZ_PATH = AQUI / "confusion_matrix.png"
ROC_PATH = AQUI / "roc_curves.png"
GRADCAM_PATH = AQUI / "grad_cam_output.png"

# Etiquetas de las figuras (metricas.py).
TITULO_CURVAS = "MobileNetV3Large — Curvas de entrenamiento"
TITULO_MATRIZ = "MobileNetV3Large — Evaluación en test"
TITULO_ROC = "MobileNetV3Large — Curvas ROC (One-vs-Rest)"
ETIQUETA_LOSS = "Categorical Crossentropy"


def configurar_logging() -> None:
    """Inicializa logging (nivel desde IS_IT_AI_LOG, def. INFO) y consola UTF-8."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8")  # consola Windows usa cp1252
        except Exception:  # noqa: BLE001
            pass
    nivel = getattr(logging, os.environ.get("IS_IT_AI_LOG", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
