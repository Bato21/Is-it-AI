"""Definición del modelo PyTorch v2 (única fuente de verdad).

Antes la arquitectura estaba repetida en ``entrenar.py``, ``consumidor.py`` y
``exportar_movil.py``. Vive aquí una sola vez; los demás scripts la importan.
También centraliza el guardado/carga del checkpoint ``{model_state, config}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

import config

ARQUITECTURA = "mobilenet_v3_large + cabeza Linear(960->256)->Dropout->Linear(256->3)"


def construir_modelo(num_clases: int = config.N_CLASES, preentrenado: bool = False) -> nn.Module:
    """Crea MobileNetV3Large con la cabeza del proyecto.

    ``preentrenado=True`` carga los pesos de ImageNet (para entrenar); con
    ``False`` la red queda sin pesos, lista para ``load_state_dict`` desde un
    checkpoint (consumidor / exportación).
    """
    pesos = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if preentrenado else None
    net = mobilenet_v3_large(weights=pesos)
    in_f = net.classifier[0].in_features  # 960
    net.classifier = nn.Sequential(
        nn.Linear(in_f, config.DENSE_UNITS),
        nn.Hardswish(),
        nn.Dropout(config.DROPOUT),
        nn.Linear(config.DENSE_UNITS, num_clases),
    )
    return net


def set_features_trainable(
    net: nn.Module, trainable: bool, ultimos_bloques: int | None = None
) -> None:
    """Congela/descongela el extractor de características.

    Si ``ultimos_bloques`` se indica, solo descongela los últimos N bloques
    (fine-tuning parcial, equivalente a ``FINE_TUNE_AT`` en Keras).
    """
    bloques = list(net.features.children())
    for b in bloques:
        for p in b.parameters():
            p.requires_grad = False
    if trainable:
        objetivo = bloques if ultimos_bloques is None else bloques[-ultimos_bloques:]
        for b in objetivo:
            for p in b.parameters():
                p.requires_grad = True


def guardar_checkpoint(net: nn.Module, ruta: Path) -> None:
    """Guarda ``{model_state, config}`` para reconstruir el modelo sin el script."""
    cfg = {
        "clases": config.CLASES,
        "img_size": config.IMG_SIZE,
        "mean": config.MEAN,
        "std": config.STD,
        "arquitectura": ARQUITECTURA,
    }
    torch.save({"model_state": net.state_dict(), "config": cfg}, ruta)


def cargar_checkpoint(ruta: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Carga el checkpoint y devuelve ``(net en modo eval, config)``."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe '{ruta}'. Entrena primero: python entrenar.py")
    ckpt = torch.load(ruta, map_location="cpu")
    cfg = ckpt["config"]
    net = construir_modelo(num_clases=len(cfg["clases"]), preentrenado=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, cfg
