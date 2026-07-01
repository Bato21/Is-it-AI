"""Exporta el modelo entrenado para Android — PyTorch v2.

  python exportar_movil.py

Genera, en esta carpeta:
  modelo_diapositivas.torchscript.pt  — TorchScript (genérico)
  modelo_diapositivas.ptl             — bundle Lite para PyTorch Mobile

En Android: ``org.pytorch:pytorch_android_lite``.
Entrada: 1x3x224x224, normalizada con mean/std de ImageNet (ver config del checkpoint).
"""

from __future__ import annotations

import logging
import sys

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

import config
from modelo import cargar_checkpoint

logger = logging.getLogger("pytorch.exportar")


def main() -> None:
    config.configurar_logging()
    logger.info("Cargando modelo: %s", config.MODELO_PATH)
    try:
        net, cfg = cargar_checkpoint(config.MODELO_PATH)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    ejemplo = torch.randn(1, 3, cfg["img_size"], cfg["img_size"])
    traced = torch.jit.trace(net, ejemplo)

    ts = config.AQUI / "modelo_diapositivas.torchscript.pt"
    traced.save(str(ts))
    print(f"Guardado TorchScript: {ts}")

    opt = optimize_for_mobile(traced)
    ptl = config.AQUI / "modelo_diapositivas.ptl"
    opt._save_for_lite_interpreter(str(ptl))
    print(f"Guardado Lite: {ptl}")
    print(
        "Android: org.pytorch:pytorch_android_lite . "
        "Entrada 1x3x224x224, normalizar con mean/std ImageNet."
    )


if __name__ == "__main__":
    main()
