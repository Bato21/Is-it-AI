"""Consumidor del modelo de diapositivas — PyTorch v2.

Carga ``modelo_diapositivas.pt`` (checkpoint {model_state, config} generado por
entrenar.py) y predice la clase de una o varias imágenes. Independiente del
entrenamiento; la arquitectura vive en modelo.py.

  python consumidor.py imagen.png
  python consumidor.py carpeta/
  python consumidor.py img1.png img2.png ...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from modelo import cargar_checkpoint

logger = logging.getLogger("pytorch.consumidor")


def expandir(args: list[str]) -> list[Path]:
    """Expande archivos y carpetas a una lista ordenada de imágenes válidas."""
    rutas: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            rutas += [f for f in sorted(p.iterdir()) if f.suffix.lower() in config.EXTS]
        elif p.suffix.lower() in config.EXTS:
            rutas.append(p)
    return rutas


def main() -> None:
    config.configurar_logging()
    if len(sys.argv) < 2:
        sys.exit("Uso: python consumidor.py <imagen|carpeta> [...]")

    logger.info("Cargando modelo: %s", config.MODELO_PATH)
    try:
        net, cfg = cargar_checkpoint(config.MODELO_PATH)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    clases = cfg["clases"]
    transformar = transforms.Compose(
        [
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )

    rutas = expandir(sys.argv[1:])
    if not rutas:
        sys.exit("No se encontraron imágenes válidas.")

    for ruta in rutas:
        x = transformar(Image.open(ruta).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(net(x), dim=1)[0].numpy()
        idx = int(probs.argmax())
        barra = "█" * int(probs[idx] * 20)
        # Salida orientada al usuario => print (no logging).
        print(f"\n  {ruta.name}")
        print(f"  -> {clases[idx]}  ({probs[idx]:.1%})  {barra}")
        for i, clase in enumerate(clases):
            print(f"     {clase:<14} {probs[i]:.1%}")


if __name__ == "__main__":
    main()
