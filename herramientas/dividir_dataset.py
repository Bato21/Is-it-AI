"""
dividir_dataset.py - Reporta el balance de clases del dataset.

    python herramientas/dividir_dataset.py

Los scripts de entrenamiento hacen su propia división train/val/test internamente
(70/15/15 en la v2). Este script es solo un chequeo rápido de balance: clases muy
desbalanceadas sesgan el modelo.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
CLASES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def archivos(clase: str) -> list[Path]:
    d = DATA / clase
    return [p for p in d.iterdir() if p.suffix.lower() in EXTS] if d.exists() else []


def main() -> None:
    print("Balance de clases:")
    total = 0
    for clase in CLASES:
        n = len(archivos(clase))
        total += n
        print(f"  {clase:<14} {n:>5}")
    print(f"  {'TOTAL':<14} {total:>5}")
    if total == 0:
        print("\nAún no hay imágenes. Convierte presentaciones y clasifícalas en dataset/.")
    else:
        mn, mx = min(len(archivos(c)) for c in CLASES), max(len(archivos(c)) for c in CLASES)
        if mx > 0 and mn / mx < 0.5:
            print("\n  Aviso: dataset desbalanceado (la clase menor < 50% de la mayor).")


if __name__ == "__main__":
    main()
