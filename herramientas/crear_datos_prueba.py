"""
crear_datos_prueba.py - Genera imágenes sintéticas, distintas por clase, para
probar TODO el flujo (entrenar + evaluar + exportar) ANTES de tener
diapositivas reales etiquetadas.

Cada clase recibe un patrón visual distinto (para que MobileNetV3 pueda
aprender algo y confirmes que el pipeline corre de punta a punta):
  0_sin_ia      -> textura ruidosa tipo foto (parece real / humano)
  1_rastro_ia   -> base degradado + cajas de colores dispersas (mezcla)
  2_saturada_ia -> degradado suave + rectángulo "plantilla" centrado (look IA)

    python tools/crear_datos_prueba.py --por-clase 30     # crear
    python tools/crear_datos_prueba.py --limpiar          # borrar las de prueba

Los archivos se llaman  __prueba__<clase>_<i>.png  para que --limpiar los
encuentre y nunca se confundan con datos reales.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # consola UTF-8 en Windows
except Exception:
    pass

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
CLASES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
SIZE = 256
PREFIJO = "__prueba__"


def _degradado(c1, c2) -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE))
    px = img.load()
    for y in range(SIZE):
        t = y / SIZE
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        for x in range(SIZE):
            px[x, y] = (r, g, b)
    return img


def gen_sin_ia(rng: random.Random) -> Image.Image:
    # textura ruidosa tipo foto real
    img = Image.new("RGB", (SIZE, SIZE))
    px = img.load()
    for y in range(SIZE):
        for x in range(SIZE):
            base = (x * 7 + y * 13) % 90
            px[x, y] = (
                (base + rng.randint(0, 40)) % 256,
                (base + 60 + rng.randint(0, 40)) % 256,
                (base + 120 + rng.randint(0, 40)) % 256,
            )
    return img


def gen_rastro_ia(rng: random.Random) -> Image.Image:
    # mezcla: degradado + cajas de colores (algo humano + algo IA)
    img = _degradado((40, 30, 70), (20, 20, 30))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 12)):
        x, y = rng.randint(0, SIZE - 60), rng.randint(0, SIZE - 60)
        w, h = rng.randint(30, 70), rng.randint(20, 60)
        d.rectangle([x, y, x + w, y + h], fill=tuple(rng.randint(60, 255) for _ in range(3)))
    return img


def gen_saturada_ia(rng: random.Random) -> Image.Image:
    # look plantilla IA: degradado suave + rectángulo centrado
    img = _degradado((124, 92, 255), (25, 211, 197))
    d = ImageDraw.Draw(img)
    m = rng.randint(30, 50)
    d.rectangle([m, m, SIZE - m, SIZE - m], fill=(245, 245, 250))
    d.rectangle([m + 20, m + 30, SIZE - m - 20, m + 60], fill=(200, 200, 215))
    return img


GEN = {"0_sin_ia": gen_sin_ia, "1_rastro_ia": gen_rastro_ia, "2_saturada_ia": gen_saturada_ia}


def crear(por_clase: int, seed: int) -> None:
    rng = random.Random(seed)
    for clase in CLASES:
        d = DATA / clase
        d.mkdir(parents=True, exist_ok=True)
        for i in range(por_clase):
            GEN[clase](rng).save(d / f"{PREFIJO}{clase}_{i:03d}.png")
        print(f"  {clase}: {por_clase} imágenes")
    print(f"\nDatos de prueba en {DATA}. Entrena y luego corre con --limpiar.")


def limpiar() -> None:
    n = 0
    for clase in CLASES:
        d = DATA / clase
        if d.exists():
            for f in d.glob(f"{PREFIJO}*"):
                f.unlink()
                n += 1
    print(f"Eliminadas {n} imagen(es) de prueba.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-clase", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limpiar", action="store_true", help="Borra las imágenes de prueba.")
    args = ap.parse_args()
    limpiar() if args.limpiar else crear(args.por_clase, args.seed)


if __name__ == "__main__":
    main()
