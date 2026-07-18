"""
crear_subset_desbalanceado.py - Arma un subconjunto DESBALANCEADO del dataset para la v5.

Copia de forma DETERMINISTA (mismo resultado en cada corrida) desde dataset/ hacia
dataset_desbalanceado/ con 100/50/25 imágenes por clase (0/1/2), o proporcional si
alguna clase tiene menos. NO borra ni modifica dataset/ (solo lee de ahí).

Justificación de dominio: en producción la mayoría de las presentaciones son humanas,
así que una distribución realista se parece más a 100/50/25 que a 100/100/100. La v5
entrena sobre este subset para analizar el efecto del desbalance y su corrección con
pesos de clase.

    python documentacion/crear_subset_desbalanceado.py            # crear el subset
    python documentacion/crear_subset_desbalanceado.py --limpiar  # borrar el subset

Después:  python documentacion/verificar_dataset.py dataset_desbalanceado
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
ORIGEN = ROOT / "dataset"
DESTINO = ROOT / "dataset_desbalanceado"
# Distribución objetivo (clase mayoritaria humana; la clase 2 saturada, minoritaria).
OBJETIVO = {"0_sin_ia": 100, "1_rastro_ia": 50, "2_saturada_ia": 25}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 123


def limpiar() -> None:
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
        print(f"Borrado {DESTINO}")
    else:
        print(f"No existe {DESTINO}, nada que borrar.")


def crear() -> None:
    if not ORIGEN.is_dir():
        raise SystemExit(
            f"No existe la carpeta de origen '{ORIGEN}'.\n"
            "Generá datos sintéticos para probar el flujo:\n"
            "    python documentacion/crear_datos_prueba.py --por-clase 30"
        )

    # Empezar de cero: solo se toca DESTINO, jamás ORIGEN.
    if DESTINO.exists():
        shutil.rmtree(DESTINO)

    print(f"Origen : {ORIGEN}")
    print(f"Destino: {DESTINO}\n")
    print(f"{'clase':<16}{'disponibles':>12}{'copiadas':>10}")
    print("-" * 38)

    for clase, objetivo in OBJETIVO.items():
        src = ORIGEN / clase
        if not src.is_dir():
            raise SystemExit(f"Falta la carpeta de clase en el origen: {src}")

        # Determinismo: ordenar por nombre (base estable) y elegir con seed fijo, así
        # el subset es reproducible y representativo (no sesgado al primer deck alfabético).
        disponibles = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
        n = min(objetivo, len(disponibles))
        elegidas = random.Random(SEED).sample(disponibles, n)

        dst = DESTINO / clase
        dst.mkdir(parents=True, exist_ok=True)
        for p in elegidas:
            shutil.copy2(p, dst / p.name)

        print(f"{clase:<16}{len(disponibles):>12}{n:>10}")

    total = sum(len(list((DESTINO / c).iterdir())) for c in OBJETIVO)
    print("-" * 38)
    print(f"{'TOTAL':<16}{'':>12}{total:>10}\n")
    print(f"Subset desbalanceado en {DESTINO} (dataset/ intacto).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Crea dataset_desbalanceado/ (100/50/25).")
    ap.add_argument("--limpiar", action="store_true", help="Borra dataset_desbalanceado/.")
    args = ap.parse_args()
    limpiar() if args.limpiar else crear()


if __name__ == "__main__":
    main()
