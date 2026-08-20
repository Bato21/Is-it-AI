"""
verificar_dataset.py - Chequeo formal del dataset ANTES de entrenar.

Es el eslabón "verificable" de la separación obtención <-> entrenamiento (req. 2 de
la 2a entrega): ningún script de entrenamiento toca los datos; este script solo LEE
la carpeta dataset/ y reporta su estado. Pensado para encadenarse antes de entrenar:

    python documentacion/verificar_dataset.py && python TensorFlow/v4/04_scripts.py

Qué hace:
  - Cuenta imágenes por clase (extensiones válidas: jpg/jpeg/png/webp).
  - Imprime una tabla con totales y porcentajes.
  - Avisa si el desbalance entre clases supera ~10%.
  - Avisa si alguna clase tiene menos de 30 imágenes.
  - Sale con código != 0 si falta (o queda vacía) alguna de las 3 carpetas de clase,
    para cortar el pipeline antes de entrenar sobre datos incompletos.

Uso:
    python documentacion/verificar_dataset.py                  # verifica dataset/
    python documentacion/verificar_dataset.py dataset_desbalanceado   # otra carpeta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]

# Clases MÍNIMAS que tienen que existir sí o sí: son el eje ordinal, sin ellas no hay
# problema que resolver. La lista real se descubre leyendo las subcarpetas del disco (ver
# `clases_de`), así que agregar una clase al dataset —como hizo la v10 con la compuerta
# `3_no_diapositiva`— no requiere tocar este archivo.
#
# Tenerlas hardcodeadas era un error silencioso: el verificador informaba "OK, 3 clases
# presentes" sobre un dataset de 4, y el conteo que imprimía no era el del dataset que los
# scripts de entrenamiento iban a leer.
CLASES_REQUERIDAS = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MIN_POR_CLASE = 30          # umbral de "pocas imágenes" por clase
TOLERANCIA_DESBALANCE = 0.10  # 10%: gap relativo aceptable entre la mayor y la menor


def clases_de(carpeta: Path) -> list[str]:
    """Las clases que hay en el disco, en orden alfabético — el mismo de Keras/ImageFolder.

    Se unen con las requeridas para que una clase ausente aparezca igual en la tabla como
    AUSENTE, en vez de desaparecer del reporte.
    """
    presentes = sorted(p.name for p in carpeta.iterdir() if p.is_dir()) if carpeta.is_dir() else []
    return sorted(set(presentes) | set(CLASES_REQUERIDAS))


def contar(carpeta: Path, clases: list[str]) -> list[int]:
    """Imágenes válidas por clase (misma extensión que usan los loaders)."""
    return [
        len([p for p in (carpeta / c).iterdir() if p.suffix.lower() in EXTS])
        if (carpeta / c).is_dir()
        else -1  # -1 marca "carpeta ausente"
        for c in clases
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifica el dataset antes de entrenar.")
    ap.add_argument(
        "carpeta",
        nargs="?",
        default="dataset",
        help="Carpeta de dataset relativa a la raíz del repo (def: dataset).",
    )
    args = ap.parse_args()

    data = (ROOT / args.carpeta).resolve()
    print(f"Verificando: {data}\n")

    if not data.exists():
        print(f"ERROR: no existe la carpeta '{data}'.")
        print("Generá datos sintéticos para probar el flujo:")
        print("    python documentacion/crear_datos_prueba.py --por-clase 30")
        sys.exit(1)

    clases = clases_de(data)
    conteos = contar(data, clases)
    total = sum(n for n in conteos if n > 0)

    # --- Tabla ---
    ancho = max(18, max(len(c) for c in clases) + 2)
    print(f"{'clase':<{ancho}}{'imágenes':>10}{'porcentaje':>13}")
    print("-" * (ancho + 23))
    for clase, n in zip(clases, conteos):
        if n < 0:
            print(f"{clase:<{ancho}}{'AUSENTE':>10}{'—':>13}")
        else:
            pct = (n / total * 100) if total else 0.0
            print(f"{clase:<{ancho}}{n:>10}{pct:>12.1f}%")
    print("-" * (ancho + 23))
    print(f"{'TOTAL':<{ancho}}{total:>10}\n")

    # --- Validación dura: carpetas ausentes o vacías cortan el pipeline ---
    faltantes = [c for c, n in zip(clases, conteos) if n < 0]
    vacias = [c for c, n in zip(clases, conteos) if n == 0]
    if faltantes or vacias:
        if faltantes:
            print(f"ERROR: faltan carpetas de clase: {', '.join(faltantes)}")
        if vacias:
            print(f"ERROR: carpetas de clase sin imágenes válidas: {', '.join(vacias)}")
        print("Cada clase necesita su carpeta con imágenes antes de entrenar.")
        sys.exit(1)

    # --- Avisos blandos (no cortan; son señal para el análisis) ---
    avisos = []
    mn, mx = min(conteos), max(conteos)
    if mx > 0 and (mx - mn) / mx > TOLERANCIA_DESBALANCE:
        avisos.append(
            f"desbalance {(mx - mn) / mx * 100:.0f}% entre la clase mayor ({mx}) y la "
            f"menor ({mn}); supera el {TOLERANCIA_DESBALANCE * 100:.0f}% de tolerancia."
        )
    pocas = [f"{c} ({n})" for c, n in zip(clases, conteos) if n < MIN_POR_CLASE]
    if pocas:
        avisos.append(f"clases con menos de {MIN_POR_CLASE} imágenes: {', '.join(pocas)}.")

    if avisos:
        print("Avisos:")
        for a in avisos:
            print(f"  - {a}")
    else:
        print(f"OK: {len(clases)} clases presentes, balance dentro de tolerancia.")

    sys.exit(0)


if __name__ == "__main__":
    main()
