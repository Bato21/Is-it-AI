"""
Partición canónica del dataset — COMPARTIDA por TensorFlow y PyTorch.

Por qué existe este archivo (el agujero que tapa):
  Hasta la v6 cada framework partía el dataset por su cuenta:
    - TensorFlow : image_dataset_from_directory(validation_split=0.2, seed=123)
    - PyTorch    : random_split(range(len(base)), [n_train, n_val], generator=seed 123)
  Las dos son reproducibles, pero NO dan el mismo split. Eso dejaba dos problemas
  documentados en analisis_metricas.md §4:

    1. "Split distinto entre frameworks": comparar 90.0% (PyTorch v4) contra 80.0%
       (TF v4) tiene un asterisco, porque no evalúan sobre las mismas imágenes.
    2. "Validación chica y sin estratificar": 60 imágenes repartidas 24/18/18 en vez
       de 20/20/20, así que los recalls por clase se apoyan en soportes desparejos.

  Y había un tercer problema, más grave, que apareció recién con la v6: Optuna elegía
  los hiperparámetros MAXIMIZANDO val_accuracy y después se reportaba la accuracy del
  modelo reentrenado SOBRE ESE MISMO val. Con 30 trials compitiendo por 60 imágenes,
  ese número está optimistamente sesgado: es "el mejor de 30 intentos sobre el mismo
  examen", no una estimación honesta de cómo generaliza. En la literatura esto es
  sesgo de selección, y es el error clásico de las búsquedas de hiperparámetros.

Qué hace este módulo:
  Genera UNA partición canónica, estratificada y determinista, y la guarda en
  'particion.json'. Los dos frameworks leen ESE archivo, así que a partir de la v7
  entrenan y evalúan sobre exactamente las mismas imágenes.

  Estructura de la partición (300 imgs, 100 por clase):

      dataset/  (300)
        ├── TEST      20%  = 60 imgs (20 por clase)   <- se aparta y NO se toca
        │                                                hasta el reporte final
        └── DESARROLLO 80% = 240 imgs (80 por clase)
              └── 5 folds estratificados de 48 imgs (16 por clase)
                    fold k = validación · los otros 4 = entrenamiento

  - La búsqueda de hiperparámetros (Optuna) usa SOLO desarrollo, y su objetivo es la
    MEDIA de val_accuracy sobre los 5 folds. Promediar 5 validaciones de 48 imágenes
    es mucho más estable que maximizar una sola validación de 60.
  - El TEST se usa UNA vez, al final, con los hiperparámetros ya congelados. Ese es
    el número que se reporta y se defiende.

Uso:
    python documentacion/particion_datos.py              # genera particion.json
    python documentacion/particion_datos.py --semilla 7  # otra partición
    python documentacion/particion_datos.py --mostrar    # solo imprime la que existe

Desde los scripts de entrenamiento:
    from particion_datos import cargar_particion, rutas_y_etiquetas
    part = cargar_particion()
    rutas, etiquetas = rutas_y_etiquetas(part, "entrenamiento", fold=0)

Nota: NO usa sklearn ni ningún framework. Es numpy/stdlib puro a propósito, para que
el mismo archivo se pueda importar desde el venv de TensorFlow y desde el de PyTorch
sin arrastrar dependencias cruzadas.
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# --- 1. Parámetros de la partición ---
RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "dataset"
RUTA_PARTICION = Path(__file__).resolve().parent / "particion.json"

SEED = 123          # MISMO seed que v1-v6, para no romper la continuidad del relato
N_FOLDS = 5         # folds de validación cruzada dentro de desarrollo
PROP_TEST = 0.20    # 20% apartado como test (60 de 300)

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# --- 2. Listado determinista de archivos ---
def listar_por_clase(data_dir: Path) -> dict[str, list[str]]:
    """Devuelve {clase: [rutas relativas ordenadas]}.

    Las clases salen ORDENADAS alfabéticamente, igual que las ordenan Keras
    (image_dataset_from_directory) y PyTorch (ImageFolder). Los prefijos 0_/1_/2_
    hacen que ese orden alfabético coincida con el eje ordinal de saturación —
    el gotcha que el proyecto viene resolviendo desde la v1.

    El sorted() de los archivos NO es cosmético: garantiza que dos corridas en
    máquinas distintas (donde el orden del sistema de archivos puede diferir)
    produzcan la MISMA partición a partir del mismo seed.
    """
    clases = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not clases:
        raise SystemExit(f"No hay subcarpetas de clase en {data_dir}")
    por_clase = {}
    for clase in clases:
        archivos = sorted(
            p.name for p in (data_dir / clase).iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES
        )
        if not archivos:
            raise SystemExit(f"La clase '{clase}' no tiene imágenes en {data_dir / clase}")
        por_clase[clase] = [f"{clase}/{nombre}" for nombre in archivos]
    return por_clase


def huella_dataset(por_clase: dict[str, list[str]]) -> str:
    """SHA-256 del listado completo de archivos.

    Sirve de GUARD: si mañana se agregan o borran imágenes, la huella cambia y
    cargar_particion() avisa en vez de entrenar en silencio sobre una partición
    que ya no corresponde al dataset del disco. Sin esto, agregar 20 imágenes a la
    clase 2 dejaría el test contaminado sin que nadie se entere.
    """
    todas = "\n".join(sorted(r for rutas in por_clase.values() for r in rutas))
    return hashlib.sha256(todas.encode("utf-8")).hexdigest()[:16]


# --- 3. Construcción de la partición estratificada ---
def construir_particion(data_dir: Path, seed: int, n_folds: int, prop_test: float) -> dict:
    """Arma test + folds ESTRATIFICADOS (misma proporción de clases en cada bloque).

    Estratificar = repartir clase por clase, no sobre la bolsa entera. Con 100
    imágenes por clase y 5 folds, cada fold recibe exactamente 16 de cada clase.
    Esto elimina de raíz el 24/18/18 de las versiones anteriores: ahora los soportes
    por clase son iguales y los recalls se comparan sin corregir mentalmente.

    El reparto en folds es round-robin sobre la lista YA barajada de cada clase
    (i-ésima imagen -> fold i % n_folds), que es la forma más simple de garantizar
    que los folds queden balanceados incluso si el total no es divisible.
    """
    por_clase = listar_por_clase(data_dir)
    rng = random.Random(seed)  # RNG propio: no toca el estado global de random

    test: list[str] = []
    folds: list[list[str]] = [[] for _ in range(n_folds)]

    for clase, rutas in por_clase.items():
        barajadas = list(rutas)
        rng.shuffle(barajadas)
        n_test = round(len(barajadas) * prop_test)
        test.extend(barajadas[:n_test])
        for i, ruta in enumerate(barajadas[n_test:]):
            folds[i % n_folds].append(ruta)

    # Orden estable dentro de cada bloque: la partición es un conjunto, no una
    # secuencia; ordenarla hace que el JSON sea diffeable entre corridas.
    test.sort()
    for f in folds:
        f.sort()

    clases = list(por_clase.keys())
    return {
        "meta": {
            "creado": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "n_folds": n_folds,
            "prop_test": prop_test,
            "dataset_dir": data_dir.name,
            "huella_dataset": huella_dataset(por_clase),
            "total": sum(len(v) for v in por_clase.values()),
        },
        "clases": clases,
        "test": test,
        "folds": folds,
    }


# --- 4. Guardar / cargar ---
def guardar_particion(particion: dict, ruta: Path = RUTA_PARTICION) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(particion, f, indent=2, ensure_ascii=False)


def cargar_particion(ruta: Path = RUTA_PARTICION, verificar: bool = True) -> dict:
    """Carga particion.json con dos guards: que exista y que el dataset no haya cambiado."""
    if not Path(ruta).is_file():
        raise SystemExit(
            f"No existe la partición canónica: {ruta}\n"
            "Generala una vez (los dos frameworks leen el MISMO archivo):\n"
            "    python documentacion/particion_datos.py"
        )
    with open(ruta, encoding="utf-8") as f:
        particion = json.load(f)

    if verificar and DATA_DIR.is_dir():
        actual = huella_dataset(listar_por_clase(DATA_DIR))
        guardada = particion["meta"]["huella_dataset"]
        if actual != guardada:
            raise SystemExit(
                f"El dataset CAMBIÓ desde que se generó la partición.\n"
                f"  huella guardada: {guardada}\n"
                f"  huella actual  : {actual}\n"
                "Si agregaste o borraste imágenes, regenerá la partición:\n"
                "    python documentacion/particion_datos.py\n"
                "OJO: eso reparte el test de nuevo, así que los resultados previos\n"
                "dejan de ser comparables. Anotalo en el informe."
            )
    return particion


# --- 5. Acceso a los subconjuntos ---
def rutas_y_etiquetas(particion: dict, subconjunto: str, fold: int | None = None
                      ) -> tuple[list[str], list[int]]:
    """Devuelve (rutas relativas, etiquetas enteras) del subconjunto pedido.

    subconjunto:
      "test"          -> el 20% apartado (60 imgs). Se usa UNA sola vez, al final.
      "desarrollo"    -> los 5 folds juntos (240 imgs). Para el refit final.
      "entrenamiento" -> todos los folds MENOS `fold` (192 imgs). Requiere fold.
      "validacion"    -> solo el fold `fold` (48 imgs). Requiere fold.

    La etiqueta sale del índice de la clase en particion["clases"], que está
    ordenada alfabéticamente: idéntico criterio al de ImageFolder y al de
    image_dataset_from_directory. Por eso el entero coincide en los dos frameworks.
    """
    clases = particion["clases"]
    idx_clase = {c: i for i, c in enumerate(clases)}

    if subconjunto == "test":
        rutas = list(particion["test"])
    elif subconjunto == "desarrollo":
        rutas = [r for f in particion["folds"] for r in f]
    elif subconjunto in ("entrenamiento", "validacion"):
        if fold is None:
            raise ValueError(f"El subconjunto '{subconjunto}' necesita el argumento fold.")
        if subconjunto == "validacion":
            rutas = list(particion["folds"][fold])
        else:
            rutas = [r for i, f in enumerate(particion["folds"]) if i != fold for r in f]
    else:
        raise ValueError(f"Subconjunto desconocido: {subconjunto}")

    etiquetas = [idx_clase[r.split("/")[0]] for r in rutas]
    return rutas, etiquetas


def conteos(particion: dict, rutas: list[str]) -> dict[str, int]:
    """Cuántas imágenes de cada clase hay en una lista de rutas (para imprimir soportes)."""
    return {c: sum(1 for r in rutas if r.startswith(f"{c}/")) for c in particion["clases"]}


# --- 6. Reporte por consola ---
def mostrar(particion: dict) -> None:
    meta = particion["meta"]
    clases = particion["clases"]
    print("=" * 66)
    print("  PARTICIÓN CANÓNICA (compartida por TensorFlow y PyTorch)")
    print("=" * 66)
    print(f"  archivo   : {RUTA_PARTICION.name}")
    print(f"  creada    : {meta['creado']}")
    print(f"  seed      : {meta['seed']}   ·   folds: {meta['n_folds']}   ·   "
          f"test: {meta['prop_test']:.0%}")
    print(f"  dataset   : {meta['dataset_dir']}/  ({meta['total']} imágenes)")
    print(f"  huella    : {meta['huella_dataset']}")
    print(f"  clases    : {clases}")
    print()

    ancho = max(len(c) for c in clases) + 2
    cab = f"  {'bloque':<14}{'total':>7}" + "".join(f"{c:>{ancho}}" for c in clases)
    print(cab)
    print("  " + "-" * (len(cab) - 2))

    rutas_test, _ = rutas_y_etiquetas(particion, "test")
    ct = conteos(particion, rutas_test)
    print(f"  {'TEST':<14}{len(rutas_test):>7}" + "".join(f"{ct[c]:>{ancho}}" for c in clases))

    for k in range(meta["n_folds"]):
        rutas_f, _ = rutas_y_etiquetas(particion, "validacion", fold=k)
        cf = conteos(particion, rutas_f)
        print(f"  {'fold ' + str(k):<14}{len(rutas_f):>7}"
              + "".join(f"{cf[c]:>{ancho}}" for c in clases))

    rutas_dev, _ = rutas_y_etiquetas(particion, "desarrollo")
    cd = conteos(particion, rutas_dev)
    print("  " + "-" * (len(cab) - 2))
    print(f"  {'DESARROLLO':<14}{len(rutas_dev):>7}" + "".join(f"{cd[c]:>{ancho}}" for c in clases))
    print()

    # Guard de solapamiento: test y desarrollo tienen que ser disjuntos. Si esto
    # alguna vez falla, todas las métricas del proyecto quedan invalidadas.
    solapan = set(rutas_test) & set(rutas_dev)
    if solapan:
        raise SystemExit(f"ERROR GRAVE: {len(solapan)} imágenes están en test Y en desarrollo.")
    print(f"  OK: test y desarrollo son disjuntos ({len(rutas_test)} + {len(rutas_dev)} "
          f"= {len(rutas_test) + len(rutas_dev)} imágenes, sin solapamiento).")


# --- 7. Main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera la partición canónica del dataset.")
    ap.add_argument("--semilla", type=int, default=SEED, help=f"semilla (default {SEED})")
    ap.add_argument("--folds", type=int, default=N_FOLDS, help=f"nº de folds (default {N_FOLDS})")
    ap.add_argument("--test", type=float, default=PROP_TEST,
                    help=f"proporción de test (default {PROP_TEST})")
    ap.add_argument("--mostrar", action="store_true",
                    help="solo imprime la partición existente, sin regenerarla")
    args = ap.parse_args()

    if args.mostrar:
        mostrar(cargar_particion())
        sys.exit(0)

    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"No existe la carpeta de datos: {DATA_DIR}\n"
            "Colocá el dataset ahí (subcarpetas 0_sin_ia/ 1_rastro_ia/ 2_saturada_ia/), o\n"
            "generá datos sintéticos para probar el flujo de punta a punta:\n"
            "    python documentacion/crear_datos_prueba.py --por-clase 30"
        )

    if RUTA_PARTICION.is_file():
        print(f"AVISO: {RUTA_PARTICION.name} ya existe y se va a SOBRESCRIBIR.")
        print("       Si ya reportaste resultados con la partición anterior, dejan de")
        print("       ser comparables (el test se reparte de nuevo).\n")

    particion = construir_particion(DATA_DIR, args.semilla, args.folds, args.test)
    guardar_particion(particion)
    mostrar(particion)
    print(f"\n  guardado {RUTA_PARTICION}")
