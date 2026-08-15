"""
Partición canónica v9 — CONSCIENTE DEL DOMINIO. Compartida por TensorFlow y PyTorch.

Qué cambia respecto de particion_datos.py (v7/v8) y por qué:

  La partición vieja estratificaba por CLASE y repartía al azar. Con un dataset homogéneo
  (todo renders) eso era correcto. Con el dataset v9 —que mezcla renders limpios y fotos
  de pantalla tomadas con celular— deja de serlo, y de una forma que MIENTE hacia arriba:

      Si el test se arma al azar sobre la mezcla, termina con ~75% fotos y ~25% renders.
      Los renders son mucho más fáciles (píxeles exactos, sin moiré, sin glare), así que
      inflan la accuracy reportada. El número que sale NO es el que la app va a ver en
      producción, porque la app SOLO consume fotos de cámara.

  Es el punto 2 de SUGERENCIAS_V9.md, y la regla que impone este módulo:

      >>> TEST y VALIDACIÓN son 100% FOTOS. Los renders van SIEMPRE a entrenamiento. <<<

  Lo que se obtiene es la "accuracy comercial": el porcentaje de aciertos sobre exactamente
  el tipo de imagen que la app Ionic le va a pasar al modelo. Va a ser MÁS BAJA que la de
  la v8, y esa bajada no es una regresión: es que hasta ahora se estaba midiendo otra cosa.

Estructura de la partición (1337 imgs: 1037 fotos + 300 renders):

    dataset/
      ├── TEST            20% de las FOTOS de cada clase   <- se aparta, se toca UNA vez
      ├── TEST_RENDER     20% de los RENDERS de cada clase <- solo para el diagnóstico
      ├── FOLDS (5)       el 80% restante de las FOTOS     <- fold k = validación
      └── SOLO_ENTRENAR   el 80% restante de los RENDERS   <- nunca validan, siempre entrenan

  Los renders no se tiran: entrenar con ellos sigue ayudando (aportan variedad de contenido,
  layouts y paletas que las fotos no cubren). Lo que no pueden hacer es aparecer en la
  métrica que se reporta como resultado. Por eso el grueso vive en un bloque aparte que se
  SUMA al entrenamiento de todos los folds y del refit final.

  TEST_RENDER existe por una razón distinta y es una de las entregas de la v9: permite
  MEDIR la brecha de dominio en vez de solo afirmarla. Al evaluar el MISMO modelo sobre los
  dos tests se obtienen dos números:

      accuracy sobre TEST         = la accuracy comercial (lo que la app va a ver)
      accuracy sobre TEST_RENDER  = la accuracy de laboratorio (lo que reportaban v1-v8)

  La diferencia entre ambos es, con un número concreto, la "ilusión" que describe
  SUGERENCIAS_V9.md §2. Cuesta 60 renders de entrenamiento y convierte un argumento
  cualitativo en una medición.

  Estratificación doble: cada bloque conserva la proporción de clases (como antes) Y ahora
  también la de dominios, porque los folds son puro-foto por construcción.

Guard de huella: igual que la v7. Si el dataset del disco cambia, cargar_particion() aborta
en vez de entrenar en silencio sobre una partición desactualizada.

Uso:
    python documentacion/particion_v9.py                # genera particion_v9.json
    python documentacion/particion_v9.py --mostrar      # imprime la existente
    python documentacion/particion_v9.py --sin-exif     # detección de dominio solo por nombre

Desde los scripts de entrenamiento:
    from particion_v9 import cargar_particion, rutas_y_etiquetas
    part = cargar_particion()
    rutas_tr, y_tr = rutas_y_etiquetas(part, "entrenamiento", fold=0)   # fotos + renders
    rutas_va, y_va = rutas_y_etiquetas(part, "validacion",    fold=0)   # SOLO fotos

numpy/stdlib puro (PIL solo para el EXIF, dentro de dominio.py): importable desde los dos venvs.
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dominio import FOTO, RENDER, conteo_dominios, mapa_dominios  # noqa: E402

# --- 1. Parámetros ---
RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "dataset"
RUTA_PARTICION = Path(__file__).resolve().parent / "particion_v9.json"

SEED = 123          # el MISMO de v1-v8: continuidad del relato entre versiones
N_FOLDS = 5
PROP_TEST = 0.20    # 20% de las FOTOS de cada clase (no del total)

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# --- 2. Listado determinista ---
def listar_por_clase(data_dir: Path) -> dict[str, list[str]]:
    """{clase: [rutas relativas ordenadas]}. Orden alfabético = el de Keras y ImageFolder."""
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
    """SHA-256 (16 chars) del listado completo: guard contra particiones desactualizadas."""
    todas = "\n".join(sorted(r for rutas in por_clase.values() for r in rutas))
    return hashlib.sha256(todas.encode("utf-8")).hexdigest()[:16]


# --- 3. Construcción ---
def construir_particion(data_dir: Path, seed: int, n_folds: int, prop_test: float,
                        usar_exif: bool = True) -> dict:
    """Arma test + folds SOLO CON FOTOS, y manda todos los renders a solo_entrenamiento.

    Por clase:
      fotos    -> se barajan -> primer 20% al TEST, el resto round-robin a los folds
      renders  -> enteros al bloque solo_entrenamiento

    El round-robin sobre la lista ya barajada garantiza folds del mismo tamaño por clase
    aunque el total no sea divisible por n_folds (mismo criterio que la partición v7).
    """
    por_clase = listar_por_clase(data_dir)
    todas = [r for rutas in por_clase.values() for r in rutas]
    print(f"Detectando dominio (foto vs render) de {len(todas)} imágenes"
          f"{' con EXIF' if usar_exif else ' solo por nombre'}...")
    dominios = mapa_dominios(data_dir, todas, usar_exif=usar_exif)

    rng = random.Random(seed)
    test: list[str] = []
    test_render: list[str] = []
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    solo_entrenamiento: list[str] = []
    avisos: list[str] = []

    for clase, rutas in por_clase.items():
        fotos = [r for r in rutas if dominios[r] == FOTO]
        renders = [r for r in rutas if dominios[r] == RENDER]

        if not fotos:
            # Sin fotos no hay métrica comercial posible para esta clase. Se avisa fuerte
            # y se cae al comportamiento viejo (test con renders) para no dejar la clase
            # fuera de la evaluación — pero queda registrado en la meta de la partición.
            avisos.append(
                f"La clase '{clase}' NO tiene fotos: su test se arma con renders y su "
                f"accuracy NO es comparable con la del resto."
            )
            barajadas = list(renders)
            rng.shuffle(barajadas)
            n_test = round(len(barajadas) * prop_test)
            test.extend(barajadas[:n_test])
            for i, ruta in enumerate(barajadas[n_test:]):
                folds[i % n_folds].append(ruta)
            continue

        barajadas = list(fotos)
        rng.shuffle(barajadas)
        n_test = max(1, round(len(barajadas) * prop_test))
        test.extend(barajadas[:n_test])
        for i, ruta in enumerate(barajadas[n_test:]):
            folds[i % n_folds].append(ruta)

        # Renders: la misma proporción se aparta como test_render (diagnóstico de brecha
        # de dominio) y el resto va a entrenamiento.
        barajados_r = list(renders)
        rng.shuffle(barajados_r)
        n_test_r = round(len(barajados_r) * prop_test)
        test_render.extend(barajados_r[:n_test_r])
        solo_entrenamiento.extend(barajados_r[n_test_r:])

    test.sort()
    test_render.sort()
    for f in folds:
        f.sort()
    solo_entrenamiento.sort()

    clases = list(por_clase.keys())
    return {
        "meta": {
            "creado": datetime.now().isoformat(timespec="seconds"),
            "version": "v9",
            "politica": "TEST y VALIDACION son 100% fotos; los renders solo entrenan",
            "seed": seed,
            "n_folds": n_folds,
            "prop_test": prop_test,
            "usar_exif": usar_exif,
            "dataset_dir": data_dir.name,
            "huella_dataset": huella_dataset(por_clase),
            "total": len(todas),
            "total_fotos": sum(1 for d in dominios.values() if d == FOTO),
            "total_renders": sum(1 for d in dominios.values() if d == RENDER),
            "avisos": avisos,
        },
        "clases": clases,
        "dominios": dominios,
        "test": test,
        "test_render": test_render,
        "folds": folds,
        "solo_entrenamiento": solo_entrenamiento,
    }


# --- 4. Guardar / cargar ---
def guardar_particion(particion: dict, ruta: Path = RUTA_PARTICION) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(particion, f, indent=2, ensure_ascii=False)


def cargar_particion(ruta: Path = RUTA_PARTICION, verificar: bool = True) -> dict:
    """Carga particion_v9.json con dos guards: que exista y que el dataset no haya cambiado."""
    if not Path(ruta).is_file():
        raise SystemExit(
            f"No existe la partición v9: {ruta}\n"
            "Generala una vez (los dos frameworks leen el MISMO archivo):\n"
            "    python documentacion/particion_v9.py"
        )
    with open(ruta, encoding="utf-8") as f:
        particion = json.load(f)

    if verificar and DATA_DIR.is_dir():
        actual = huella_dataset(listar_por_clase(DATA_DIR))
        guardada = particion["meta"]["huella_dataset"]
        if actual != guardada:
            raise SystemExit(
                f"El dataset CAMBIÓ desde que se generó la partición v9.\n"
                f"  huella guardada: {guardada}\n"
                f"  huella actual  : {actual}\n"
                "Regenerala:  python documentacion/particion_v9.py\n"
                "OJO: eso reparte el test de nuevo, así que los resultados previos dejan\n"
                "de ser comparables. Anotalo en el informe."
            )
    return particion


# --- 5. Acceso a los subconjuntos ---
def rutas_y_etiquetas(particion: dict, subconjunto: str, fold: int | None = None
                      ) -> tuple[list[str], list[int]]:
    """(rutas relativas, etiquetas enteras) del subconjunto pedido.

      "test"           -> 20% de las fotos. SOLO FOTOS. Es el número que se reporta.
      "test_render"    -> 20% de los renders. SOLO para medir la brecha de dominio.
      "test_mixto"     -> test + test_render. Reproduce cómo evaluaban v1-v8 (inflado).
      "desarrollo"     -> folds + renders de entrenamiento. Es el conjunto del refit final.
      "entrenamiento"  -> todos los folds menos `fold`, MÁS los renders. Requiere fold.
      "validacion"     -> solo el fold `fold`. SOLO FOTOS. Requiere fold.
      "desarrollo_fotos" -> los folds sin los renders (para diagnósticos por dominio).
      "renders"        -> el bloque solo_entrenamiento.

    La etiqueta sale del índice de la clase en particion["clases"] (orden alfabético):
    idéntico criterio al de ImageFolder y al de image_dataset_from_directory, así que el
    entero coincide en los dos frameworks.
    """
    clases = particion["clases"]
    idx_clase = {c: i for i, c in enumerate(clases)}
    renders = list(particion["solo_entrenamiento"])

    if subconjunto == "test":
        rutas = list(particion["test"])
    elif subconjunto == "test_render":
        rutas = list(particion.get("test_render", []))
    elif subconjunto == "test_mixto":
        rutas = sorted(list(particion["test"]) + list(particion.get("test_render", [])))
    elif subconjunto == "desarrollo":
        rutas = [r for f in particion["folds"] for r in f] + renders
    elif subconjunto == "desarrollo_fotos":
        rutas = [r for f in particion["folds"] for r in f]
    elif subconjunto == "renders":
        rutas = renders
    elif subconjunto in ("entrenamiento", "validacion"):
        if fold is None:
            raise ValueError(f"El subconjunto '{subconjunto}' necesita el argumento fold.")
        if subconjunto == "validacion":
            rutas = list(particion["folds"][fold])          # SOLO fotos: métrica honesta
        else:
            rutas = [r for i, f in enumerate(particion["folds"]) if i != fold
                     for r in f] + renders
    else:
        raise ValueError(f"Subconjunto desconocido: {subconjunto}")

    etiquetas = [idx_clase[r.split("/")[0]] for r in rutas]
    return rutas, etiquetas


def conteos(particion: dict, rutas: list[str]) -> dict[str, int]:
    """Cuántas imágenes de cada clase hay en una lista de rutas."""
    return {c: sum(1 for r in rutas if r.startswith(f"{c}/")) for c in particion["clases"]}


def dominios_de(particion: dict, rutas: list[str]) -> list[str]:
    """Dominio (foto/render) de cada ruta, en el mismo orden. Para segmentar métricas."""
    d = particion["dominios"]
    return [d.get(r, RENDER) for r in rutas]


# --- 6. Reporte por consola ---
def mostrar(particion: dict) -> None:
    meta = particion["meta"]
    clases = particion["clases"]
    dom = particion["dominios"]

    print("=" * 74)
    print("  PARTICIÓN CANÓNICA v9 — CONSCIENTE DEL DOMINIO")
    print("=" * 74)
    print(f"  archivo   : {RUTA_PARTICION.name}")
    print(f"  creada    : {meta['creado']}")
    print(f"  política  : {meta['politica']}")
    print(f"  seed      : {meta['seed']}   ·   folds: {meta['n_folds']}   ·   "
          f"test: {meta['prop_test']:.0%} de las fotos")
    print(f"  dataset   : {meta['dataset_dir']}/  ({meta['total']} imágenes = "
          f"{meta['total_fotos']} fotos + {meta['total_renders']} renders)")
    print(f"  huella    : {meta['huella_dataset']}")
    print(f"  clases    : {clases}")

    print("\n  Inventario por clase y dominio:")
    inv = conteo_dominios(dom, clases)
    print(f"    {'clase':<16}{'total':>7}{'fotos':>8}{'renders':>9}")
    print("    " + "-" * 40)
    for c in clases:
        print(f"    {c:<16}{inv[c]['total']:>7}{inv[c][FOTO]:>8}{inv[c][RENDER]:>9}")

    ancho = max(len(c) for c in clases) + 2
    cab = (f"\n  {'bloque':<18}{'total':>7}{'fotos':>7}"
           + "".join(f"{c:>{ancho}}" for c in clases))
    print(cab)
    print("  " + "-" * (len(cab) - 3))

    def fila(nombre, rutas):
        n_fotos = sum(1 for r in rutas if dom.get(r) == FOTO)
        c = conteos(particion, rutas)
        print(f"  {nombre:<18}{len(rutas):>7}{n_fotos:>7}"
              + "".join(f"{c[k]:>{ancho}}" for k in clases))

    fila("TEST (fotos)", rutas_y_etiquetas(particion, "test")[0])
    fila("TEST_RENDER", rutas_y_etiquetas(particion, "test_render")[0])
    for k in range(meta["n_folds"]):
        fila(f"fold {k} (val)", rutas_y_etiquetas(particion, "validacion", fold=k)[0])
    fila("SOLO ENTRENAR", rutas_y_etiquetas(particion, "renders")[0])
    print("  " + "-" * (len(cab) - 3))
    fila("DESARROLLO", rutas_y_etiquetas(particion, "desarrollo")[0])

    for aviso in meta.get("avisos", []):
        print(f"\n  !! AVISO: {aviso}")

    # --- Guards: si alguno falla, todas las métricas del proyecto quedan invalidadas ---
    rutas_test = set(rutas_y_etiquetas(particion, "test")[0])
    rutas_test_r = set(rutas_y_etiquetas(particion, "test_render")[0])
    rutas_dev = set(rutas_y_etiquetas(particion, "desarrollo")[0])

    for nombre, bloque in (("test", rutas_test), ("test_render", rutas_test_r)):
        solapan = bloque & rutas_dev
        if solapan:
            raise SystemExit(
                f"ERROR GRAVE: {len(solapan)} imágenes están en {nombre} Y en desarrollo.")
    if rutas_test & rutas_test_r:
        raise SystemExit("ERROR GRAVE: test y test_render se solapan.")

    total = len(rutas_test) + len(rutas_test_r) + len(rutas_dev)
    print(f"\n  OK: test ({len(rutas_test)}) + test_render ({len(rutas_test_r)}) + "
          f"desarrollo ({len(rutas_dev)}) = {total}, sin solapamiento.")

    # El test reportable tiene que ser puro-foto. test_render es render A PROPÓSITO y no
    # entra en este guard: es el instrumento de diagnóstico, no el número que se reporta.
    renders_en_eval = [r for r in rutas_test if dom.get(r) == RENDER]
    for k in range(meta["n_folds"]):
        renders_en_eval += [r for r in rutas_y_etiquetas(particion, "validacion", fold=k)[0]
                            if dom.get(r) == RENDER]
    if renders_en_eval:
        print(f"  !! {len(renders_en_eval)} renders están en test o en validación "
              f"(clases sin fotos: ver avisos).")
    else:
        print("  OK: NINGÚN render aparece en test ni en validación -> la accuracy que se "
              "reporte\n      es la accuracy comercial (solo fotos de cámara).")
        print("  OK: test_render queda apartado para MEDIR la brecha de dominio "
              "(foto vs render).")


# --- 7. Main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera la partición canónica v9 (domain-aware).")
    ap.add_argument("--semilla", type=int, default=SEED, help=f"semilla (default {SEED})")
    ap.add_argument("--folds", type=int, default=N_FOLDS, help=f"nº de folds (default {N_FOLDS})")
    ap.add_argument("--test", type=float, default=PROP_TEST,
                    help=f"proporción de test sobre las FOTOS (default {PROP_TEST})")
    ap.add_argument("--sin-exif", action="store_true",
                    help="detectar dominio solo por nombre de archivo (más rápido)")
    ap.add_argument("--mostrar", action="store_true",
                    help="solo imprime la partición existente, sin regenerarla")
    args = ap.parse_args()

    if args.mostrar:
        mostrar(cargar_particion())
        sys.exit(0)

    if not DATA_DIR.is_dir():
        raise SystemExit(f"No existe la carpeta de datos: {DATA_DIR}")

    if RUTA_PARTICION.is_file():
        print(f"AVISO: {RUTA_PARTICION.name} ya existe y se va a SOBRESCRIBIR.")
        print("       Si ya reportaste resultados con la partición anterior, dejan de")
        print("       ser comparables (el test se reparte de nuevo).\n")

    particion = construir_particion(DATA_DIR, args.semilla, args.folds, args.test,
                                    usar_exif=not args.sin_exif)
    guardar_particion(particion)
    mostrar(particion)
    print(f"\n  guardado {RUTA_PARTICION}")
