"""
Partición canónica v10 — 4 CLASES: el eje ordinal + la compuerta de rechazo.

QUÉ CAMBIA RESPECTO DE LA v9
----------------------------
La v9 partía 3 clases ordinales con una regla de dominio: TEST y VALIDACIÓN son 100% fotos,
los renders solo entrenan. Esa regla se mantiene intacta — es la que hace que el número que
se reporta sea la accuracy comercial y no la de laboratorio.

Lo que cambia es que ahora hay una CUARTA clase, `3_no_diapositiva`, que no está en el eje
ordinal: es la compuerta del producto ("esto ni siquiera es una diapositiva"). La lógica de
reparto no necesita cambiar para soportarla —`construir_particion` de la v9 ya itera sobre
las clases que encuentre— así que este módulo la REUSA en vez de copiarla. Lo que agrega es
lo que sí es propio de la v10:

  · su propio archivo (particion_v10.json) y su propia huella, para que la partición de la
    v9 siga existiendo y los resultados de esa versión se puedan seguir reproduciendo;
  · la constante N_ORDINALES = 3, que los scripts de entrenamiento le pasan a metricas.py
    para que QWK, MAE y los errores 0<->2 se calculen solo sobre el eje;
  · un guard nuevo y específico: la clase de rechazo TIENE que tener fotos.

POR QUÉ ESE GUARD ES EL IMPORTANTE DE ESTA VERSIÓN
--------------------------------------------------
Las 450 imágenes de la clase de rechazo se bajaron de datasets públicos, y la mitad son
capturas de pantalla y documentos escaneados: renders. Si TODA la clase quedara marcada como
render, pasarían dos cosas, las dos silenciosas y las dos fatales para el experimento:

  1. La clase no tendría test. `construir_particion` cae entonces a su rama de compatibilidad
     y le arma un test con renders — y ese test ya no es comparable con el de las otras tres.
  2. Peor: el modelo podría aprender el ATAJO de dominio. Si en el test las clases 0-2 son
     fotos y la clase 3 son renders, separar la clase 3 se vuelve trivial (basta con detectar
     la textura de render) y el recall de rechazo saldría ~1.00 sin que el modelo haya
     aprendido nada sobre el CONTENIDO. En la app —donde el negativo es la foto real de un
     escritorio— ese modelo fallaría por completo.

Por eso la obtención estratifica a propósito: 300 fotografías de cámara reales (SUN397, COCO)
que declaran dominio `foto`, y 150 capturas/escaneos que declaran `render`. Este módulo aborta
si esa proporción no llegó al disco. Ver `documentacion/negativos_v10.py` y el bloque
"El manifiesto de procedencia" de `documentacion/dominio.py`.

Uso:
    python documentacion/particion_v10.py                # genera particion_v10.json
    python documentacion/particion_v10.py --mostrar      # imprime la existente
    python documentacion/particion_v10.py --sin-exif     # dominio solo por nombre + manifiesto

Desde los scripts de entrenamiento:
    from particion_v10 import N_ORDINALES, cargar_particion, rutas_y_etiquetas
    part = cargar_particion()
    rutas_tr, y_tr = rutas_y_etiquetas(part, "entrenamiento", fold=0)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dominio import FOTO, RENDER, conteo_dominios  # noqa: E402
from particion_v9 import (  # noqa: E402
    N_FOLDS, PROP_TEST, SEED, conteos, construir_particion, dominios_de, guardar_particion,
    huella_dataset, listar_por_clase, rutas_y_etiquetas,
)
import particion_v9  # noqa: E402

# --- 1. Parámetros ---
RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "dataset"
RUTA_PARTICION = Path(__file__).resolve().parent / "particion_v10.json"

CLASE_RECHAZO = "3_no_diapositiva"

# Cuántas de las clases forman el EJE ORDINAL de saturación. Las que sobran (acá una: la de
# rechazo) están fuera del eje. Este número es el que los scripts de entrenamiento le pasan a
# metricas.resumen_completo(..., n_ordinales=N_ORDINALES); ver el encabezado de metricas.py.
N_ORDINALES = 3

# Mínimo de fotos que tiene que tener la clase de rechazo para que su test sea comparable con
# el de las otras. Con PROP_TEST=0.20, 100 fotos dan 20 imágenes de test: por debajo de eso la
# métrica de la compuerta no distingue nada (cada imagen valdría 5 puntos de recall).
MIN_FOTOS_RECHAZO = 100


# --- 2. Construcción ---
def construir(data_dir: Path = DATA_DIR, seed: int = SEED, n_folds: int = N_FOLDS,
              prop_test: float = PROP_TEST, usar_exif: bool = True) -> dict:
    """Arma la partición v10 reusando la lógica de la v9 y le estampa la meta de esta versión."""
    particion = construir_particion(data_dir, seed, n_folds, prop_test, usar_exif=usar_exif)
    particion["meta"].update({
        "version": "v10",
        "politica": ("TEST y VALIDACION son 100% fotos; los renders solo entrenan. "
                     "4 clases: eje ordinal 0-1-2 + compuerta de rechazo 3."),
        "n_ordinales": N_ORDINALES,
        "clase_rechazo": CLASE_RECHAZO if CLASE_RECHAZO in particion["clases"] else None,
    })
    return particion


def cargar_particion(ruta: Path = RUTA_PARTICION, verificar: bool = True) -> dict:
    """Carga particion_v10.json con los mismos dos guards de la v9 (existe · dataset intacto)."""
    if not Path(ruta).is_file():
        raise SystemExit(
            f"No existe la partición v10: {ruta}\n"
            "Generala una vez (los dos frameworks leen el MISMO archivo):\n"
            "    python documentacion/particion_v10.py\n"
            "Y antes, si todavía no bajaste la clase de rechazo:\n"
            "    python documentacion/negativos_v10.py"
        )
    # El guard de huella vive en la v9 y es idéntico; se reusa apuntándolo a este archivo.
    return particion_v9.cargar_particion(ruta, verificar)


# --- 3. Reporte por consola ---
def mostrar(particion: dict) -> None:
    meta = particion["meta"]
    clases = particion["clases"]
    dom = particion["dominios"]
    k = meta.get("n_ordinales", len(clases))

    print("=" * 78)
    print("  PARTICIÓN CANÓNICA v10 — 4 CLASES (eje ordinal + compuerta de rechazo)")
    print("=" * 78)
    print(f"  archivo   : {RUTA_PARTICION.name}")
    print(f"  creada    : {meta['creado']}")
    print(f"  política  : {meta['politica']}")
    print(f"  seed      : {meta['seed']}   ·   folds: {meta['n_folds']}   ·   "
          f"test: {meta['prop_test']:.0%} de las fotos")
    print(f"  dataset   : {meta['dataset_dir']}/  ({meta['total']} imágenes = "
          f"{meta['total_fotos']} fotos + {meta['total_renders']} renders)")
    print(f"  huella    : {meta['huella_dataset']}")
    print(f"  clases    : {clases}")
    print(f"  eje ordinal: {clases[:k]}   ·   fuera del eje: {clases[k:] or '—'}")

    print("\n  Inventario por clase y dominio:")
    inv = conteo_dominios(dom, clases)
    print(f"    {'clase':<18}{'total':>7}{'fotos':>8}{'renders':>9}   rol")
    print("    " + "-" * 64)
    for i, c in enumerate(clases):
        rol = "eje ordinal" if i < k else "COMPUERTA DE RECHAZO"
        print(f"    {c:<18}{inv[c]['total']:>7}{inv[c][FOTO]:>8}{inv[c][RENDER]:>9}   {rol}")

    ancho = max(len(c) for c in clases) + 2
    cab = (f"\n  {'bloque':<18}{'total':>7}{'fotos':>7}"
           + "".join(f"{c:>{ancho}}" for c in clases))
    print(cab)
    print("  " + "-" * (len(cab) - 3))

    def fila(nombre, rutas):
        n_fotos = sum(1 for r in rutas if dom.get(r) == FOTO)
        c = conteos(particion, rutas)
        print(f"  {nombre:<18}{len(rutas):>7}{n_fotos:>7}"
              + "".join(f"{c[x]:>{ancho}}" for x in clases))

    fila("TEST (fotos)", rutas_y_etiquetas(particion, "test")[0])
    fila("TEST_RENDER", rutas_y_etiquetas(particion, "test_render")[0])
    for f in range(meta["n_folds"]):
        fila(f"fold {f} (val)", rutas_y_etiquetas(particion, "validacion", fold=f)[0])
    fila("SOLO ENTRENAR", rutas_y_etiquetas(particion, "renders")[0])
    print("  " + "-" * (len(cab) - 3))
    fila("DESARROLLO", rutas_y_etiquetas(particion, "desarrollo")[0])

    for aviso in meta.get("avisos", []):
        print(f"\n  !! AVISO: {aviso}")

    # --- Guards heredados de la v9: si alguno falla, todas las métricas quedan invalidadas ---
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

    renders_en_eval = [r for r in rutas_test if dom.get(r) == RENDER]
    for f in range(meta["n_folds"]):
        renders_en_eval += [r for r in rutas_y_etiquetas(particion, "validacion", fold=f)[0]
                            if dom.get(r) == RENDER]
    if renders_en_eval:
        print(f"  !! {len(renders_en_eval)} renders están en test o en validación.")
    else:
        print("  OK: ningún render en test ni en validación -> la accuracy reportada es la "
              "comercial.")

    # --- Guard PROPIO de la v10: la compuerta necesita fotos ---
    if CLASE_RECHAZO not in clases:
        raise SystemExit(
            f"ERROR: no existe la clase de rechazo '{CLASE_RECHAZO}' en {DATA_DIR}.\n"
            "Bajala primero:  python documentacion/negativos_v10.py")

    fotos_rechazo = inv[CLASE_RECHAZO][FOTO]
    test_rechazo = sum(1 for r in rutas_test if r.startswith(f"{CLASE_RECHAZO}/"))
    print(f"\n  COMPUERTA DE RECHAZO — {CLASE_RECHAZO}")
    print(f"    {inv[CLASE_RECHAZO]['total']} imágenes = {fotos_rechazo} fotos + "
          f"{inv[CLASE_RECHAZO][RENDER]} renders   ·   test: {test_rechazo} fotos")
    if fotos_rechazo < MIN_FOTOS_RECHAZO:
        raise SystemExit(
            f"\nERROR: la clase de rechazo tiene solo {fotos_rechazo} fotos (mínimo "
            f"{MIN_FOTOS_RECHAZO}).\n"
            "Sin fotos, su test se arma con renders y el modelo puede resolver la compuerta\n"
            "detectando la TEXTURA de render en vez del contenido: el recall saldría ~1.00 en\n"
            "el laboratorio y la app fallaría con la primera foto real de un escritorio.\n"
            "Revisá que documentacion/negativos_v10.py haya escrito _procedencia.json con las\n"
            "familias 'escena' y 'objeto' declaradas como dominio 'foto'.")
    print("    OK: la compuerta se evalúa sobre fotografías reales, no sobre capturas —")
    print("        el modelo no puede resolverla por la textura del dominio.")


# --- 4. Main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera la partición canónica v10 (4 clases).")
    ap.add_argument("--semilla", type=int, default=SEED, help=f"semilla (default {SEED})")
    ap.add_argument("--folds", type=int, default=N_FOLDS, help=f"nº de folds (default {N_FOLDS})")
    ap.add_argument("--test", type=float, default=PROP_TEST,
                    help=f"proporción de test sobre las FOTOS (default {PROP_TEST})")
    ap.add_argument("--sin-exif", action="store_true",
                    help="detectar dominio solo por nombre y manifiesto (más rápido)")
    ap.add_argument("--mostrar", action="store_true",
                    help="solo imprime la partición existente, sin regenerarla")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.mostrar:
        mostrar(cargar_particion())
        sys.exit(0)

    if not DATA_DIR.is_dir():
        raise SystemExit(f"No existe la carpeta de datos: {DATA_DIR}")

    clases_en_disco = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    if CLASE_RECHAZO not in clases_en_disco:
        raise SystemExit(
            f"Falta la clase de rechazo '{CLASE_RECHAZO}'. Bajala primero:\n"
            "    python documentacion/negativos_v10.py")

    if RUTA_PARTICION.is_file():
        print(f"AVISO: {RUTA_PARTICION.name} ya existe y se va a SOBRESCRIBIR.")
        print("       Si ya reportaste resultados con la partición anterior, dejan de")
        print("       ser comparables (el test se reparte de nuevo).\n")

    particion = construir(DATA_DIR, args.semilla, args.folds, args.test,
                          usar_exif=not args.sin_exif)
    guardar_particion(particion, RUTA_PARTICION)
    mostrar(particion)
    print(f"\n  guardado {RUTA_PARTICION}")
    print("\n  Siguiente paso: entrenar los tres modelos de la v10")
    print("    TensorFlow/.venv/Scripts/python TensorFlow/v10/10_scripts.py")
    print("    PyTorch/.venv/Scripts/python    PyTorch/v10/10_scripts.py")
    print("    PyTorch/.venv/Scripts/python    PyTorch/v10/10_modelo_c_desequilibrado.py")
