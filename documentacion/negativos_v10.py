"""
Obtención de la CLASE DE RECHAZO de la v10 — `3_no_diapositiva`.

EL PROBLEMA QUE RESUELVE ESTA CLASE
-----------------------------------
Hasta la v9 el modelo tenía 3 salidas y ninguna de ellas era "esto no es una diapositiva".
Un softmax de 3 clases SIEMPRE reparte 1.0 entre sus 3 opciones: si el usuario apunta la
cámara a su escritorio, a la cara de un compañero o a una página de Excel, el modelo no puede
abstenerse — devuelve un nivel de saturación de IA, con confianza alta, para algo que ni
siquiera es una diapositiva. Es el modo de falla más embarazoso del producto, porque no falla
"un poco": responde con seguridad a una pregunta que nadie hizo.

La v10 agrega una CUARTA clase que no está en el eje ordinal: es la compuerta (`gate`) del
producto. Si gana, la app no reporta nivel de IA — dice que la foto no es una diapositiva.

POR QUÉ 450 IMÁGENES (la pregunta que originó este script)
----------------------------------------------------------
Las tres clases existentes tienen 434 / 435 / 468 (promedio 446). El número elegido es 450, y
la razón es que es el número que NO privilegia a la clase nueva ni la castiga:

  COTA SUPERIOR — distorsión de la prior. Si la clase de rechazo fuera mucho más grande que
  las demás, dominaría las métricas macro (es la clase más fácil de las cuatro: separar
  "diapositiva" de "no diapositiva" es un margen visual enorme comparado con separar "rastro"
  de "saturada"). La accuracy reportada subiría sin que el producto mejorara en lo que
  importa. Además obligaría a meter pesos de clase, y eso rompería la comparabilidad con la
  v9, que entrena con cross-entropy plana.

  COTA INFERIOR — varianza intra-clase. "Todo lo que no es una diapositiva" es un conjunto
  abierto: no es un concepto, son al menos cuatro familias visuales distintas (una escena,
  un objeto, una interfaz, un documento). Con menos de ~100 por familia el modelo memoriza
  ejemplos en vez de aprender el complemento.

  450 satisface las dos: es un cuarto exacto del dataset resultante (450 / 1787 = 25.2%), o
  sea el reparto neutro de un problema de 4 clases, y deja ~112 por familia.

  Y —esto es lo importante— el número NO se afirma: se MIDE. Los scripts de entrenamiento de
  la v10 corren una ablación con el 25%, el 50% y el 100% de esta clase y reportan cómo se
  mueven el recall de rechazo y el recall de la clase 0. Si la curva ya está plana en 225, el
  informe lo dice; si sigue subiendo en 450, también. Ver `documentacion/analisis_v10.md`.

LAS CUATRO FAMILIAS, Y POR QUÉ NO ALCANZA CON "FOTOS AL AZAR"
-------------------------------------------------------------
Si la clase de rechazo fuera solo fotos de perros y de playas, el modelo aprendería la
frontera trivial "¿hay una pantalla rectangular con texto?" y seguiría analizando como
diapositiva la foto de un monitor con una planilla abierta — que es el caso que de verdad
pasa. Por eso la clase se estratifica a propósito en dos mitades:

  NEGATIVOS FÁCILES (300 imgs, dominio FOTO) — lo que el usuario fotografía por accidente.
    escena   160  SUN397: habitaciones, oficinas, calles, aulas.
    objeto   140  COCO:  personas, objetos, escritorios, comida.

  NEGATIVOS DIFÍCILES (150 imgs, dominio RENDER) — lo que está EN una pantalla y no es una
  diapositiva. Es la frontera que decide si el producto sirve.
    interfaz 100  capturas de páginas web y de UI de escritorio/móvil.
    documento 50  formularios y documentos escaneados: texto sobre fondo blanco, que es
                  visualmente lo más parecido a una diapositiva sobria de la clase 0.

DOMINIO DECLARADO (y por qué no se deja que lo adivine el heurístico)
--------------------------------------------------------------------
`dominio.py` decide foto-vs-render por EXIF y por nombre de archivo. Ninguna de las dos
señales sirve acá: estas imágenes vienen de datasets académicos que ya perdieron el EXIF y
cuyos nombres no son de galería. El heurístico las mandaría TODAS a RENDER (su caso
conservador), y entonces la clase de rechazo quedaría sin fotos: su test se armaría con
renders y su accuracy dejaría de ser comparable con la del resto.

Pero acá no hace falta adivinar: sabemos de dónde salió cada archivo. Una foto de SUN397 ES
una fotografía de cámara (el dominio "foto" del proyecto), y una captura de wave-ui ES un
render. Este script escribe esa procedencia en `_procedencia.json` y `dominio.py` la consulta
como señal de máxima confianza — evidencia declarada en la obtención, que es más fuerte que
cualquier heurístico sobre el nombre. Ver el encabezado de `dominio.py`.

  Consecuencia buscada: los 150 negativos difíciles quedan como RENDER y por lo tanto (a) no
  entran nunca en el test, y (b) reciben `augmentar(..., "render")` a fuerza 1.0, o sea la
  simulación completa de foto de pantalla. Entrenan como "captura de una pantalla mostrando
  algo que no es una diapositiva", que es exactamente el negativo difícil que se busca.

TRANSPORTE: la API pública de Hugging Face (datasets-server)
------------------------------------------------------------
    https://datasets-server.huggingface.co/rows?dataset=...&config=...&split=...
Devuelve JSON con una URL por imagen. No hace falta instalar `datasets`, ni `pyarrow`, ni
bajar los zips completos (SUN397 pesa 37 GB; acá se bajan 160 imágenes). Solo urllib + PIL,
que es la misma regla de dependencias que el resto de `documentacion/`.

El muestreo es DETERMINISTA: para cada fuente se recorren páginas repartidas uniformemente a
lo largo del split y dentro de cada página se toman filas equiespaciadas. Con la misma semilla
salen las mismas filas, y quedan anotadas en el manifiesto (dataset + índice de fila), así que
la obtención es auditable y repetible aunque las URLs firmadas de la API caduquen.

LICENCIAS: las cuatro fuentes son datasets académicos de uso libre para investigación y sus
licencias quedan registradas por imagen en el manifiesto. Las imágenes NO se versionan en git
(`dataset/` está en .gitignore), igual que el resto del dataset del proyecto.

Uso:
    python documentacion/negativos_v10.py                 # baja las 450
    python documentacion/negativos_v10.py --total 900     # otra cantidad (mantiene proporciones)
    python documentacion/negativos_v10.py --mostrar       # inventario de lo ya bajado
    python documentacion/negativos_v10.py --limpiar       # borra la clase entera
"""

import argparse
import io
import json
import random
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "dataset"
CLASE = "3_no_diapositiva"
DESTINO = DATA_DIR / CLASE
MANIFIESTO = DESTINO / "_procedencia.json"

API = "https://datasets-server.huggingface.co/rows"
SEED = 123          # la semilla del proyecto desde la v1
TOTAL = 450         # ver el encabezado: un cuarto exacto del dataset resultante
LADO_MAX = 1024     # se guardan a 1024 px de lado mayor; el caché las baja a 224 igual
CALIDAD = 92
TIMEOUT = 45
REINTENTOS = 4
PAUSA_PAGINA = 0.5   # freno entre páginas para no agotar la cuota de la API pública

FOTO, RENDER = "foto", "render"


# --- 1. Las cuatro familias y sus fuentes ---
# `peso` es la fracción del total que le toca a la familia; `dominio` es la procedencia
# declarada que va al manifiesto y que después lee dominio.py.
FAMILIAS = {
    "escena": {
        "peso": 160 / TOTAL,
        "dominio": FOTO,
        "que_es": "habitaciones, oficinas, aulas, calles: el encuadre accidental",
        "fuentes": [
            {"dataset": "clip-benchmark/wds_sun397", "config": "default", "split": "test",
             "columna": "webp", "licencia": "SUN397 (Xiao et al. 2010) — uso académico"},
        ],
    },
    "objeto": {
        "peso": 140 / TOTAL,
        "dominio": FOTO,
        "que_es": "personas, objetos, escritorios, comida: el otro encuadre accidental",
        "fuentes": [
            {"dataset": "sayakpaul/coco-30-val-2014", "config": "default", "split": "train",
             "columna": "image", "licencia": "COCO 2014 (Lin et al.) — CC BY 4.0"},
        ],
    },
    "interfaz": {
        "peso": 100 / TOTAL,
        "dominio": RENDER,
        "que_es": "NEGATIVO DIFÍCIL: una pantalla que no muestra una diapositiva",
        "fuentes": [
            {"dataset": "agentsea/wave-ui", "config": "default", "split": "train",
             "columna": "image", "licencia": "wave-ui — Apache 2.0", "cuota": 0.6},
            {"dataset": "Zexanima/website_screenshots_image_dataset", "config": "default",
             "split": "train", "columna": "image",
             "licencia": "website-screenshots (Roboflow) — CC BY 4.0", "cuota": 0.4},
        ],
    },
    "documento": {
        "peso": 50 / TOTAL,
        "dominio": RENDER,
        "que_es": "NEGATIVO DIFÍCIL: texto sobre fondo blanco, lo más parecido a la clase 0",
        "fuentes": [
            # DocVQA va primero porque tiene 1000 filas y variedad real (informes, tablas,
            # cartas, formularios). FUNSD tiene 149 y sirve de complemento, no de fuente
            # principal: pedirle 50 imágenes a un split de 149 fuerza al muestreo a
            # concentrarse y devuelve menos de las pedidas.
            {"dataset": "nielsr/docvqa_1200_examples", "config": "default", "split": "train",
             "columna": "image", "licencia": "DocVQA (Mathew et al. 2021) — uso académico",
             "cuota": 0.7},
            {"dataset": "nielsr/funsd-layoutlmv3", "config": "funsd", "split": "train",
             "columna": "image", "licencia": "FUNSD (Jaume et al. 2019) — uso académico"},
        ],
    },
}


# --- 2. Cliente de la API ---
def _espera_tras_error(e: Exception, intento: int, base: float) -> float:
    """Cuánto esperar antes de reintentar. El 429 se trata distinto que el resto.

    La API pública limita la tasa de requests y responde 429 cuando se la satura. Reintentar
    a los 2 segundos con 429 es tirar el reintento a la basura: hay que esperar de verdad.
    Si la respuesta trae Retry-After se respeta, y si no se hace backoff exponencial desde
    15 s. El resto de los errores (500 mientras la API arma su caché, timeouts) se reintentan
    rápido, que es lo que corresponde para un fallo transitorio de servidor.
    """
    if isinstance(e, urllib.error.HTTPError) and e.code == 429:
        try:
            return float(e.headers.get("Retry-After") or 0) or 15.0 * (2 ** intento)
        except (TypeError, ValueError):
            return 15.0 * (2 ** intento)
    return base * (intento + 1)


def _pedir_json(url: str) -> dict:
    """GET con reintentos. La API devuelve 500 mientras arma su caché y 429 si se la satura."""
    ultimo = None
    for intento in range(REINTENTOS):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                return json.loads(r.read())
        except Exception as e:                                    # noqa: BLE001
            ultimo = e
            time.sleep(_espera_tras_error(e, intento, 1.5))
    raise RuntimeError(f"{url[:110]}... -> {ultimo}")


def _pedir_bytes(url: str) -> bytes:
    ultimo = None
    for intento in range(REINTENTOS):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:                                    # noqa: BLE001
            ultimo = e
            time.sleep(_espera_tras_error(e, intento, 1.0))
    raise RuntimeError(f"descarga fallida: {ultimo}")


def filas(dataset: str, config: str, split: str, offset: int, largo: int) -> dict:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": largo})
    return _pedir_json(f"{API}?{q}")


def _url_de_celda(celda) -> str | None:
    """La API devuelve las imágenes como {'src': url, ...} y a veces como lista de eso."""
    if isinstance(celda, dict):
        return celda.get("src")
    if isinstance(celda, list) and celda:
        return _url_de_celda(celda[0])
    if isinstance(celda, str) and celda.startswith("http"):
        return celda
    return None


# --- 3. Muestreo determinista ---
def indices_muestreados(total_filas: int, n: int, semilla: int,
                        por_pagina: int = 50) -> list[int]:
    """n índices de fila repartidos por TODO el split, de forma determinista.

    Por qué no `rng.sample(range(total), n)` a secas: cada fila suelta costaría un request a
    la API (450 requests). Y por qué no tomar n filas consecutivas: los splits de estos
    datasets suelen venir ORDENADOS POR CLASE, así que 160 filas seguidas de SUN397 serían
    160 fotos de dos o tres categorías de escena — justo lo contrario de la variedad que la
    clase de rechazo necesita.

    La solución intermedia: se eligen páginas repartidas uniformemente a lo largo del split y
    dentro de cada página se toman filas equiespaciadas. Cubre el rango completo, cuesta
    `n_paginas` requests en vez de `n`, y con la misma semilla da exactamente las mismas filas.

    CASO BORDE (splits chicos): si se pide una fracción grande del split, el esquema de páginas
    se degrada — las páginas se solapan, los índices se repiten y el `set` final devuelve
    MENOS filas de las pedidas, en silencio. Cuando n cubre más del 30% del split se usa
    directamente un paso constante sobre el rango completo, que en ese régimen es lo correcto:
    cubre todo, no repite y devuelve exactamente n.
    """
    n = min(n, total_filas)
    if n >= 0.3 * total_filas:
        paso = total_filas / n
        arranque = random.Random(semilla).random() * paso
        return sorted({min(total_filas - 1, int(arranque + i * paso)) for i in range(n)})

    n_paginas = max(1, min(24, (n + 3) // 4))
    por_pag = [n // n_paginas + (1 if i < n % n_paginas else 0) for i in range(n_paginas)]

    rng = random.Random(semilla)
    jitter = rng.random()          # corre el arranque para que dos fuentes no coincidan
    indices: list[int] = []
    for p, cuantas in enumerate(por_pag):
        if cuantas == 0:
            continue
        base = int((p + jitter) * total_filas / n_paginas) % total_filas
        base = min(base, max(0, total_filas - por_pagina))
        ancho = min(por_pagina, total_filas - base)
        paso = max(1, ancho // cuantas)
        indices += [base + (k * paso) % ancho for k in range(cuantas)]
    return sorted(set(indices))[:n]


# --- 4. Descarga de una fuente ---
def bajar_fuente(fuente: dict, familia: str, dominio: str, n: int, semilla: int,
                 destino: Path, ya: dict) -> list[dict]:
    """Baja hasta n imágenes de una fuente. Devuelve los registros del manifiesto."""
    ds, cfg, split = fuente["dataset"], fuente["config"], fuente["split"]
    col = fuente["columna"]
    apodo = ds.split("/")[-1].replace("wds_", "").replace("_image_dataset", "")

    cabecera = filas(ds, cfg, split, 0, 1)
    total_filas = int(cabecera.get("num_rows_total") or 0)
    if not total_filas:
        raise RuntimeError(f"{ds}: el split '{split}' no declara filas")
    print(f"    {ds}  ({total_filas} filas) -> {n} imágenes")

    objetivo = indices_muestreados(total_filas, n, semilla)
    registros: list[dict] = []
    pendientes = list(objetivo)

    while pendientes and len(registros) < n:
        offset = pendientes[0]
        largo = 50
        lote = filas(ds, cfg, split, offset, largo)
        disponibles = {offset + i: fila["row"] for i, fila in enumerate(lote.get("rows", []))}
        consumidos = []

        for idx in pendientes:
            if idx not in disponibles or len(registros) >= n:
                continue
            consumidos.append(idx)
            nombre = f"neg_{familia}_{apodo}_{idx:06d}.jpg"
            reg = {"archivo": nombre, "familia": familia, "dominio": dominio,
                   "dataset": ds, "config": cfg, "split": split, "fila": idx,
                   "licencia": fuente["licencia"]}

            if nombre in ya:                       # ya estaba bajada: reanudable
                registros.append(reg)
                continue

            url = _url_de_celda(disponibles[idx].get(col))
            if not url:
                continue
            try:
                crudo = _pedir_bytes(url)
                guardar_jpeg(crudo, destino / nombre)
            except Exception as e:                 # noqa: BLE001
                print(f"      [salteada] fila {idx}: {e}")
                continue
            registros.append(reg)
            if len(registros) % 25 == 0:
                print(f"      {len(registros)}/{n}")

        pendientes = [i for i in pendientes if i not in consumidos and i > offset]
        if not consumidos:                          # la página no aportó nada: evitar bucle
            pendientes = pendientes[1:]
        # Freno entre páginas. La API pública limita la tasa y responde 429 cuando se la
        # satura; medio segundo por página cuesta ~10 s en toda la descarga y evita quedarse
        # sin cuota a la mitad, que es mucho más caro (backoff de 15 s por reintento).
        time.sleep(PAUSA_PAGINA)

    return registros


def guardar_jpeg(crudo: bytes, destino: Path) -> None:
    """Normaliza a JPEG RGB con lado mayor <= LADO_MAX.

    Se re-codifica en vez de guardar los bytes tal cual por tres razones concretas:
      - unifica el formato (llegan PNG con alfa, WEBP y JPEG; el caché de imagenes.py
        haría convert('RGB') igual, pero acá además se evita guardar PNG de 3 MB);
      - acota el tamaño en disco: 450 imágenes a 1024 px pesan ~60 MB en vez de ~400 MB;
      - deja las imágenes con la misma cadena de compresión que el resto del dataset, así
        la clase de rechazo no se distingue de las otras tres por su firma JPEG — que sería
        exactamente el atajo que la v9 se ocupó de eliminar con la augmentation.
    """
    with Image.open(io.BytesIO(crudo)) as im:
        im = im.convert("RGB")
        if max(im.size) > LADO_MAX:
            escala = LADO_MAX / max(im.size)
            im = im.resize((max(1, round(im.width * escala)), max(1, round(im.height * escala))),
                           Image.LANCZOS)
        im.save(destino, format="JPEG", quality=CALIDAD, optimize=True)


# --- 5. Manifiesto ---
def guardar_manifiesto(registros: list[dict], total_pedido: int) -> None:
    por_dominio = {FOTO: 0, RENDER: 0}
    por_familia: dict[str, int] = {}
    for r in registros:
        por_dominio[r["dominio"]] += 1
        por_familia[r["familia"]] = por_familia.get(r["familia"], 0) + 1

    with open(MANIFIESTO, "w", encoding="utf-8") as f:
        json.dump({
            "clase": CLASE,
            "version": "v10",
            "proposito": "clase de rechazo: lo que NO es una diapositiva",
            "generado_por": "documentacion/negativos_v10.py",
            "semilla": SEED,
            "total_pedido": total_pedido,
            "total_obtenido": len(registros),
            "por_familia": por_familia,
            "por_dominio": por_dominio,
            "familias": {k: {"que_es": v["que_es"], "dominio": v["dominio"]}
                         for k, v in FAMILIAS.items()},
            "imagenes": sorted(registros, key=lambda r: r["archivo"]),
        }, f, indent=2, ensure_ascii=False)


def leer_manifiesto() -> dict | None:
    if not MANIFIESTO.is_file():
        return None
    with open(MANIFIESTO, encoding="utf-8") as f:
        return json.load(f)


def mostrar() -> None:
    man = leer_manifiesto()
    if not man:
        raise SystemExit(f"No existe {MANIFIESTO.relative_to(RAIZ)}. Corré el script sin flags.")
    reales = sorted(p.name for p in DESTINO.iterdir()
                    if p.is_file() and p.suffix.lower() == ".jpg")
    print("=" * 74)
    print(f"  CLASE DE RECHAZO — {CLASE}")
    print("=" * 74)
    print(f"  archivos en disco : {len(reales)}")
    print(f"  en el manifiesto  : {man['total_obtenido']} / {man['total_pedido']} pedidos")
    print(f"\n  {'familia':<12}{'n':>6}{'dominio':>10}   qué es")
    print("  " + "-" * 70)
    for fam, n in man["por_familia"].items():
        info = man["familias"][fam]
        print(f"  {fam:<12}{n:>6}{info['dominio']:>10}   {info['que_es'][:44]}")
    print("  " + "-" * 70)
    print(f"  {'TOTAL':<12}{man['total_obtenido']:>6}"
          f"   ({man['por_dominio'][FOTO]} fotos + {man['por_dominio'][RENDER]} renders)")
    print("\n  Las FOTOS pueden entrar al test (son fotografías de cámara reales).")
    print("  Los RENDERS solo entrenan, y la augmentation los convierte en 'captura de")
    print("  pantalla fotografiada': el negativo difícil que el producto necesita.")

    fuentes = {}
    for img in man["imagenes"]:
        fuentes.setdefault(img["dataset"], {"n": 0, "lic": img["licencia"]})["n"] += 1
    print(f"\n  {'fuente':<48}{'n':>5}   licencia")
    print("  " + "-" * 70)
    for ds, d in sorted(fuentes.items()):
        print(f"  {ds:<48}{d['n']:>5}   {d['lic']}")


# --- 6. Main ---
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baja la clase de rechazo 3_no_diapositiva desde datasets públicos.")
    ap.add_argument("--total", type=int, default=TOTAL,
                    help=f"cuántas imágenes en total (default {TOTAL})")
    ap.add_argument("--semilla", type=int, default=SEED, help=f"semilla (default {SEED})")
    ap.add_argument("--mostrar", action="store_true", help="inventario de lo ya bajado")
    ap.add_argument("--limpiar", action="store_true", help="borra la clase entera y sale")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.limpiar:
        if DESTINO.is_dir():
            shutil.rmtree(DESTINO)
            print(f"borrado {DESTINO.relative_to(RAIZ)}")
        else:
            print("no había nada que borrar")
        return

    if args.mostrar:
        mostrar()
        return

    if not DATA_DIR.is_dir():
        raise SystemExit(f"No existe la carpeta de datos: {DATA_DIR}")
    DESTINO.mkdir(parents=True, exist_ok=True)

    ya = {p.name: p for p in DESTINO.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"}
    if ya:
        print(f"Ya hay {len(ya)} imágenes en {DESTINO.name}/: se reanuda sin volver a bajarlas.\n")

    print("=" * 74)
    print(f"  OBTENIENDO LA CLASE DE RECHAZO — {args.total} imágenes")
    print("=" * 74)

    registros: list[dict] = []
    t0 = time.time()
    for i, (familia, cfg) in enumerate(FAMILIAS.items()):
        n_familia = round(args.total * cfg["peso"])
        print(f"\n  [{familia}]  {n_familia} imgs · dominio declarado: {cfg['dominio']}")
        print(f"    {cfg['que_es']}")
        restante = n_familia
        for j, fuente in enumerate(cfg["fuentes"]):
            cuota = fuente.get("cuota")
            n_fuente = restante if cuota is None or j == len(cfg["fuentes"]) - 1 \
                else round(n_familia * cuota)
            n_fuente = min(n_fuente, restante)
            if n_fuente <= 0:
                continue
            try:
                nuevos = bajar_fuente(fuente, familia, cfg["dominio"], n_fuente,
                                      args.semilla + 17 * i + j, DESTINO, ya)
            except Exception as e:                                # noqa: BLE001
                print(f"    [FUENTE CAÍDA] {fuente['dataset']}: {e}")
                continue
            registros += nuevos
            restante -= len(nuevos)
        if restante > 0:
            print(f"    !! faltaron {restante} imágenes de esta familia")

    # El manifiesto ACUMULA entre corridas, no se reemplaza.
    #
    # Es el detalle que hace que reanudar funcione de verdad. Si una fuente se cae a mitad de
    # camino (la API pública responde 429 cuando se la satura), esta corrida devuelve solo lo
    # suyo; sobrescribir el manifiesto con eso dejaría sin procedencia declarada a todo lo que
    # ya estaba bien bajado. Y como el paso siguiente borra lo que no figura en el manifiesto,
    # un error transitorio de red terminaría BORRANDO datos buenos.
    #
    # Se conservan las entradas previas cuyo archivo sigue en disco, y las nuevas pisan a las
    # viejas si coinciden en nombre (misma fuente, misma fila: es la misma imagen).
    en_disco = {p.name for p in DESTINO.iterdir()
                if p.is_file() and p.suffix.lower() == ".jpg"}
    fusionados: dict[str, dict] = {}
    previo = leer_manifiesto()
    if previo:
        for r in previo.get("imagenes", []):
            if r.get("archivo") in en_disco:
                fusionados[r["archivo"]] = r
    for r in registros:
        fusionados[r["archivo"]] = r
    todos = list(fusionados.values())

    guardar_manifiesto(todos, args.total)
    print(f"\n  esta corrida: {len(registros)} imágenes en {(time.time() - t0) / 60:.1f} min")
    print(f"  acumulado   : {len(todos)}/{args.total}")
    print(f"  guardado {MANIFIESTO.relative_to(RAIZ)}")

    # INVARIANTE: el manifiesto y el disco dicen lo mismo.
    #
    # Un .jpg sobrante de una corrida con otra semilla u otro --total SIGUE SIENDO una imagen
    # de la clase para particion_v10.py, pero NO figura en el manifiesto — así que dominio.py
    # no encuentra su procedencia declarada y cae al heurístico, que la marca RENDER. Serían
    # fotografías reales entrando como si fueran capturas, con el dominio mal etiquetado y sin
    # que nada avise. Se borran, ahora que el manifiesto ya está fusionado y no puede
    # confundir "no lo bajé en esta corrida" con "no le conozco la procedencia".
    huerfanos = [p for p in DESTINO.iterdir()
                 if p.is_file() and p.suffix.lower() == ".jpg" and p.name not in fusionados]
    for p in huerfanos:
        p.unlink()
    if huerfanos:
        print(f"  borrados {len(huerfanos)} archivos sin procedencia declarada")
    print()
    mostrar()

    if len(registros) < args.total:
        print(f"\n  AVISO: faltaron {args.total - len(registros)}. Volvé a correr el script:")
        print("         reanuda sin re-bajar lo que ya está.")
    print("\n  Siguiente paso:")
    print("    python documentacion/verificar_dataset.py")
    print("    python documentacion/particion_v10.py")


if __name__ == "__main__":
    main()
