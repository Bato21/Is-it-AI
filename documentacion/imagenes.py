"""
Carga y caché de imágenes — COMPARTIDA por TensorFlow, PyTorch y el tercer modelo.

Por qué existe (el costo que elimina):
  La v9 entrena TRES modelos sobre el MISMO dataset de 1337 imágenes. Decodificar JPEG y
  redimensionar a 224x224 cuesta ~3.5 minutos por pasada completa. Sin caché eso se paga
  una vez por script y por semilla: TensorFlow lo paga, PyTorch lo vuelve a pagar, el
  modelo C lo paga de nuevo — media hora de CPU quemada en decodificar los mismos JPEG.

  Con caché se paga UNA vez y los tres scripts leen un .npy de 200 MB en dos segundos.

Por qué además MEJORA la comparación (no es solo velocidad):
  Los tres modelos leen EXACTAMENTE el mismo array de píxeles. Sin esto, TensorFlow
  decodificaría con tf.io.decode_image y PyTorch con PIL, y las dos librerías difieren en
  el algoritmo de redimensionado (antialias, half-pixel centers). Esas diferencias de
  fracciones de píxel se propagan por la red y contaminan la comparación TF vs PyTorch:
  parte de la diferencia entre frameworks sería de la librería de imagen, no del modelo.
  Fijando el array de entrada, la única variable que queda es el framework. Es la misma
  lógica que llevó a compartir particion_datos.py y metricas.py en la v7.

Formato del caché:
  dataset_cache/imgs_224.npy    uint8 (N, 224, 224, 3) RGB, 0-255
  dataset_cache/indice.json     {"rutas": [...], "huella": "...", "tamano": 224}

  El orden del array sigue el de "rutas", así que buscar una imagen es indexar un dict.
  La huella del dataset se guarda adentro: si alguien agrega imágenes, el caché se
  invalida solo y se regenera (mismo guard que particion_v9.py).

Redimensionado: bilineal CON antialias, aplastando el aspecto a cuadrado.
  Aplastar (en vez de recortar) es la convención del proyecto desde la v1, y es la correcta
  acá: una diapositiva 16:9 recortada a cuadrado pierde los bordes laterales, que es
  justamente donde suelen estar las marcas de agua y los artefactos de generación.
  La app Ionic hace lo MISMO en el navegador (ver InferenceService), así que la paridad
  entrenamiento/producción se mantiene.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "dataset"
CACHE_DIR = RAIZ / "dataset_cache"
TAMANO = 224


def _rutas_cache(tamano: int) -> tuple[Path, Path]:
    return CACHE_DIR / f"imgs_{tamano}.npy", CACHE_DIR / f"indice_{tamano}.json"


def cargar_una(ruta: Path, tamano: int = TAMANO) -> np.ndarray:
    """Una imagen -> uint8 (tamano, tamano, 3) RGB. La función de referencia del proyecto.

    convert("RGB") normaliza de una los tres casos que trae el dataset: JPEG ya RGB, PNG
    con canal alfa (RGBA) y PNG en escala de grises. Sin eso, las 20 imágenes PNG del
    dataset entrarían con 1 o 4 canales y romperían el stack.

    draft() es el truco que hace tolerable construir el caché: las fotos del dataset son de
    4000x3000 (12 Mpx) y decodificarlas enteras para tirar el 99.7% de los píxeles cuesta
    100 ms cada una. draft() le pide al decodificador JPEG que salte directo a una escala
    DCT reducida (1/8 acá, o sea 500x375), que sigue siendo mayor que los 224 finales.
    Baja a 31 ms/imagen — de 2.2 minutos de caché a 45 segundos — sin pérdida visible,
    porque el remuestreo final se hace igual desde una resolución superior a la de destino.
    En PNG es un no-op: el formato no tiene escalas intermedias.

    El resize usa BILINEAR, que en Pillow aplica el filtro escalado al factor de reducción
    (o sea, hace antialiasing real al achicar). Sin eso, bajar 18x produciría aliasing
    severo justo sobre el texto de las diapositivas, que es la señal más informativa.
    """
    with Image.open(ruta) as im:
        im.draft("RGB", (tamano, tamano))
        return np.asarray(im.convert("RGB").resize((tamano, tamano), Image.BILINEAR))


def construir_cache(rutas_relativas: list[str], huella: str, tamano: int = TAMANO,
                    data_dir: Path = DATA_DIR, verbose: bool = True) -> np.ndarray:
    """Decodifica todas las imágenes y las guarda en el caché. Devuelve el array."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy, idx = _rutas_cache(tamano)
    n = len(rutas_relativas)
    if verbose:
        print(f"  construyendo caché de {n} imágenes a {tamano}x{tamano} "
              f"(~{n * tamano * tamano * 3 / 1e6:.0f} MB)...")

    arr = np.zeros((n, tamano, tamano, 3), dtype=np.uint8)
    for i, rel in enumerate(rutas_relativas):
        arr[i] = cargar_una(data_dir / rel, tamano)
        if verbose and (i + 1) % 250 == 0:
            print(f"    {i + 1}/{n}")

    np.save(npy, arr)
    with open(idx, "w", encoding="utf-8") as f:
        json.dump({"rutas": rutas_relativas, "huella": huella, "tamano": tamano}, f)
    if verbose:
        print(f"  guardado {npy.name}")
    return arr


def cargar_cache(rutas_relativas: list[str], huella: str, tamano: int = TAMANO,
                 data_dir: Path = DATA_DIR, verbose: bool = True
                 ) -> tuple[np.ndarray, dict[str, int]]:
    """Devuelve (array de imágenes, {ruta_relativa: índice}). Reconstruye si hace falta.

    El caché se considera válido solo si la huella del dataset coincide Y la lista de rutas
    es idéntica. Cualquier diferencia lo regenera: es preferible perder 3 minutos a entrenar
    sobre un caché viejo, que es un error silencioso e imposible de diagnosticar después.
    """
    npy, idx = _rutas_cache(tamano)
    arr = None
    if npy.is_file() and idx.is_file():
        with open(idx, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("huella") == huella and meta.get("rutas") == rutas_relativas:
            arr = np.load(npy, mmap_mode="r")
            if verbose:
                print(f"  caché válido: {npy.name} ({arr.shape})")
        elif verbose:
            print("  caché obsoleto (cambió el dataset): se regenera")
    if arr is None:
        arr = construir_cache(rutas_relativas, huella, tamano, data_dir, verbose)
    return arr, {r: i for i, r in enumerate(rutas_relativas)}


def cargar_para_particion(particion: dict, tamano: int = TAMANO, verbose: bool = True
                          ) -> tuple[np.ndarray, dict[str, int]]:
    """Atajo: arma el caché con TODAS las imágenes que menciona una partición."""
    rutas = sorted(particion["dominios"].keys())
    return cargar_cache(rutas, particion["meta"]["huella_dataset"], tamano, verbose=verbose)


# --- Generación de vistas aumentadas (compartida por los tres modelos) ---
def generar_vistas(imgs: np.ndarray, indice: dict[str, int], rutas: list[str],
                   etiquetas: list[int], dominios: list[str], n_aug: int, semilla: int,
                   verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 vista limpia + n_aug vistas aumentadas por imagen. Devuelve (X uint8, Y, dominio).

    Las vistas se generan UNA vez con semilla fija en lugar de re-sortearse por época.
    Es el mismo tradeoff consciente que tomó la v8: se pierde algo de diversidad de
    augmentation a cambio de que (a) el experimento sea 100% reproducible, (b) se puedan
    pre-computar los embeddings de la fase 1 y (c) los dos frameworks vean EXACTAMENTE las
    mismas vistas, que es lo que hace que la comparación sea un espejo real.

    Las vistas de una misma imagen quedan CONSECUTIVAS: la vista limpia de la imagen i está
    en la posición i*(n_aug+1). Los scripts usan esa propiedad para excluir de validación
    todas las vistas de una imagen a la vez y evitar fuga entre vistas hermanas.
    """
    from aug_pantalla import augmentar

    rng = np.random.default_rng(semilla)
    por_img = n_aug + 1
    total = len(rutas) * por_img
    X = np.zeros((total, imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
    Y = np.zeros(total, dtype=np.int64)
    D = np.empty(total, dtype=object)

    k = 0
    for ruta, etiqueta, dom in zip(rutas, etiquetas, dominios):
        base = imgs[indice[ruta]]
        for v in range(por_img):
            X[k] = base if v == 0 else augmentar(base, dom, rng)
            Y[k] = etiqueta
            D[k] = dom
            k += 1
        if verbose and k % 600 == 0:
            print(f"    vistas {k}/{total}")
    return X, Y, D
