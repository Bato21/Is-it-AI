"""
traer_negativos.py — Adquisición del conjunto NEGATIVO para la puerta de la v10.

QUÉ PROBLEMA RESUELVE
    Hasta la v9 el modelo asume que TODA imagen que recibe es una diapositiva: el softmax
    de 3 clases reparte probabilidad entre "sin IA / rastro / saturada" aunque le pasen la
    foto de un perro. En la app eso es un bug de producto: el usuario apunta a cualquier
    cosa y recibe un veredicto con aire de certeza.

    La v10 agrega una PUERTA binaria ("¿esto es una diapositiva?") delante del eje ordinal.
    Para entrenarla hace falta un conjunto negativo, y este script lo construye.

LAS DOS FAMILIAS DE NEGATIVOS, Y POR QUÉ HACEN FALTA LAS DOS
    faciles/    Fotografías de escenas cotidianas (personas, objetos, comida, calles).
                Son el caso mayoritario en la app: el usuario apunta a su escritorio, a
                una pared, a una persona. Ya son fotos de cámara real, así que entran al
                test de fotos SIN simulación: su dominio es exactamente el de despliegue.
                Fuente: COCO val2017 (CC BY 4.0) — 5000 imágenes, se muestrean N con semilla.

    dificiles/  Páginas de documento. Son lo ÚNICO que se parece de verdad a una
                diapositiva; sin ellas la puerta aprende "documento vs mundo real", que es
                trivial, y falla justo en el caso que importa. Dos maquetas a propósito:
                papers de arXiv (dos columnas, texto denso) y páginas de revista
                (multi-columna a color, fotos grandes, titulares) — estas últimas son las
                más caras, porque son casi una diapositiva.
                Fuente: Internet Archive, renderizadas con pdftoppm.

EL RIESGO QUE ESTE DISEÑO EVITA (y que ya mordió al proyecto una vez)
    aug_pantalla.py existe porque el moiré corría el riesgo de correlacionar con la clase.
    Acá pasa lo mismo un nivel más arriba: si TODOS los negativos fueran renders y todos
    los positivos fotos, la puerta aprendería "render vs foto" en vez de "documento vs
    diapositiva" — un atajo perfecto que daría 99% en validación y 0% de utilidad real.

    Por eso los fáciles son fotos reales (mismo dominio que los positivos) y los difíciles,
    que son renders, reciben el mismo tratamiento que los 300 renders del proyecto: viven
    en entrenamiento y pasan por aug_pantalla.py. Ver documentacion/particion_v10.py.

USO
    python documentacion/traer_negativos.py                    # todo, con los N por defecto
    python documentacion/traer_negativos.py --fuente coco      # solo los fáciles
    python documentacion/traer_negativos.py --fuente docs      # solo los difíciles
    python documentacion/traer_negativos.py --n-faciles 300 --n-dificiles 300

Es idempotente: si la carpeta destino ya tiene la cantidad pedida, no vuelve a bajar nada.
Sin dependencias del proyecto: urllib + PIL + pdftoppm (todos ya presentes).
"""

import argparse
import io
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
DESTINO = ROOT / "dataset_negativos"
FACILES = DESTINO / "faciles"
DIFICILES = DESTINO / "dificiles"

COCO_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
IA_BUSQUEDA = "https://archive.org/advancedsearch.php"
IA_METADATA = "https://archive.org/metadata"
IA_DESCARGA = "https://archive.org/download"

# Dos colecciones de Internet Archive, elegidas por MAQUETA y no por tema, porque la
# maqueta es lo que la puerta mira:
#
#   arxiv          papers en LaTeX: dos columnas, texto denso, fórmulas, figuras chicas
#                  en gris. Es el documento "serio", el opuesto visual de una diapositiva.
#   magazine_rack  páginas de revista: multi-columna a color, fotos grandes, titulares
#                  tipográficos. Es lo que MÁS se parece a una diapositiva sin serlo, así
#                  que son los negativos más caros y los que de verdad entrenan la puerta.
#
# Se usa Internet Archive y no la API de arXiv porque esta última tiene un límite por IP
# agresivo (429 tras pocas consultas). IA espeja el mismo material sin ese problema.
COLECCIONES = {"arxiv": 0.5, "magazine_rack": 0.5}     # colección -> proporción del total

MAX_PDF_MB = 30          # las revistas escaneadas llegan a 200 MB; no aportan más por eso
PAGINAS_POR_PDF = 4      # pocas por documento: variedad de documentos > páginas del mismo

LADO_MAX = 1024      # se reescala para que el conjunto no pese GB; 1024 > 224 con margen
CALIDAD_JPEG = 90
SEMILLA = 123        # la del proyecto, para que el muestreo sea reproducible


def log(msg: str) -> None:
    print(msg, flush=True)


def ya_estan(carpeta: Path, n: int) -> bool:
    if not carpeta.exists():
        return False
    tiene = len(list(carpeta.glob("*.jpg")))
    if tiene >= n:
        log(f"  {carpeta.name}/ ya tiene {tiene} imágenes (>= {n}); no se baja nada.")
        return True
    return False


def guardar_reescalada(img: Image.Image, destino: Path) -> None:
    """Convierte a RGB, limita el lado mayor y guarda JPEG. Mantiene el conjunto liviano."""
    img = img.convert("RGB")
    if max(img.size) > LADO_MAX:
        escala = LADO_MAX / max(img.size)
        img = img.resize((round(img.width * escala), round(img.height * escala)),
                         Image.LANCZOS)
    img.save(destino, "JPEG", quality=CALIDAD_JPEG)


# --------------------------------------------------------------------------------------
# FÁCILES — COCO val2017
# --------------------------------------------------------------------------------------
def traer_coco(n: int) -> None:
    """Baja val2017.zip una vez y extrae n imágenes muestreadas con semilla fija.

    Se baja el zip completo (~780 MB) en vez de pedir imágenes sueltas porque COCO no
    publica un índice liviano de nombres: sin las anotaciones (241 MB) no se puede saber
    qué IDs existen. Un zip conocido, con licencia declarada y muestreo determinista es
    más defendible que raspar URLs sueltas.
    """
    if ya_estan(FACILES, n):
        return
    FACILES.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_local = Path(tmp) / "val2017.zip"
        log(f"  bajando COCO val2017 (~780 MB) desde {COCO_ZIP} ...")
        with urllib.request.urlopen(COCO_ZIP, timeout=120) as r, open(zip_local, "wb") as f:
            total, leido, ultimo = int(r.headers.get("Content-Length", 0)), 0, -1
            while chunk := r.read(1 << 20):
                f.write(chunk)
                leido += len(chunk)
                if total:
                    pct = leido * 100 // total
                    if pct >= ultimo + 10:
                        ultimo = pct
                        log(f"    {pct}%  ({leido // (1 << 20)} MB)")
        log("  descarga lista; muestreando ...")

        with zipfile.ZipFile(zip_local) as z:
            nombres = sorted(x for x in z.namelist() if x.lower().endswith(".jpg"))
            log(f"    el zip trae {len(nombres)} imágenes")
            random.Random(SEMILLA).shuffle(nombres)
            guardadas = 0
            for nombre in nombres:
                if guardadas >= n:
                    break
                try:
                    with z.open(nombre) as fh:
                        img = Image.open(io.BytesIO(fh.read()))
                    guardar_reescalada(img, FACILES / f"coco_{Path(nombre).stem}.jpg")
                    guardadas += 1
                except Exception as exc:      # una imagen corrupta no puede cortar la corrida
                    log(f"    (salteada {nombre}: {exc})")
    log(f"  faciles/: {guardadas} imágenes de COCO val2017")


# --------------------------------------------------------------------------------------
# DIFÍCILES — páginas de papers de arXiv
# --------------------------------------------------------------------------------------
def json_de(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Is-it-AI/1.0 (proyecto academico)"}),
        timeout=timeout,
    ) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ids_coleccion(coleccion: str, cuantos: int) -> list[str]:
    """Identificadores de Internet Archive de una colección, con PDF de texto disponible.

    Se pide una ventana MÁS GRANDE que la necesaria y se muestrea localmente con la semilla
    del proyecto: así el conjunto es reproducible sin depender de que IA ordene igual en
    cada corrida.
    """
    consulta = urllib.parse.quote(f'collection:{coleccion} AND format:"Text PDF"')
    url = (f"{IA_BUSQUEDA}?q={consulta}&fl%5B%5D=identifier"
           f"&rows={cuantos * 4}&output=json")
    try:
        d = json_de(url)
        ids = [x["identifier"] for x in d["response"]["docs"]]
        random.Random(SEMILLA).shuffle(ids)
        log(f"    {coleccion}: {d['response']['numFound']} items, se toman {len(ids)} candidatos")
        return ids
    except Exception as exc:
        log(f"    {coleccion}: sin resultados ({exc})")
        return []


def url_pdf(identificador: str) -> tuple[str, float] | None:
    """Ubica el PDF del item vía la API de metadata y devuelve (url, tamaño_MB).

    No se adivina el nombre del archivo: en la colección arxiv el item es 'arxiv-0710.5767'
    pero el PDF se llama '0710.5767.pdf', y en magazine_rack no hay patrón. Una consulta de
    metadata por item es barata y evita bajar páginas de error de 146 bytes.
    """
    try:
        d = json_de(f"{IA_METADATA}/{identificador}", timeout=40)
    except Exception:
        return None
    for f in d.get("files", []):
        nombre = f.get("name", "")
        if nombre.lower().endswith(".pdf"):
            try:
                mb = int(f.get("size") or 0) / (1 << 20)
            except (TypeError, ValueError):
                mb = 0.0
            if 0 < mb <= MAX_PDF_MB:
                return f"{IA_DESCARGA}/{identificador}/{urllib.parse.quote(nombre)}", mb
    return None


def traer_documentos(n: int) -> None:
    """Baja PDFs de Internet Archive y renderiza sus primeras páginas como negativos duros."""
    if ya_estan(DIFICILES, n):
        return
    DIFICILES.mkdir(parents=True, exist_ok=True)

    if not shutil.which("pdftoppm"):
        log("  ERROR: falta pdftoppm (paquete poppler-utils). Se omiten los difíciles.")
        return

    guardadas = len(list(DIFICILES.glob("*.jpg")))
    cupos = {c: round(n * prop) for c, prop in COLECCIONES.items()}
    log(f"  cupo por colección: {cupos}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for coleccion, cupo in cupos.items():
            objetivo = min(n, guardadas + cupo)
            log(f"  --- {coleccion}: hasta {objetivo - guardadas} páginas ---")
            ids = ids_coleccion(coleccion, cupo // PAGINAS_POR_PDF + 8)

            for ident in ids:
                if guardadas >= objetivo:
                    break
                enlace = url_pdf(ident)
                if enlace is None:
                    continue
                url, mb = enlace
                pdf = tmpd / "actual.pdf"
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Is-it-AI/1.0 (proyecto academico)"})
                    with urllib.request.urlopen(req, timeout=180) as r, open(pdf, "wb") as f:
                        shutil.copyfileobj(r, f)
                except Exception as exc:
                    log(f"    ({ident}: no se pudo bajar — {exc})")
                    continue

                base = tmpd / "pag"
                try:
                    subprocess.run(
                        ["pdftoppm", "-jpeg", "-r", "150", "-f", "1", "-l",
                         str(PAGINAS_POR_PDF), str(pdf), str(base)],
                        check=True, capture_output=True, timeout=180,
                    )
                except Exception as exc:
                    log(f"    ({ident}: no se pudo renderizar — {exc})")
                    pdf.unlink(missing_ok=True)
                    continue

                seguro = re.sub(r"[^A-Za-z0-9_.-]", "_", ident)[:60]
                for pag in sorted(tmpd.glob("pag-*.jpg")):
                    if guardadas < objetivo:
                        try:
                            guardar_reescalada(
                                Image.open(pag),
                                DIFICILES / f"{coleccion}_{seguro}_{pag.stem.split('-')[-1]}.jpg")
                            guardadas += 1
                        except Exception:
                            pass
                    pag.unlink(missing_ok=True)
                pdf.unlink(missing_ok=True)

                if guardadas % 40 < PAGINAS_POR_PDF:
                    log(f"    {guardadas}/{n} páginas")
                time.sleep(0.5)     # cortesía con Internet Archive
    log(f"  dificiles/: {guardadas} páginas de documento")


def escribir_procedencia(n_f: int, n_d: int) -> None:
    (DESTINO / "PROCEDENCIA.md").write_text(f"""# Procedencia del conjunto negativo (v10)

Generado por `documentacion/traer_negativos.py` (semilla {SEMILLA}).

| Carpeta | N | Fuente | Licencia | Dominio |
|---|---|---|---|---|
| `faciles/` | {n_f} | COCO val2017 (`{COCO_ZIP}`) | Imágenes CC BY 4.0; anotaciones CC BY 4.0 | **foto de cámara real** |
| `dificiles/` | {n_d} | Internet Archive (colecciones `arxiv` y `magazine_rack`), renderizados a 150 dpi con `pdftoppm` | Acceso abierto arXiv (licencia por paper) | **render** |

## Por qué dos familias

`faciles/` es el caso mayoritario de la app: el usuario apunta el teléfono a algo que no es
una diapositiva. Son fotografías reales, así que su dominio ya coincide con el de
despliegue y pueden entrar al test de fotos sin simulación.

`dificiles/` es el caso que decide la calidad de la puerta: páginas de documento con texto
denso, figuras y fondo claro, que es lo único visualmente cercano a una diapositiva. Al ser
renders reciben el mismo trato que los 300 renders del proyecto — entrenamiento +
`aug_pantalla.py` — para que la puerta no pueda resolver el problema por textura.

## Reproducir

```bash
python documentacion/traer_negativos.py
```

Es idempotente y el muestreo es determinista con la semilla {SEMILLA}.
""", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", choices=["todo", "coco", "docs"], default="todo")
    ap.add_argument("--n-faciles", type=int, default=650)
    ap.add_argument("--n-dificiles", type=int, default=650)
    args = ap.parse_args()

    DESTINO.mkdir(parents=True, exist_ok=True)
    log(f"Destino: {DESTINO}")

    if args.fuente in ("todo", "coco"):
        log("\n[1/2] FÁCILES — fotos de escenas reales (COCO val2017)")
        traer_coco(args.n_faciles)
    if args.fuente in ("todo", "docs"):
        log("\n[2/2] DIFÍCILES — páginas de documento (Internet Archive)")
        traer_documentos(args.n_dificiles)

    n_f = len(list(FACILES.glob("*.jpg"))) if FACILES.exists() else 0
    n_d = len(list(DIFICILES.glob("*.jpg"))) if DIFICILES.exists() else 0
    escribir_procedencia(n_f, n_d)

    log(f"\nLISTO  faciles={n_f}  dificiles={n_d}  total={n_f + n_d}")
    log(f"Procedencia y licencias en {DESTINO / 'PROCEDENCIA.md'}")


if __name__ == "__main__":
    main()
