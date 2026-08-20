"""
Detección de DOMINIO de cada imagen del dataset — COMPARTIDA por TensorFlow y PyTorch.

Qué es el "dominio" y por qué aparece recién en la v9:
  Hasta la v8 el dataset era homogéneo: todas las imágenes eran RENDERS limpios, o sea
  páginas de un PDF/PPT exportadas a JPG/PNG por software. Pixel perfecto, sin ruido de
  cámara, sin perspectiva, sin reflejos.

  La v9 cambia el destino del modelo: deja de vivir en un notebook y pasa a vivir en la
  cámara de un celular dentro de una app Ionic. Ahí la entrada YA NO ES un render: es una
  FOTOGRAFÍA de una pantalla que muestra la diapositiva. Entre esas dos cosas hay una
  brecha de dominio (domain gap) enorme:

      render                          foto de pantalla
      ------                          ----------------
      píxeles exactos del software    remuestreo cámara -> moiré (interferencia entre
                                      la grilla de píxeles del monitor y la del sensor)
      iluminación uniforme            glare, sombras, viñeteo
      encuadre perfecto               perspectiva trapezoidal, marco del monitor visible
      sin compresión adicional        JPEG de la cámara + balance de blancos + ruido ISO

  Un modelo entrenado SOLO con renders y evaluado SOLO con renders reporta una accuracy
  que no significa nada comercialmente: es la accuracy de un producto que nadie va a usar.
  Esto es exactamente el punto 2 de SUGERENCIAS_V9.md ("split de validación honesto").

Para poder medir eso hace falta saber, imagen por imagen, si es foto o render. Este módulo
es ese clasificador — y a propósito NO es un modelo de ML, es un conjunto de reglas
verificables. Un heurístico que se puede auditar leyendo 30 líneas vale más que una red
que acierta el 99% y no se puede explicar en la interrogación oral.

Las señales, en orden de confianza:

  0. PROCEDENCIA DECLARADA (v10) -> lo que diga el manifiesto. Cuando una imagen se obtuvo
     con un script del proyecto, su origen no hay que inferirlo: está registrado. Es la
     señal más fuerte de todas porque no es evidencia indirecta, es el acta de obtención.
     Ver el bloque "El manifiesto de procedencia" más abajo.
  1. EXIF con marca/modelo de cámara  -> FOTO (evidencia dura: el archivo declara que salió
     de un sensor). Es la señal más fuerte de las inferidas.
  2. Nombre de archivo con patrón de cámara -> FOTO. Las galerías de Android/iOS nombran
     por timestamp (20260815_143534.jpg, IMG_20260815_143534.jpg, PXL_20260815_143534.jpg).
     Sirve cuando el EXIF se perdió (típico al pasar fotos por WhatsApp/Telegram, que las
     recomprime y les borra los metadatos).
  3. Nombre con patrón de exportación de deck -> RENDER (..._page-0001.jpg es la salida de
     pdftoppm / Adobe / LibreOffice).

  Si ninguna dispara, se asume RENDER: es el caso conservador. Etiquetar una foto como
  render mete una foto en el conjunto de entrenamiento (pérdida menor); etiquetar un render
  como foto lo metería en el TEST de fotos y CONTAMINARÍA la métrica comercial, que es
  justamente el número que la v9 vino a hacer creíble.

EL MANIFIESTO DE PROCEDENCIA (agregado en la v10)
-------------------------------------------------
La v10 sumó la clase de rechazo `3_no_diapositiva`, que no se fotografió: se bajó de datasets
públicos con `documentacion/negativos_v10.py`. Esas imágenes rompen las tres señales de
arriba: los datasets académicos ya perdieron el EXIF y sus nombres no son de galería, así que
el heurístico las mandaría TODAS a RENDER por su caso conservador. La consecuencia sería
grave y silenciosa: la clase de rechazo quedaría sin fotos, su test se armaría con renders y
su accuracy dejaría de ser comparable con la de las otras tres.

Pero acá no hay nada que inferir. Una foto de SUN397 ES una fotografía de cámara; una captura
de pantalla de wave-ui ES un render. El script de obtención lo sabe y lo escribe en
`<clase>/_procedencia.json`. Este módulo lo lee y lo respeta por encima de todo lo demás.

  Formato mínimo esperado:
      {"imagenes": [{"archivo": "neg_escena_sun397_000413.jpg", "dominio": "foto"}, ...]}

  Si el archivo no existe, o una imagen no figura en él, se cae a las señales 1-3 de siempre:
  el manifiesto AGREGA información, nunca es un requisito. Las 1337 imágenes de las clases
  0-2 no tienen manifiesto y se siguen resolviendo exactamente igual que en la v9.

numpy/PIL/stdlib puro, sin torch ni tf, para poder importarse desde los dos venvs — misma
regla que particion_datos.py y metricas.py.
"""

import json
import re
from pathlib import Path

FOTO = "foto"
RENDER = "render"

# Nombre del manifiesto que deja el script de obtención dentro de la carpeta de una clase.
# Empieza con "_" para que ordene antes que las imágenes y se distinga de un dato a simple
# vista; y es .json, extensión que NO figura en EXTENSIONES de particion_v9/v10, así que los
# listadores de imágenes lo ignoran solos.
ARCHIVO_PROCEDENCIA = "_procedencia.json"

# --- 1. Patrones de nombre de archivo ---
# Galerías de cámara: 20260815_143534.jpg · IMG_20260815_143534.jpg · PXL_20260815_143534.jpg
# El grupo (?:...) inicial acepta el prefijo opcional; el cuerpo es AAAAMMDD_HHMMSS.
PATRON_CAMARA = re.compile(
    r"^(?:img_|pxl_|photo_|dsc_|dscn|p_)?\d{8}[_\-]?\d{6}", re.IGNORECASE
)
# Otros esquemas comunes de cámara/celular sin fecha: IMG_1234.JPG, DSC01234.JPG
PATRON_CAMARA_SECUENCIAL = re.compile(r"^(?:img|dsc|dscn|p)[_\-]?\d{4,5}$", re.IGNORECASE)

# Exportación de un deck a imágenes: "Charla_page-0001.jpg", "slide12.png", "diapositiva-3.png"
PATRON_RENDER = re.compile(
    r"(_page[-_]?\d+|[-_]slide[-_]?\d+|[-_]diapositiva[-_]?\d+)$", re.IGNORECASE
)

# Marcas de EXIF que NO son cámaras (editores que dejan Software pero no Make/Model real).
_MARCAS_NO_CAMARA = {"", "unknown", "n/a", "none"}

# Prefijos que agregan el explorador de archivos / la nube al duplicar un archivo.
# Sin quitarlos, "Copia de 20260815_124529.jpg" NO matchea el patrón de cámara y la foto
# se clasificaría como render. En este dataset eso pasaba en 326 archivos de la clase 0:
# es la diferencia entre tener test de fotos para las 3 clases o para ninguna.
PATRON_PREFIJO_COPIA = re.compile(
    r"^(?:copia de |copy of |copia \(\d+\) de |\(\d+\)\s*)+", re.IGNORECASE
)
# Sufijos de duplicado: "foto (1).jpg", "foto - copia.jpg"
PATRON_SUFIJO_COPIA = re.compile(r"(?:\s*\(\d+\)|\s*-\s*copia|\s*-\s*copy)+$", re.IGNORECASE)


def normalizar_tallo(tallo: str) -> str:
    """Quita prefijos/sufijos de duplicado para que los patrones vean el nombre original."""
    t = PATRON_PREFIJO_COPIA.sub("", tallo).strip()
    return PATRON_SUFIJO_COPIA.sub("", t).strip()


# --- 2. Lectura de EXIF ---
def _exif_de_camara(ruta: Path) -> bool:
    """True si el archivo trae EXIF con Make o Model de cámara reales.

    Se lee con PIL para no agregar dependencias (piexif/exifread). Los tags 271 (Make) y
    272 (Model) son los del estándar EXIF. Si el archivo no tiene EXIF, no es JPEG, o PIL
    falla al abrirlo, se devuelve False sin explotar: la decisión cae a las reglas de
    nombre, que siempre están disponibles.
    """
    try:
        from PIL import Image

        with Image.open(ruta) as im:
            exif = im.getexif()
            if not exif:
                return False
            make = str(exif.get(271, "")).strip().lower()
            model = str(exif.get(272, "")).strip().lower()
            return (make not in _MARCAS_NO_CAMARA) or (model not in _MARCAS_NO_CAMARA)
    except Exception:
        return False


# --- 2b. Procedencia declarada (señal 0, la de máxima confianza) ---
def cargar_procedencias(data_dir: Path) -> dict[str, str]:
    """{ruta_relativa: dominio} leído de los `_procedencia.json` de cada clase.

    Recorre las carpetas de clase y junta lo que declare cada manifiesto. Devuelve un dict
    vacío si no hay ninguno, que es el caso de las clases 0-2 (fotografiadas y clasificadas a
    mano, sin script de obtención de por medio).

    Un manifiesto ilegible se ignora en silencio A PROPÓSITO: la alternativa —abortar— haría
    que un JSON a medio escribir dejara el proyecto entero sin poder entrenar, cuando la
    degradación correcta es volver a las señales heurísticas de siempre. Lo que sí se valida
    es el VALOR: solo se aceptan "foto" y "render", porque un dominio inventado se propagaría
    a la partición y rompería la política de test sin que nada avise.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return {}
    fuera: dict[str, str] = {}
    for carpeta in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        manifiesto = carpeta / ARCHIVO_PROCEDENCIA
        if not manifiesto.is_file():
            continue
        try:
            with open(manifiesto, encoding="utf-8") as f:
                datos = json.load(f)
        except Exception:                                          # noqa: BLE001
            continue
        for img in datos.get("imagenes", []):
            nombre = img.get("archivo")
            dom = img.get("dominio")
            if nombre and dom in (FOTO, RENDER):
                fuera[f"{carpeta.name}/{nombre}"] = dom
    return fuera


# --- 3. La regla completa ---
def dominio_de(ruta: Path, usar_exif: bool = True) -> str:
    """Devuelve FOTO o RENDER para una imagen del dataset.

    `usar_exif=False` desactiva la lectura de metadatos y decide solo por nombre: es
    ~50x más rápido y se usa cuando hay que clasificar miles de archivos y ya se sabe
    (porque el audit lo confirmó) que los nombres alcanzan.
    """
    ruta = Path(ruta)
    tallo = normalizar_tallo(ruta.stem)

    if PATRON_RENDER.search(tallo):
        return RENDER
    if PATRON_CAMARA.match(tallo) or PATRON_CAMARA_SECUENCIAL.match(tallo):
        return FOTO
    if usar_exif and _exif_de_camara(ruta):
        return FOTO
    return RENDER


def mapa_dominios(data_dir: Path, rutas_relativas: list[str], usar_exif: bool = True,
                  usar_procedencia: bool = True) -> dict[str, str]:
    """{ruta_relativa: dominio} para una lista de rutas relativas a data_dir.

    La procedencia declarada (señal 0) gana sobre el heurístico; para todo lo que no figure
    en ningún manifiesto se aplican las señales 1-3 exactamente como en la v9.
    """
    data_dir = Path(data_dir)
    declarados = cargar_procedencias(data_dir) if usar_procedencia else {}
    return {r: declarados.get(r) or dominio_de(data_dir / r, usar_exif)
            for r in rutas_relativas}


def conteo_dominios(dominios: dict[str, str], clases: list[str]) -> dict[str, dict[str, int]]:
    """{clase: {'foto': n, 'render': m, 'total': n+m}} — el resumen que se imprime."""
    out = {c: {FOTO: 0, RENDER: 0, "total": 0} for c in clases}
    for ruta, dom in dominios.items():
        clase = ruta.split("/")[0]
        if clase in out:
            out[clase][dom] += 1
            out[clase]["total"] += 1
    return out
