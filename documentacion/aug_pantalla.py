"""
Simulación de FOTOGRAFÍA DE PANTALLA — augmentation de dominio, compartida TF/PyTorch.

El problema que resuelve (SUGERENCIAS_V9.md §1 y §4):

  El dataset v9 tiene 1037 fotos de pantalla y 300 renders limpios. Las fotos ya cubren la
  brecha de dominio de forma NATURAL, que es la mejor manera de cubrirla. Pero quedan dos
  agujeros que la captura no puede tapar sola:

  AGUJERO 1 — los renders arrastran un atajo.
    Los 300 renders viven solo en entrenamiento. Si la red descubre que "textura de render"
    y "textura de foto" son separables (y lo son: el moiré es una firma clarísima), puede
    usar ESE canal como atajo en vez de mirar el contenido de la diapositiva. Cuando el
    reparto de renders por clase no es idéntico — acá es 108/94/98, casi parejo pero no
    exacto — el atajo tiene señal aprovechable. Aplicando la simulación a los renders, la
    red deja de poder distinguirlos y se ve OBLIGADA a mirar el contenido.

  AGUJERO 2 — las fotos cubren las pantallas que había a mano, no todas.
    Cada panel genera su propio patrón de moiré, cada ambiente su propio glare. Simular
    parámetros que no se fotografiaron (otras frecuencias de rejilla, otros ángulos, otras
    temperaturas de color) es más barato que comprar monitores.

  Y encima está el riesgo explícito de §4: que la red vea moiré en una diapositiva humana
  limpia (0_sin_ia), lea ese ruido como "artefacto de IA" y la mande a 2_saturada_ia. La
  defensa es que el moiré aparezca en las TRES clases por igual durante el entrenamiento,
  de modo que deje de correlacionar con la etiqueta. Eso es exactamente lo que hace aplicar
  esta simulación con la misma probabilidad en las tres clases.

Los seis efectos, y qué artefacto físico imita cada uno:

  perspectiva   fotografiar desde un costado o desde abajo -> el rectángulo se vuelve trapecio
  moire         batido entre la rejilla de subpíxeles del monitor y la del sensor de la cámara
  glare         una luz del techo o una ventana reflejada en el panel + viñeteo del lente
  temperatura   el balance de blancos automático tira a cálido (tungsteno) o frío (LED)
  ruido         ruido de sensor con poca luz (ISO alto)
  jpeg          la compresión con pérdida que aplica la cámara, encima de la del render

Implementación: numpy + PIL, SIN OpenCV. Los dos venvs del proyecto tienen Pillow pero
NINGUNO tiene cv2 instalado (opencv-python figura en los requirements para los scripts de
cámara de escritorio, pero no está en el entorno), así que depender de cv2 rompería el
script en la máquina donde se entrena. PIL alcanza: Image.transform(PERSPECTIVE) hace el
warp y el resto es aritmética de numpy.

Contrato de la API: todo entra y sale como np.ndarray uint8 (H, W, 3) en RGB, 0-255.
Ese es el formato común entre tf.image y torchvision, así que el MISMO código produce
exactamente las mismas vistas en los dos frameworks — que es la condición para que la
comparación TF vs PyTorch de la v9 siga siendo un espejo y no dos experimentos distintos.
"""

import io

import numpy as np
from PIL import Image

# --- 1. Parámetros por defecto de cada efecto ---
# Rangos elegidos mirando las fotos reales del dataset: la simulación tiene que quedar
# DENTRO de lo que produce un celular, no exagerar. Un augmentation más agresivo que la
# realidad enseña a la red a resistir cosas que nunca va a ver, y le roba capacidad.
CFG = {
    "p_perspectiva": 0.7, "perspectiva_max": 0.06,   # 6% del lado, ~ fotografiar a 15-20°
    "p_moire": 0.6, "moire_amp": (0.03, 0.12), "moire_periodo": (6.0, 40.0),
    "p_glare": 0.5, "glare_intensidad": (0.08, 0.24), "glare_radio": (0.20, 0.50),
    "p_vineta": 0.5, "vineta_fuerza": (0.05, 0.22),
    "p_temperatura": 0.6, "temperatura_max": 0.10,
    "p_ruido": 0.6, "ruido_sigma": (2.0, 9.0),
    "p_jpeg": 0.8, "jpeg_calidad": (55, 92),
    "p_desenfoque": 0.25, "desenfoque_radio": (0.4, 1.1),
}


# --- 2. Perspectiva ---
def _coeficientes_perspectiva(origen: np.ndarray, destino: np.ndarray) -> tuple:
    """Resuelve los 8 coeficientes que PIL necesita para Image.transform(PERSPECTIVE).

    PIL mapea el destino HACIA el origen (transformación inversa), y pide (a..h) tales que
        x_orig = (a·x + b·y + c) / (g·x + h·y + 1)
        y_orig = (d·x + e·y + f) / (g·x + h·y + 1)
    Cada par de puntos aporta 2 ecuaciones; con 4 esquinas quedan 8 ecuaciones y 8
    incógnitas, que se resuelven por mínimos cuadrados (lstsq es estable si el sistema
    queda casi singular, cosa que pasa cuando el warp es muy chico).
    """
    matriz = []
    for (xd, yd), (xo, yo) in zip(destino, origen):
        matriz.append([xd, yd, 1, 0, 0, 0, -xo * xd, -xo * yd])
        matriz.append([0, 0, 0, xd, yd, 1, -yo * xd, -yo * yd])
    A = np.array(matriz, dtype=np.float64)
    b = np.array(origen, dtype=np.float64).reshape(8)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return tuple(coef)


def perspectiva(img: np.ndarray, rng: np.random.Generator, maximo: float = None) -> np.ndarray:
    """Trapecio: mueve las 4 esquinas al azar hasta `maximo`·lado.

    Imita fotografiar la pantalla desde un costado, desde abajo o inclinando el teléfono.
    Los bordes que quedan vacíos se rellenan por reflexión del propio contenido (fillcolor
    negro dejaría una banda negra rectísima que la red aprendería a reconocer como "esta
    imagen fue aumentada" — otro atajo, justo el que estamos tratando de eliminar).
    """
    maximo = CFG["perspectiva_max"] if maximo is None else maximo
    h, w = img.shape[:2]
    dx, dy = maximo * w, maximo * h
    esquinas = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    desplazadas = esquinas + rng.uniform(-1, 1, size=(4, 2)) * np.array([dx, dy])

    pil = Image.fromarray(img)
    coef = _coeficientes_perspectiva(desplazadas, esquinas)
    warp = pil.transform((w, h), Image.PERSPECTIVE, coef, resample=Image.BILINEAR)
    out = np.asarray(warp).astype(np.uint8)

    # Zona inválida: se calcula warpeando una máscara de unos con la MISMA transformación,
    # en vez de buscar píxeles negros en la salida. Buscar negros falla dos veces: no detecta
    # el borde interpolado (bilineal deja píxeles oscuros pero no nulos) y sí detecta el
    # contenido genuinamente negro de la diapositiva. La máscara warpeada es exacta.
    mascara = Image.new("L", (w, h), 255).transform(
        (w, h), Image.PERSPECTIVE, coef, resample=Image.BILINEAR)
    invalida = np.asarray(mascara) < 250
    if invalida.any():
        # Se rellena con la imagen original desplazada (espejo horizontal). Rellenar con
        # negro dejaría cuñas rectas de color plano que la red aprende a reconocer como
        # "esta imagen fue aumentada": otro atajo, justo el que la v9 quiere eliminar.
        espejo = np.asarray(pil.transpose(Image.FLIP_LEFT_RIGHT))
        out = np.where(invalida[..., None], espejo, out).astype(np.uint8)
    return out


# --- 3. Moiré ---
def moire(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Patrón de interferencia: modula el brillo con una sinusoide 2D de baja frecuencia.

    El moiré real nace del ALIASING entre dos rejillas casi iguales (los subpíxeles del
    monitor y los fotositos del sensor). La diferencia de frecuencias produce un batido de
    frecuencia MUCHO más baja que las dos originales — por eso se ve como bandas anchas y
    no como un patrón fino. Se modela directamente ese batido:

        factor(x, y) = 1 + A·sin(2π·(x·cosθ + y·senθ)/T + φ)

    con periodo T entre 6 y 40 px y amplitud A entre 3% y 12%. Se aplica a los tres canales
    con un desfase leve entre ellos, porque los subpíxeles RGB del panel están en posiciones
    distintas y el moiré real sale coloreado, no gris.
    """
    h, w = img.shape[:2]
    amp = rng.uniform(*CFG["moire_amp"])
    periodo = rng.uniform(*CFG["moire_periodo"])
    theta = rng.uniform(0, np.pi)
    fase = rng.uniform(0, 2 * np.pi)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proy = xx * np.cos(theta) + yy * np.sin(theta)

    salida = img.astype(np.float32)
    for canal in range(3):
        desfase = fase + canal * rng.uniform(0.0, 0.9)   # subpíxeles RGB desplazados
        salida[..., canal] *= 1.0 + amp * np.sin(2 * np.pi * proy / periodo + desfase)
    return np.clip(salida, 0, 255).astype(np.uint8)


# --- 4. Glare y viñeteo ---
def glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reflejo especular: suma una mancha gaussiana clara en una posición al azar.

    Es aditivo y no multiplicativo a propósito: un reflejo LAVA la imagen (sube el negro,
    baja el contraste local) en vez de escalarla. Ese lavado es justamente lo que arruina
    la lectura del texto en las fotos reales con la luz mal puesta.
    """
    h, w = img.shape[:2]
    intensidad = rng.uniform(*CFG["glare_intensidad"]) * 255.0
    radio = rng.uniform(*CFG["glare_radio"]) * max(h, w)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mancha = intensidad * np.exp(-d2 / (2 * radio ** 2))
    return np.clip(img.astype(np.float32) + mancha[..., None], 0, 255).astype(np.uint8)


def vineta(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Viñeteo del lente: oscurece hacia las esquinas. Presente en casi toda cámara de celular."""
    h, w = img.shape[:2]
    fuerza = rng.uniform(*CFG["vineta_fuerza"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    factor = 1.0 - fuerza * np.clip(r, 0, 1.5) ** 2
    return np.clip(img.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)


# --- 5. Temperatura de color ---
def temperatura_color(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Balance de blancos: escala R y B en sentidos opuestos.

    Positivo = cálido (tungsteno, sube R baja B). Negativo = frío (LED/fluorescente).
    Es el artefacto MÁS peligroso para este proyecto en particular: la clase 2_saturada_ia
    se caracteriza por colores saturados, y un balance de blancos agresivo puede fabricar
    saturación donde no la había. Por eso el rango es conservador (±10%) y por eso se aplica
    por igual a las tres clases.
    """
    maximo = CFG["temperatura_max"]
    t = rng.uniform(-maximo, maximo)
    escala = np.array([1.0 + t, 1.0, 1.0 - t], dtype=np.float32)
    return np.clip(img.astype(np.float32) * escala, 0, 255).astype(np.uint8)


# --- 6. Ruido, desenfoque y JPEG ---
def ruido_gauss(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ruido de sensor. sigma 2-9 sobre 255 = lo que produce un celular con poca luz."""
    sigma = rng.uniform(*CFG["ruido_sigma"])
    ruido = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def desenfoque(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Desenfoque suave: el autofoco del celular no siempre acierta.

    OJO con la interacción de diseño: la app Ionic RECHAZA las fotos borrosas con el filtro
    de varianza del Laplaciano (SUGERENCIAS §5.2), así que el modelo no debería necesitar
    resistir desenfoques fuertes. El radio se mantiene bajo (≤1.1 px) justamente para cubrir
    solo la franja que el filtro deja pasar.
    """
    from PIL import ImageFilter

    radio = rng.uniform(*CFG["desenfoque_radio"])
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radio)))


def jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Recompresión JPEG: bloques de 8x8 y ringing alrededor del texto.

    Es el artefacto más específico de este dataset: las diapositivas son texto sobre fondo
    plano, o sea el peor caso para JPEG, y el ringing alrededor de las letras es una firma
    fortísima. Un modelo que nunca lo vio se desconcierta con la primera foto real.
    """
    calidad = int(rng.integers(*CFG["jpeg_calidad"]))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=calidad)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


# --- 7. El pipeline completo ---
def simular_foto_pantalla(img: np.ndarray, rng: np.random.Generator,
                          fuerza: float = 1.0) -> np.ndarray:
    """Aplica la cadena completa de efectos, cada uno con su probabilidad.

    El ORDEN imita la cadena física real y no es intercambiable:
        geometría (perspectiva)  ->  óptica (glare, viñeta)  ->  panel (moiré)
        ->  sensor (temperatura, ruido)  ->  archivo (desenfoque, JPEG)
    Comprimir a JPEG antes de agregar el ruido, por ejemplo, produciría un ruido limpio sin
    bloques: físicamente imposible, y la red aprendería a detectar esa imposibilidad.

    `fuerza` escala todas las PROBABILIDADES (no las magnitudes): con 0.5 se aplica la mitad
    de los efectos por imagen, que es lo que se usa sobre las fotos reales — ya tienen sus
    artefactos propios y encimarles otra tanda completa las saca del dominio real.
    """
    out = img
    if rng.random() < CFG["p_perspectiva"] * fuerza:
        out = perspectiva(out, rng)
    if rng.random() < CFG["p_glare"] * fuerza:
        out = glare(out, rng)
    if rng.random() < CFG["p_vineta"] * fuerza:
        out = vineta(out, rng)
    if rng.random() < CFG["p_moire"] * fuerza:
        out = moire(out, rng)
    if rng.random() < CFG["p_temperatura"] * fuerza:
        out = temperatura_color(out, rng)
    if rng.random() < CFG["p_ruido"] * fuerza:
        out = ruido_gauss(out, rng)
    if rng.random() < CFG["p_desenfoque"] * fuerza:
        out = desenfoque(out, rng)
    if rng.random() < CFG["p_jpeg"] * fuerza:
        out = jpeg(out, rng)
    return out


# --- 8. Augmentation fotométrica clásica (la de v2-v8, en numpy) ---
def aug_clasica(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotación pequeña + zoom + brillo + contraste: la augmentation de siempre del proyecto.

    Se mantiene porque ataca otra cosa que la simulación de pantalla: la variación de
    ENCUADRE y EXPOSICIÓN, no la de dominio. Las dos se aplican juntas en entrenamiento.
    """
    h, w = img.shape[:2]
    pil = Image.fromarray(img)

    if rng.random() < 0.7:                                    # rotación ±5°
        ang = rng.uniform(-5, 5)
        pil = pil.rotate(ang, resample=Image.BILINEAR, expand=False)
        # Rotar con expand=False rellena las esquinas de NEGRO. Esas cuñas negras rectas
        # son un artefacto puramente sintético: la red las aprende como marca de "imagen
        # aumentada" y el augmentation termina enseñando lo contrario de lo que busca.
        # Se eliminan recortando el rectángulo inscrito, que para un lado L y un ángulo θ
        # mide L/(cos|θ| + sen|θ|), y devolviendo al tamaño original.
        rad = abs(np.deg2rad(ang))
        inscrito = 1.0 / (np.cos(rad) + np.sin(rad))
        cw, ch = int(w * inscrito), int(h * inscrito)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        pil = pil.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.7:                                    # zoom in hasta 10%
        z = rng.uniform(1.0, 1.10)
        nw, nh = max(1, int(w / z)), max(1, int(h / z))
        x0 = int(rng.uniform(0, w - nw)) if w > nw else 0
        y0 = int(rng.uniform(0, h - nh)) if h > nh else 0
        pil = pil.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.BILINEAR)

    out = np.asarray(pil).astype(np.float32)
    if rng.random() < 0.7:                                    # brillo ±20%
        out *= rng.uniform(0.8, 1.2)
    if rng.random() < 0.7:                                    # contraste ±20%
        media = out.mean()
        out = media + (out - media) * rng.uniform(0.8, 1.2)
    return np.clip(out, 0, 255).astype(np.uint8)


def augmentar(img: np.ndarray, dominio: str, rng: np.random.Generator) -> np.ndarray:
    """Augmentation COMPLETA para entrenamiento, ajustada según el dominio de la imagen.

    La asimetría es deliberada y es el corazón de la estrategia:

      render -> fuerza 1.0. Hay que convertirlo en algo indistinguible de una foto, si no
                la red lo usa de atajo (AGUJERO 1 del encabezado).
      foto   -> fuerza 0.45. Ya trae sus artefactos reales; encimarle otra tanda completa
                la empuja fuera del dominio que la app va a ver y perjudica más que ayuda.
    """
    out = aug_clasica(img, rng)
    fuerza = 1.0 if dominio == "render" else 0.45
    return simular_foto_pantalla(out, rng, fuerza=fuerza)
