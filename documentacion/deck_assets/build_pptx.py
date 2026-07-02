"""Construye documentacion/presentacion_is_it_ai.pptx.

Deck técnico (16:9, tema oscuro) del estado real del repo: v1 baseline + v2
data augmentation en TensorFlow y PyTorch. Números tomados de los Resultado_*
y del documento de avances; nada inventado.

Transiciones (fade) y animaciones de entrada (fade secuencial) se inyectan como
XML OOXML nativo de PowerPoint, porque python-pptx no las expone.

Corre con el venv de herramientas (tiene python-pptx):
  herramientas/.venv/Scripts/python.exe documentacion/deck_assets/build_pptx.py
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
SALIDA = RAIZ / "documentacion" / "presentacion_is_it_ai.pptx"

BG = RGBColor(0x0B, 0x0E, 0x14)
FG = RGBColor(0xE8, 0xEA, 0xF2)
MUT = RGBColor(0x8A, 0x91, 0xA6)
VIO = RGBColor(0x7C, 0x5C, 0xFF)
CYA = RGBColor(0x19, 0xD3, 0xC5)
TF_C = RGBColor(0xFF, 0x8A, 0x3D)
PT_C = RGBColor(0xEE, 0x4C, 0x2C)
ROJO = RGBColor(0xF8, 0x71, 0x71)
CARD = RGBColor(0x14, 0x18, 0x22)

FONT_H = "Segoe UI"          # títulos (bold)
FONT_B = "Segoe UI"          # cuerpo
ANCHO, ALTO = Inches(13.333), Inches(7.5)

NSMAP_P = "http://schemas.openxmlformats.org/presentationml/2006/main"


# ---------------------------------------------------------------- helpers ---
def fondo(slide) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG


def barra_gradiente(slide, y=Inches(1.18), x=Inches(0.75), w=Inches(2.6), h=Pt(4)):
    shp = slide.shapes.add_shape(1, x, y, w, h)  # 1 = rectangle
    shp.line.fill.background()
    shp.fill.gradient()
    stops = shp.fill.gradient_stops
    stops[0].color.rgb = VIO
    stops[0].position = 0.0
    stops[1].color.rgb = CYA
    stops[1].position = 1.0
    try:
        shp.fill.gradient_angle = 0
    except Exception:
        pass
    shp.shadow.inherit = False
    return shp


def texto(slide, x, y, w, h, contenido, size=18, color=FG, bold=False,
          align=PP_ALIGN.LEFT, font=FONT_B, line_spacing=1.15):
    """contenido: str o lista de (texto, dict-overrides)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    items = contenido if isinstance(contenido, list) else [(contenido, {})]
    for i, (txt, ov) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ov.get("align", align)
        p.line_spacing = ov.get("line_spacing", line_spacing)
        p.space_after = Pt(ov.get("space_after", 6))
        r = p.add_run()
        r.text = txt
        f = r.font
        f.name = ov.get("font", font)
        f.size = Pt(ov.get("size", size))
        f.bold = ov.get("bold", bold)
        f.color.rgb = ov.get("color", color)
        if ov.get("italic"):
            f.italic = True
    return tb


def vinetas(slide, x, y, w, h, items, size=17, gap=10):
    """Viñetas con guion largo y color de acento por ítem opcional."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        txt, kw = (it, {}) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.18
        p.space_after = Pt(gap)
        rb = p.add_run()
        rb.text = "—  "
        rb.font.color.rgb = kw.get("bullet", CYA)
        rb.font.size = Pt(kw.get("size", size))
        rb.font.bold = True
        r = p.add_run()
        r.text = txt
        r.font.name = FONT_B
        r.font.size = Pt(kw.get("size", size))
        r.font.color.rgb = kw.get("color", FG)
        r.font.bold = kw.get("bold", False)
    return tb


def titulo(slide, txt, kicker=None):
    if kicker:
        texto(slide, Inches(0.75), Inches(0.28), Inches(11.8), Inches(0.4),
              kicker.upper(), size=13, color=CYA, bold=True)
    texto(slide, Inches(0.72), Inches(0.55), Inches(11.9), Inches(0.75),
          txt, size=30, color=FG, bold=True, font=FONT_H)
    barra_gradiente(slide)


def pie(slide, n, total):
    texto(slide, Inches(0.75), Inches(7.06), Inches(6), Inches(0.35),
          "Is-it-AI · Frameworks de IA · UDD", size=10, color=MUT)
    texto(slide, Inches(12.1), Inches(7.06), Inches(0.9), Inches(0.35),
          f"{n:02d} / {total}", size=10, color=MUT, align=PP_ALIGN.RIGHT)


def imagen(slide, ruta, x, y, w=None, h=None, borde=True):
    pic = slide.shapes.add_picture(str(ruta), x, y, width=w, height=h)
    if borde:
        pic.line.color.rgb = RGBColor(0x2A, 0x30, 0x42)
        pic.line.width = Pt(1)
    return pic


def tarjeta(slide, x, y, w, h, color_borde=CYA):
    shp = slide.shapes.add_shape(5, x, y, w, h)  # 5 = rounded rectangle
    shp.fill.solid()
    shp.fill.fore_color.rgb = CARD
    shp.line.color.rgb = color_borde
    shp.line.width = Pt(1.25)
    shp.shadow.inherit = False
    return shp


def notas(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt


# ------------------------------------------------- transiciones/animación ---
def transicion_fade(slide, thru_black=False):
    el = slide._element
    tr = etree.SubElement(el, f"{{{NSMAP_P}}}transition")
    tr.set("spd", "med")
    fade = etree.SubElement(tr, f"{{{NSMAP_P}}}fade")
    if thru_black:
        fade.set("thruBlk", "1")


def _timing_xml(shape_ids, dur=700, stagger=350):
    """Timing OOXML: primer efecto al click, resto encadenado (after previous)."""
    P = f"{{{NSMAP_P}}}"
    cid = [7]  # ids 1-6 reservados abajo

    def nid():
        cid[0] += 1
        return str(cid[0])

    efectos = []
    delay = 0
    for i, spid in enumerate(shape_ids):
        node_type = "clickEffect" if i == 0 else "afterEffect"
        efectos.append(f"""
          <p:par>
            <p:cTn id="{nid()}" presetID="10" presetClass="entr" presetSubtype="0"
                   fill="hold" grpId="0" nodeType="{node_type}">
              <p:stCondLst><p:cond delay="{delay}"/></p:stCondLst>
              <p:childTnLst>
                <p:set>
                  <p:cBhvr>
                    <p:cTn id="{nid()}" dur="1" fill="hold">
                      <p:stCondLst><p:cond delay="0"/></p:stCondLst>
                    </p:cTn>
                    <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
                    <p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>
                  </p:cBhvr>
                  <p:to><p:strVal val="visible"/></p:to>
                </p:set>
                <p:animEffect transition="in" filter="fade">
                  <p:cBhvr>
                    <p:cTn id="{nid()}" dur="{dur}"/>
                    <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
                  </p:cBhvr>
                </p:animEffect>
              </p:childTnLst>
            </p:cTn>
          </p:par>"""
        )
        delay = stagger  # los siguientes parten tras el anterior

    xml = f"""<p:timing xmlns:p="{NSMAP_P}"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:tnLst>
    <p:par>
      <p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">
        <p:childTnLst>
          <p:seq concurrent="1" nextAc="seek">
            <p:cTn id="2" dur="indefinite" nodeType="mainSeq">
              <p:childTnLst>
                <p:par>
                  <p:cTn id="3" fill="hold">
                    <p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>
                    <p:childTnLst>
                      <p:par>
                        <p:cTn id="4" fill="hold">
                          <p:stCondLst><p:cond delay="0"/></p:stCondLst>
                          <p:childTnLst>{"".join(efectos)}
                          </p:childTnLst>
                        </p:cTn>
                      </p:par>
                    </p:childTnLst>
                  </p:cTn>
                </p:par>
              </p:childTnLst>
            </p:cTn>
            <p:prevCondLst>
              <p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>
            </p:prevCondLst>
            <p:nextCondLst>
              <p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>
            </p:nextCondLst>
          </p:seq>
        </p:childTnLst>
      </p:cTn>
    </p:par>
  </p:tnLst>
</p:timing>"""
    return etree.fromstring(xml.encode())


def animar(slide, shapes, dur=700, stagger=350):
    """Fade de entrada secuencial para las shapes dadas (1 click, luego solas)."""
    ids = [s.shape_id for s in shapes]
    slide._element.append(_timing_xml(ids, dur, stagger))


# ------------------------------------------------------------------ build ---
prs = Presentation()
prs.slide_width = ANCHO
prs.slide_height = ALTO
BLANK = prs.slide_layouts[6]
TOTAL = 14


def nueva():
    s = prs.slides.add_slide(BLANK)
    fondo(s)
    return s


# 1 — Portada (pantalla Stitch, recortada a 16:9)
s = nueva()
src = Image.open(AQUI / "stitch_portada.png")
w0, h0 = src.size
h_obj = int(w0 * 9 / 16)
top = (h0 - h_obj) // 2
src.crop((0, top, w0, top + h_obj)).save(AQUI / "_portada_169.png")
imagen(s, AQUI / "_portada_169.png", 0, 0, w=ANCHO, borde=False)
texto(s, Inches(0.75), Inches(6.95), Inches(8), Inches(0.4),
      "Frameworks de IA · UDD · Avance 1", size=12, color=MUT)
transicion_fade(s, thru_black=True)
notas(s, "Portada generada con Stitch (design system 'Is-it-AI Dark'). "
         "Detector de huella de IA en diapositivas; 3 niveles ordinales.")

# 2 — El problema
s = nueva()
titulo(s, "«¿Hecha con IA?» es la pregunta equivocada", kicker="El problema")
b = vinetas(s, Inches(0.75), Inches(1.7), Inches(7.3), Inches(4.4), [
    "El instinto inicial: clasificador binario IA / no-IA.",
    ("El profesor objetó — y la objeción reencuadró el proyecto.", {"bold": True}),
    "La procedencia es multidimensional (layout, texto, imágenes) y parcial: «esqueleto generado + texto humano» no cae limpio en un sí/no.",
    ("Una foto contiene el resultado visual, no el proceso.", {"color": CYA, "bold": True}),
], size=18)
t = tarjeta(s, Inches(8.5), Inches(2.2), Inches(4.0), Inches(3.0), VIO)
tx = texto(s, Inches(8.8), Inches(2.7), Inches(3.4), Inches(2.2), [
    ("Una CNN solo ve píxeles:", {"size": 16, "color": MUT}),
    ("solo puede aprender clases que se vean distintas.", {"size": 20, "bold": True}),
])
pie(s, 2, TOTAL)
transicion_fade(s)
animar(s, [b, t, tx])
notas(s, "Mensaje clave 1 (parte a): el binario es deshonesto con el enorme medio asistido. "
         "El punto decisivo es que la foto trae el resultado, no el proceso.")

# 3 — Qué medimos
s = nueva()
titulo(s, "No medimos procedencia. Medimos artefactos.", kicker="La decisión de diseño")
c = texto(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(1.5), [
    ("Densidad de artefactos visuales de IA", {"size": 40, "bold": True, "color": CYA, "align": PP_ALIGN.CENTER}),
    ("observables en la imagen — el label dice lo que el modelo realmente puede aprender", {"size": 18, "color": MUT, "align": PP_ALIGN.CENTER}),
])
b = vinetas(s, Inches(1.7), Inches(4.2), Inches(10), Inches(2.2), [
    "La huella de plantilla generada y las imágenes generadas dejan rastro visual.",
    "El origen del texto no deja rastro visual → queda explícitamente fuera del alcance.",
    ("Entendimos qué puede y qué no puede aprender el modelo antes de escribir código.", {"bold": True}),
], size=17)
pie(s, 3, TOTAL)
transicion_fade(s)
animar(s, [c, b])
notas(s, "Mensaje clave 1: no procedencia, densidad de artefactos. Es el argumento intelectual "
         "más fuerte de la presentación.")

# 4 — Tres niveles
s = nueva()
titulo(s, "Tres niveles — un eje ordinal", kicker="Las clases")
g = imagen(s, AQUI / "gradiente_clases.png", Inches(1.0), Inches(1.75), w=Inches(11.3), borde=False)
b = vinetas(s, Inches(0.9), Inches(5.35), Inches(11.6), Inches(1.5), [
    "Clase 1 definida por aspecto (densidad), no por proceso — si no, es un cajón de sastre inaprendible.",
    ("La frontera 1↔2 es el punto más frágil del esquema (protocolo de etiquetado pendiente).", {"bullet": ROJO}),
    "El softmax de 3 unidades entrega confianza por nivel, no un sí/no rígido.",
], size=16)
pie(s, 4, TOTAL)
transicion_fade(s)
animar(s, [g, b])
notas(s, "Clases nominales aunque el eje tiene orden: se acepta el costo (regresión ordinal sería "
         "over-engineering ahora). Por eso la matriz de confusión es el instrumento clave.")

# 5 — Dataset
s = nueva()
titulo(s, "De presentaciones a dataset etiquetado", kicker="Los datos")
g = imagen(s, AQUI / "pipeline_dataset.png", Inches(0.7), Inches(1.65), w=Inches(11.9), borde=False)
b = vinetas(s, Inches(0.9), Inches(4.6), Inches(11.5), Inches(2.1), [
    ("300 imágenes reales, balanceadas (~100 por clase) — primer lote con señal.", {"bold": True}),
    "Principio: invertir el problema — la procedencia entrega el label automáticamente (Zenodo pre-2022 → clase 0; generar con Gamma/Copilot/Canva → clase 2).",
    "Gotcha resuelto: Keras y PyTorch ordenan clases alfabéticamente → prefijos 0_/1_/2_ fijan el índice (verificado en cada corrida).",
    "La clase 2, la más cara de producir, fija el techo de tamaño de las otras dos.",
], size=15.5)
pie(s, 5, TOTAL)
transicion_fade(s)
animar(s, [g, b])
notas(s, "Render con LibreOffice headless (python-pptx solo extrae imágenes sin layout). "
         "JPEG ~q90 para match de dominio con fotos de teléfono.")

# 6 — Dos frameworks
s = nueva()
titulo(s, "El mismo experimento, dos frameworks", kicker="Requisito dual del curso")
g = imagen(s, AQUI / "frameworks.png", Inches(1.1), Inches(1.7), w=Inches(11.1), borde=False)
b = vinetas(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(1.3), [
    "Scripts espejo v1 y v2 en TensorFlow/ y PyTorch/ — mismos datos, arquitectura, épocas y figuras.",
    ("Resultados comparables entre frameworks: ese contraste es el objetivo del curso.", {"bold": True}),
], size=16)
pie(s, 6, TOTAL)
transicion_fade(s)
animar(s, [g, b])
notas(s, "Diferencia idiomática central: augmentation como capas del modelo (Keras, se apagan solas "
         "en inferencia) vs transform del DataLoader de train (PyTorch).")

# 7 — v1 baseline
s = nueva()
titulo(s, "v1 — una CNN que sobreajusta a propósito", kicker="El arco experimental · 1 de 2")
g = imagen(s, AQUI / "arquitectura_cnn.png", Inches(0.7), Inches(1.8), w=Inches(11.9), borde=False)
b = vinetas(s, Inches(0.9), Inches(4.7), Inches(11.5), Inches(1.9), [
    "Sin augmentation, sin dropout, sin regularización, sin transfer learning — diseñada para memorizar.",
    "El objetivo pedagógico: ver el sobreajuste primero, y que ese diagnóstico motive la v2.",
    ("Corrida placeholder (67 imgs): train y val al 100% → sin señal. Se pausó y se juntaron datos reales.", {"color": MUT}),
], size=16)
pie(s, 7, TOTAL)
transicion_fade(s)
animar(s, [g, b])
notas(s, "3.79M parámetros, 99% en la primera densa (Flatten 59.168 → 64). "
         "Misma arquitectura exacta en ambos frameworks (Keras y nn.Sequential).")

# 8 — v1 resultados
s = nueva()
titulo(s, "v1 — el sobreajuste está en la loss, no en la accuracy", kicker="El arco experimental · resultados v1")
g = imagen(s, AQUI / "overfit_loss.png", Inches(0.65), Inches(2.75), w=Inches(7.6), borde=False)
t = tarjeta(s, Inches(8.55), Inches(1.9), Inches(4.1), Inches(4.3), VIO)
tx = texto(s, Inches(8.85), Inches(2.1), Inches(3.5), Inches(4.0), [
    ("TensorFlow (300 imgs)", {"size": 15, "bold": True, "color": TF_C}),
    ("acc  1.000 → 0.967", {"size": 15, "font": "Consolas"}),
    ("loss 0.010 → 0.103  (~10×)", {"size": 15, "font": "Consolas", "color": ROJO}),
    ("PyTorch (300 imgs)", {"size": 15, "bold": True, "color": PT_C, "space_after": 2}),
    ("acc  0.996 → 0.833", {"size": 15, "font": "Consolas"}),
    ("loss 0.029 → 0.406", {"size": 15, "font": "Consolas", "color": ROJO}),
    ("El 96.7% de val es engañoso: es memorización de renders casi duplicados, no la métrica de despliegue.", {"size": 13.5, "color": MUT}),
])
pie(s, 8, TOTAL)
transicion_fade(s)
animar(s, [g, t, tx])
notas(s, "Las clases son muy separables en renders limpios → la accuracy casi no muestra brecha; "
         "la loss delata la memorización (0.01 vs 0.10). Figuras reales en TensorFlow/v1 y PyTorch/v1.")

# 9 — v2 augmentation
s = nueva()
titulo(s, "v2 — data augmentation domain-aware", kicker="El arco experimental · 2 de 2")
g = imagen(s, AQUI / "augmentation.png", Inches(0.7), Inches(1.7), w=Inches(11.9), borde=False)
b = vinetas(s, Inches(0.9), Inches(4.85), Inches(11.5), Inches(1.9), [
    ("Deliberadamente SIN flip horizontal: una diapo espejada tiene el texto en espejo — nunca ocurre.", {"bold": True}),
    "Lección: las transformaciones deben ser realistas para el dominio (copiar el flip de otro clasificador habría sido un error).",
    "Además se instrumenta: gráfico de loss (ahí estaba la señal) + matriz de confusión nativa (sin sklearn).",
], size=16)
pie(s, 9, TOTAL)
transicion_fade(s)
animar(s, [g, b])
notas(s, "Keras: RandomRotation(0.05), RandomZoom(0.1), RandomBrightness(0.2), RandomContrast(0.2) como capas. "
         "PyTorch: RandomAffine(18°, zoom) + ColorJitter en el transform del loader de train.")

# 10 — v2 resultados TF
s = nueva()
titulo(s, "v2 en TensorFlow — el overfitting se murió", kicker="Resultados v2 · TensorFlow")
g = imagen(s, RAIZ / "TensorFlow" / "v2" / "Figure_1.png", Inches(0.65), Inches(1.8), w=Inches(8.1))
t = tarjeta(s, Inches(9.05), Inches(1.9), Inches(3.6), Inches(4.1), TF_C)
tx = texto(s, Inches(9.3), Inches(2.1), Inches(3.1), Inches(3.8), [
    ("acc  0.771 → 0.750", {"size": 15, "font": "Consolas"}),
    ("loss 0.529 → 0.550", {"size": 15, "font": "Consolas", "color": CYA}),
    ("Loss train y val pegadas → firma de un modelo bien regularizado.", {"size": 14}),
    ("El 75% no es retroceso: es honestidad (el 96.7% era memorización).", {"size": 14}),
    ("Pero quedó sub-entrenado: las curvas seguían mejorando en la época 10.", {"size": 14, "color": ROJO}),
])
pie(s, 10, TOTAL)
transicion_fade(s)
animar(s, [g, t, tx])
notas(s, "Figura real: TensorFlow/v2/Figure_1.png (corrida de 10 épocas, 300 imágenes). "
         "Con augmentation cada época cuesta más; 10 no alcanzan.")

# 11 — v2 resultados PyTorch
s = nueva()
titulo(s, "v2 en PyTorch — la misma historia", kicker="Resultados v2 · PyTorch")
g = imagen(s, RAIZ / "PyTorch" / "v2" / "Figure_1.png", Inches(0.65), Inches(1.8), w=Inches(8.1))
t = tarjeta(s, Inches(9.05), Inches(1.9), Inches(3.6), Inches(4.1), PT_C)
tx = texto(s, Inches(9.3), Inches(2.1), Inches(3.1), Inches(3.8), [
    ("acc  0.708 → 0.717", {"size": 15, "font": "Consolas"}),
    ("loss 0.669 → 0.707", {"size": 15, "font": "Consolas", "color": CYA}),
    ("Sin brecha train/val — la augmentation regularizó igual que en TF.", {"size": 14}),
    ("Dos frameworks, mismos datos, misma conclusión: la lectura del experimento es robusta.", {"size": 14, "bold": True}),
])
pie(s, 11, TOTAL)
transicion_fade(s)
animar(s, [g, t, tx])
notas(s, "Figura real: PyTorch/v2/Figure_1.png. Val 71.7% vs TF 75% — mismo orden; "
         "las diferencias vienen del split y de detalles de implementación de las transforms.")

# 12 — Matrices de confusión
s = nueva()
titulo(s, "La matriz de confusión es el instrumento", kicker="Evaluación")
lt = texto(s, Inches(1.0), Inches(1.55), Inches(5.6), Inches(0.4),
           [("TensorFlow (val 24/18/18)", {"size": 14, "bold": True, "color": TF_C})])
g1 = imagen(s, RAIZ / "TensorFlow" / "v2" / "Figure_2_matriz.png", Inches(1.0), Inches(1.95), h=Inches(3.55))
lt2 = texto(s, Inches(7.0), Inches(1.55), Inches(5.6), Inches(0.4),
            [("PyTorch (val 60)", {"size": 14, "bold": True, "color": PT_C})])
g2 = imagen(s, RAIZ / "PyTorch" / "v2" / "Figure_2_matriz.png", Inches(7.0), Inches(1.95), h=Inches(3.55))
b = vinetas(s, Inches(0.9), Inches(5.75), Inches(11.7), Inches(1.4), [
    "La mayoría del error está entre vecinos (0↔1, 1↔2): esperable y barato en un eje ordinal.",
    ("Punto de atención: 3 diapos «saturada» → «sin rastro» en TF (1 en PyTorch). Confundir extremos es grave.", {"bullet": ROJO, "bold": True}),
    ("Salvedad: matrices de un modelo sub-entrenado → puede ser ruido de no-convergencia. Se decide re-entrenando.", {"color": MUT}),
], size=15)
pie(s, 12, TOTAL)
transicion_fade(s)
animar(s, [lt, g1, lt2, g2, b])
notas(s, "Mensaje clave 3: la accuracy esconde QUÉ se confunde. Si el error 0↔2 persiste tras "
         "converger, apunta a etiquetado (frontera 1↔2), no a arquitectura.")

# 13 — Diagnóstico + limitaciones
s = nueva()
titulo(s, "Leer el estado, no amontonar técnicas", kicker="Diagnóstico y pendientes honestos")
t = tarjeta(s, Inches(0.75), Inches(1.7), Inches(5.7), Inches(2.5), CYA)
tx = texto(s, Inches(1.05), Inches(1.9), Inches(5.1), Inches(2.1), [
    ("La lección de método", {"size": 16, "bold": True, "color": CYA}),
    ("No sobreajustamos: sub-entrenamos. Agregar dropout o L1/L2 ahora sería lo contrario de lo que hace falta.", {"size": 15}),
    ("La receta no es «siempre sumar»: es responder a lo que el diagnóstico muestra.", {"size": 15, "bold": True}),
])
b = vinetas(s, Inches(6.95), Inches(1.8), Inches(5.8), Inches(4.9), [
    ("Protocolo de etiquetado 1↔2: sin escribir (el pendiente más importante).", {"bullet": ROJO}),
    ("Domain gap sin probar: todo es sobre renders limpios; ni una foto de teléfono aún → números optimistas.", {"bullet": ROJO}),
    ("Confusión 0↔2 sin resolver (¿ruido o etiquetado?).", {"bullet": ROJO}),
    "Split no estratificado (24/18/18, no 20/20/20).",
    "Dataset chico: 300 es el primer lote; la clase 2 fija el techo.",
    ("MobileNetV3 + TFLite: pendiente (intencional — primero el arco v1→v2).", {"color": MUT}),
], size=15, gap=12)
pie(s, 13, TOTAL)
transicion_fade(s)
animar(s, [t, tx, b])
notas(s, "Mensaje clave 2 y 4: el arco v1→v2 demuestra método; lo que falta está identificado y "
         "priorizado. En un avance, la honestidad suma.")

# 14 — Roadmap + cierre
s = nueva()
titulo(s, "Qué sigue", kicker="Roadmap")
g = imagen(s, AQUI / "roadmap.png", Inches(0.7), Inches(1.75), w=Inches(11.9), borde=False)
c = texto(s, Inches(0.9), Inches(5.1), Inches(11.5), Inches(1.6), [
    ("¿Preguntas?", {"size": 44, "bold": True, "color": CYA, "align": PP_ALIGN.CENTER}),
    ("Is-it-AI · Frameworks de IA · UDD", {"size": 15, "color": MUT, "align": PP_ALIGN.CENTER}),
])
pie(s, 14, TOTAL)
transicion_fade(s, thru_black=True)
animar(s, [g, c])
notas(s, "Inmediato: EarlyStopping(val_loss, patience=8, restore_best_weights) y techo alto de épocas. "
         "v3 depende del re-diagnóstico: domain gap o protocolo de etiquetado. Luego MobileNetV3 y export móvil.")

prs.save(SALIDA)
print(f"OK -> {SALIDA}")
