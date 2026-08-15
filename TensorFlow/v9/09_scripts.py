"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 9  (PRODUCCIÓN: brecha de dominio + fine-tuning en dos fases)

QUÉ CAMBIA RESPECTO DE LA v8 Y POR QUÉ:
  Las versiones v1-v8 fueron un estudio: cada una agregaba UNA técnica y medía si servía.
  El destino del modelo era un notebook. La v9 cambia el destino: el modelo se va a vivir
  dentro de la cámara de un celular, en una app Ionic. Eso cambia dos cosas de raíz.

  1. CAMBIA LA ENTRADA. La app no consume renders de PDF: consume FOTOGRAFÍAS de una
     pantalla. Entre esas dos cosas hay una brecha de dominio (moiré, glare, perspectiva,
     JPEG de cámara). Por eso el dataset v9 sumó 1037 fotos de pantalla a los 300 renders.

  2. CAMBIA LA MÉTRICA. Un test que mezcla renders y fotos reporta un número inflado por
     los renders, que son mucho más fáciles. La v9 evalúa sobre un test 100% FOTOS
     (particion_v9.py) y ese número es la "accuracy comercial".

  Se espera que la accuracy BAJE respecto de la v8. Eso NO es una regresión: la v8 medía
  sobre renders limpios, o sea sobre un problema más fácil que el que el producto resuelve.
  Para que esto quede demostrado y no afirmado, la v9 evalúa el MISMO modelo sobre los dos
  tests (fotos y renders apartados) y reporta la brecha con un número.

EL EXPERIMENTO DE ESTA VERSIÓN (SUGERENCIAS_V9.md §3):
  Dos fases de entrenamiento, comparadas como brazo A y brazo B:

    FASE 1 — BACKBONE CONGELADO. Es la receta de la v7/v8: MobileNetV3-Small congelada
             como extractor de rasgos + la cabeza con los hiperparámetros que Optuna eligió
             en la v7. Los filtros vienen de ImageNet y no se tocan.

    FASE 2 — FINE-TUNING PROGRESIVO. Se descongelan las ÚLTIMAS capas convolucionales y se
             sigue entrenando con un learning rate 1000x más bajo (1e-5 contra el ~5.9e-3
             de la cabeza). La hipótesis: los filtros de ImageNet nunca vieron moiré ni
             glare de pantalla, y con permiso para moverse un poco pueden adaptarse a esas
             texturas. El LR bajísimo es lo que evita que ese permiso destruya lo aprendido
             en ImageNet (catastrophic forgetting).

  La hipótesis se declara ANTES de correr, como en la v8: se espera que la fase 2 gane, y
  se espera que gane MÁS en el test de fotos que en el de renders, porque es justamente el
  dominio nuevo el que los filtros congelados no cubrían. Si ganara parejo en los dos, la
  mejora vendría de tener más capacidad, no de adaptarse al dominio.

DOS DECISIONES TÉCNICAS QUE HAY QUE PODER DEFENDER:

  a) LAS BATCHNORM SE QUEDAN CONGELADAS aunque su capa esté descongelada. Una BN en modo
     entrenamiento recalcula media y varianza con el batch actual y actualiza sus medias
     móviles. Con lotes de 32 y un LR de 1e-5, las estadísticas se mueven mucho más rápido
     que los pesos, y el modelo se desestabiliza justo cuando se le pide que cambie poco.
     En Keras alcanza con poner capa.trainable = False: la BN pasa sola a modo inferencia.
     (En PyTorch NO alcanza con requires_grad_(False) — hay que llamar .eval() sobre el
     módulo. Es el gotcha que el proyecto documentó en la v4 y sigue siendo el contraste
     más limpio entre los dos frameworks.)

  b) LOS HIPERPARÁMETROS DE LA CABEZA SE HARDCODEAN. Son los que Optuna eligió en la v7 por
     validación cruzada. La v9 NO vuelve a buscarlos: buscar hiperparámetros sobre un test
     que ahora es más chico y más difícil solo agregaría sesgo de selección. Se congelan y
     se reportan. Es lo que pide SUGERENCIAS_V9.md §3 ("quema estos hiperparámetros").

PRESUPUESTO DE CÓMPUTO:
  El entorno es CPU (24 hilos, sin GPU). Medido sobre este dataset: el backbone congelado
  procesa ~530 img/s en inferencia y el fine-tuning con solo el último bloque descongelado
  corre a ~380 img/s. Con 3207 vistas de desarrollo, una época de fase 2 cuesta 8 segundos.
  Por eso las DOS fases corren con las 5 semillas del proyecto y la comparación A/B es
  completamente pareada: no hace falta recortar semillas para que el experimento entre en
  el presupuesto.

  (La fase 1 igual entrena sobre embeddings PRE-COMPUTADOS: con el backbone congelado sus
  salidas no cambian entre épocas ni entre semillas, así que computarlas una vez y reusarlas
  da el mismo resultado exacto y ahorra 5 pasadas completas.)

ESPEJO: PyTorch/v9/09_scripts.py hace exactamente lo mismo con las mismas vistas y la misma
partición. Las diferencias que queden entre los dos son del framework, no del experimento.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# La consola de Windows usa cp1252 y no sabe codificar las flechas ↑/↓ que imprime
# metricas.imprimir_agregado(): sin esto el script muere con UnicodeEncodeError DESPUÉS de
# haber entrenado todo. errors="replace" mantiene la codificación nativa (los acentos se
# siguen viendo bien en la terminal) y degrada solo los caracteres que no existen en ella.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np
import tensorflow as tf

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402
from imagenes import cargar_para_particion, generar_vistas  # noqa: E402
from particion_v9 import cargar_particion, conteos, dominios_de, rutas_y_etiquetas  # noqa: E402

tf.get_logger().setLevel("ERROR")

# --- 1. Parámetros ---
SIZE = (224, 224)
BATCH_SIZE = 32
BATCH_FT = 16          # lotes más chicos en fine-tuning: menos memoria y gradiente más ruidoso,
                       # que con LR 1e-5 actúa como regularización extra.

N_VISTAS_AUG = 2       # 1 limpia + 2 aumentadas por imagen de desarrollo
SEED_VISTAS = 999      # fija para TODAS las semillas: los dos brazos y los dos frameworks
                       # ven EXACTAMENTE las mismas vistas.

SEMILLAS = [123, 7, 42, 2024, 31]   # las mismas de v7/v8: continuidad del relato
SEMILLAS_FT = SEMILLAS              # la fase 2 corre las mismas -> comparación pareada

CAPAS_DESCONGELADAS = 12   # últimas capas de MobileNetV3-Small liberadas en la fase 2.
                           # SUGERENCIAS_V9.md sugiere 10-15; 12 deja libre el último bloque
                           # convolucional completo sin tocar los filtros de bajo nivel
                           # (bordes, color), que son universales y no hay razón para mover.
LR_FINETUNE = 1e-5
EPOCAS_FT = 4              # SUGERENCIAS §3 pide "un par de épocas adicionales". Con lotes de
                           # 16 sobre 3207 vistas son ~800 actualizaciones por época; 4 épocas
                           # alcanzan para adaptar el último bloque sin que el LR de 1e-5
                           # llegue a mover los filtros lo suficiente como para olvidar
                           # ImageNet (catastrophic forgetting).

AQUI = Path(__file__).resolve().parent


# --- 2. Datos ---
def preparar_datos():
    """Carga la partición, el caché de imágenes y arma las vistas de cada bloque."""
    particion = cargar_particion()
    clases = particion["clases"]
    print("Orden de clases:", clases)

    meta = particion["meta"]
    print(f"\nPartición v9 (huella {meta['huella_dataset']}) — política: {meta['politica']}")

    imgs, indice = cargar_para_particion(particion)

    bloques = {}
    for nombre in ("desarrollo", "test", "test_render"):
        rutas, y = rutas_y_etiquetas(particion, nombre)
        doms = dominios_de(particion, rutas)
        n_fotos = sum(1 for d in doms if d == "foto")
        print(f"  {nombre:<12}: {len(rutas):>4} imgs  ({n_fotos} fotos / "
              f"{len(rutas) - n_fotos} renders)  {conteos(particion, rutas)}")
        bloques[nombre] = (rutas, y, doms)

    # Vistas aumentadas SOLO en desarrollo. Los tests se evalúan sobre la imagen limpia:
    # es lo que la app va a recibir, y aumentar el test mediría otra cosa.
    print(f"\nGenerando vistas de desarrollo (1 limpia + {N_VISTAS_AUG} aumentadas, "
          f"semilla {SEED_VISTAS})...")
    t0 = time.time()
    rutas_dev, y_dev, doms_dev = bloques["desarrollo"]
    X_dev, Y_dev, D_dev = generar_vistas(imgs, indice, rutas_dev, y_dev, doms_dev,
                                         N_VISTAS_AUG, SEED_VISTAS)
    print(f"  {X_dev.shape}  ({time.time() - t0:.0f}s)")

    tests = {}
    for nombre in ("test", "test_render"):
        rutas, y, _ = bloques[nombre]
        tests[nombre] = (np.stack([imgs[indice[r]] for r in rutas]),
                         np.array(y, dtype=np.int64))

    return particion, clases, X_dev, Y_dev, D_dev, tests


# --- 3. Arquitectura ---
def construir_base() -> tf.keras.Model:
    """MobileNetV3-Small preentrenada en ImageNet, sin la cabeza de clasificación.

    include_preprocessing=True (el default) mete la normalización ImageNet DENTRO del grafo,
    así que el modelo espera píxeles 0-255 crudos. Esa decisión, heredada de la v4, es la
    que permite que la app Ionic pase el canvas tal cual sin replicar la normalización en
    JavaScript: menos código en el cliente y menos superficie para un bug de paridad.
    """
    return tf.keras.applications.MobileNetV3Small(
        input_shape=(*SIZE, 3), include_top=False, weights="imagenet")


def construir_cabeza(params, emb_dim: int, num_clases: int) -> tf.keras.Sequential:
    """Cabeza con los hiperparámetros congelados de la v7. Devuelve LOGITS (sin softmax).

    Se mantiene la convención que impuso la v8: la cabeza termina en logits y el softmax se
    aplica aparte. Sirve para poder aplicar temperatura, y además es lo que espera la
    pérdida from_logits=True, que es numéricamente más estable.
    """
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(emb_dim,))], name="cabeza")
    for _ in range(params["n_capas"]):
        m.add(tf.keras.layers.Dense(params["units"], activation="relu"))
        m.add(tf.keras.layers.Dropout(params["dropout"]))
    m.add(tf.keras.layers.Dense(num_clases))
    return m


def optimizador(params, lr=None):
    lr = params["lr"] if lr is None else lr
    return {"adam": tf.keras.optimizers.Adam(lr),
            "rmsprop": tf.keras.optimizers.RMSprop(lr),
            "sgd": tf.keras.optimizers.SGD(lr, momentum=0.9)}[params["optimizer"]]


def modelo_completo(base, cabeza) -> tf.keras.Model:
    """Backbone + GAP + cabeza, de punta a punta: imágenes 0-255 -> logits."""
    entrada = tf.keras.Input(shape=(*SIZE, 3))
    x = base(entrada)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(entrada, cabeza(x))


def descongelar(base, n_capas: int) -> int:
    """Libera las últimas `n_capas` de la base, dejando TODAS las BatchNorm congeladas.

    Ver la decisión (a) del encabezado: una BN en modo entrenamiento mueve sus estadísticas
    mucho más rápido que lo que el LR de 1e-5 mueve los pesos, y desestabiliza justo la
    fase que pide cambios chicos. En Keras, trainable=False sobre una BatchNormalization la
    pasa automáticamente a modo inferencia; no hace falta nada más.
    """
    base.trainable = True
    for capa in base.layers[:-n_capas]:
        capa.trainable = False
    for capa in base.layers:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False
    return sum(1 for c in base.layers if c.trainable)


# --- 4. Alimentación por lotes y evaluación ---
class LoteImagenes(tf.keras.utils.PyDataset):
    """Entrega lotes uint8 -> float32 sin materializar el dataset entero en float32.

    Por qué no se le pasa el array directo a fit(): las 3207 vistas de desarrollo en
    float32 ocupan 1.9 GB (uint8 son 480 MB). Keras convierte el array completo a tensor
    antes de empezar, así que pasarle el numpy en float32 reservaría esa memoria de golpe.
    Convirtiendo lote a lote el pico baja a unos pocos MB y el entrenamiento no cambia en
    nada: la conversión es exacta, no hay pérdida de información.
    """

    def __init__(self, X, Y=None, batch=32, semilla=0, barajar=False, **kwargs):
        super().__init__(**kwargs)
        self.X, self.Y, self.batch, self.barajar = X, Y, batch, barajar
        self.rng = np.random.default_rng(semilla)
        self.orden = np.arange(len(X))
        if barajar:
            self.rng.shuffle(self.orden)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def __getitem__(self, i):
        idx = self.orden[i * self.batch:(i + 1) * self.batch]
        x = self.X[idx].astype(np.float32)
        return x if self.Y is None else (x, self.Y[idx])

    def on_epoch_end(self):
        if self.barajar:
            self.rng.shuffle(self.orden)


def probabilidades_desde_logits(logits: np.ndarray) -> np.ndarray:
    return tf.nn.softmax(np.asarray(logits), axis=1).numpy()


def evaluar(logits: np.ndarray, Y, clases) -> dict:
    """Convierte logits en el resumen completo de métricas del proyecto."""
    prob = probabilidades_desde_logits(logits)
    r = metricas.resumen_completo(Y, prob, clases)
    r["y_prob"] = prob
    return r


# --- 5. Main ---
if __name__ == "__main__":
    t_inicio = time.time()

    with open(RAIZ / "TensorFlow/v7/mejores_hiperparametros.json", encoding="utf-8") as f:
        cfg_v7 = json.load(f)
    PARAMS = cfg_v7["params"]
    EPOCHS_CABEZA = cfg_v7["epochs_refit"]

    particion, clases, X_dev, Y_dev, D_dev, tests = preparar_datos()
    num_clases = len(clases)

    print(f"\nHiperparámetros CONGELADOS de la v7 (elegidos por CV, sin ver el test):")
    print(f"  {PARAMS}   ·   épocas de la cabeza: {EPOCHS_CABEZA}")
    print(f"Fine-tuning: últimas {CAPAS_DESCONGELADAS} capas · lr={LR_FINETUNE} · "
          f"{EPOCAS_FT} épocas · BatchNorm congeladas")

    # --- 5.1 Embeddings pre-computados (fase 1) ---
    # El backbone está congelado, así que sus salidas NO cambian entre épocas ni entre
    # semillas: computarlas una vez y entrenar la cabeza sobre ellas da exactamente el mismo
    # resultado que pasar las imágenes cada época, y cuesta 100x menos.
    base_congelada = construir_base()
    base_congelada.trainable = False
    entrada = tf.keras.Input(shape=(*SIZE, 3))
    extractor = tf.keras.Model(
        entrada,
        tf.keras.layers.GlobalAveragePooling2D()(base_congelada(entrada, training=False)),
        name="extractor")
    EMB_DIM = extractor.output_shape[-1]

    print(f"\nPre-computando embeddings ({EMB_DIM}-d) con el backbone congelado...")
    t0 = time.time()
    E_dev = extractor.predict(LoteImagenes(X_dev, batch=BATCH_SIZE), verbose=0)
    E_tests = {k: extractor.predict(LoteImagenes(v[0], batch=BATCH_SIZE), verbose=0)
               for k, v in tests.items()}
    print(f"  desarrollo {E_dev.shape}  ({time.time() - t0:.0f}s)")

    # --- 5.2 FASE 1: cabeza sobre backbone congelado ---
    print("\n" + "=" * 78)
    print(f"  FASE 1 — BACKBONE CONGELADO  ·  {len(SEMILLAS)} semillas")
    print("=" * 78)

    res_f1 = {"test": [], "test_render": []}
    cabezas = {}
    for semilla in SEMILLAS:
        tf.keras.utils.set_random_seed(semilla)
        cabeza = construir_cabeza(PARAMS, EMB_DIM, num_clases)
        cabeza.compile(optimizer=optimizador(PARAMS),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        cabeza.fit(E_dev, Y_dev, epochs=EPOCHS_CABEZA, batch_size=BATCH_SIZE,
                   verbose=0, shuffle=True)
        cabezas[semilla] = cabeza

        linea = f"  semilla {semilla:<5}"
        for nombre in ("test", "test_render"):
            r = evaluar(cabeza.predict(E_tests[nombre], verbose=0), tests[nombre][1], clases)
            r["semilla"] = semilla
            res_f1[nombre].append(r)
            linea += f"   {nombre}: acc={r['accuracy']:.4f} 0<->2={r['extremos_total']}"
        print(linea)

    # --- 5.3 FASE 2: fine-tuning progresivo ---
    print("\n" + "=" * 78)
    print(f"  FASE 2 — FINE-TUNING PROGRESIVO  ·  {len(SEMILLAS_FT)} semillas")
    print("=" * 78)

    res_f2 = {"test": [], "test_render": []}
    modelos_ft = {}
    for semilla in SEMILLAS_FT:
        t0 = time.time()
        tf.keras.utils.set_random_seed(semilla)

        # Se parte de la cabeza YA entrenada en la fase 1 con esta MISMA semilla: la fase 2
        # continúa el entrenamiento, no lo reinicia. Si la cabeza arrancara de cero con el
        # backbone descongelado, sus gradientes iniciales (grandes, aleatorios) arrastrarían
        # los filtros preentrenados antes de que la cabeza aprendiera nada útil. Entrenar
        # primero la cabeza y recién después soltar el backbone es EL motivo de que el
        # fine-tuning se haga en dos fases y no en una.
        base = construir_base()
        base.set_weights(base_congelada.get_weights())
        n_entrenables = descongelar(base, CAPAS_DESCONGELADAS)
        modelo = modelo_completo(base, cabezas[semilla])
        modelo.compile(optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        modelo.fit(LoteImagenes(X_dev, Y_dev, batch=BATCH_FT, semilla=semilla, barajar=True),
                   epochs=EPOCAS_FT, verbose=0)
        modelos_ft[semilla] = modelo

        linea = f"  semilla {semilla:<5}"
        for nombre in ("test", "test_render"):
            r = evaluar(modelo.predict(LoteImagenes(tests[nombre][0], batch=BATCH_SIZE),
                                       verbose=0), tests[nombre][1], clases)
            r["semilla"] = semilla
            res_f2[nombre].append(r)
            linea += f"   {nombre}: acc={r['accuracy']:.4f} 0<->2={r['extremos_total']}"
        print(linea + f"   ({time.time() - t0:.0f}s, {n_entrenables} capas entrenables)")

    # --- 5.4 Agregados y comparación A/B ---
    agr = {
        "fase1": {k: metricas.agregar_semillas(v) for k, v in res_f1.items()},
        "fase2": {k: metricas.agregar_semillas(v) for k, v in res_f2.items()},
    }
    # Comparación pareada: solo las semillas que corrieron las DOS fases.
    res_f1_pareado = {k: [r for r in v if r["semilla"] in SEMILLAS_FT]
                      for k, v in res_f1.items()}
    agr_pareado = {k: metricas.agregar_semillas(v) for k, v in res_f1_pareado.items()}

    print()
    metricas.imprimir_agregado(agr["fase1"]["test"],
                               "FASE 1 — congelado · TEST DE FOTOS (accuracy comercial)")
    print()
    metricas.imprimir_agregado(agr["fase2"]["test"],
                               "FASE 2 — fine-tuning · TEST DE FOTOS (accuracy comercial)")

    print("\n" + "=" * 78)
    print(f"  COMPARACIÓN A/B sobre el TEST DE FOTOS  ·  semillas pareadas {SEMILLAS_FT}")
    print("=" * 78)
    print(f"  {'métrica':<24}{'F1 congelado':>14}{'F2 finetune':>13}{'F2-F1':>10}  veredicto")
    print("  " + "-" * 76)
    veredictos = {}
    for clave in metricas.CLAVES_ESCALARES:
        m1, s1 = agr_pareado["test"][clave]["media"], agr_pareado["test"][clave]["std"]
        m2, s2 = agr["fase2"]["test"][clave]["media"], agr["fase2"]["test"][clave]["std"]
        delta = m2 - m1
        # Mismo criterio declarado de antemano que usó la v8: una diferencia solo cuenta si
        # supera la suma de los desvíos entre semillas. Con 207 imágenes de test el ruido es
        # menor que con las 60 de la v8, pero sigue existiendo y decirlo es más honesto que
        # festejar decimales.
        if abs(delta) <= s1 + s2:
            v = "ruido"
        else:
            v = "GANA FINETUNE" if (delta > 0) == metricas.MAS_ES_MEJOR[clave] else "gana congelado"
        veredictos[clave] = v
        print(f"  {metricas.ETIQUETAS[clave]:<24}{m1:>14.4f}{m2:>13.4f}{delta:>+10.4f}  {v}")
    print("  " + "-" * 76)
    print("  'ruido' = |F2-F1| menor o igual que la suma de los desvíos de ambas fases.")

    # --- 5.5 LA BRECHA DE DOMINIO (el resultado propio de la v9) ---
    print("\n" + "=" * 78)
    print("  BRECHA DE DOMINIO — el mismo modelo sobre fotos y sobre renders")
    print("=" * 78)
    print(f"  {'':<18}{'TEST fotos':>13}{'TEST renders':>14}{'brecha':>10}")
    print("  " + "-" * 56)
    brechas = {}
    for fase, etiqueta in (("fase1", "FASE 1 congelado"), ("fase2", "FASE 2 finetune")):
        a_foto = agr[fase]["test"]["accuracy"]["media"]
        a_rend = agr[fase]["test_render"]["accuracy"]["media"]
        brechas[fase] = {"fotos": a_foto, "renders": a_rend, "brecha": a_rend - a_foto}
        print(f"  {etiqueta:<18}{a_foto:>13.4f}{a_rend:>14.4f}{a_rend - a_foto:>+10.4f}")
    print("  " + "-" * 56)
    print("  brecha = accuracy(renders) - accuracy(fotos). El SIGNO es el que informa:")
    print("    brecha > 0 : acierta más sobre renders. Es la ilusión de SUGERENCIAS §2 —")
    print("                 parte de la accuracy de v1-v8 venía del dominio fácil.")
    print("    brecha < 0 : acierta más sobre FOTOS. El dominio difícil pasó a ser el render,")
    print("                 que es lo esperable cuando el 78% del entrenamiento son fotos y")
    print("                 la augmentation además disfraza de foto a los renders.")

    b1, b2 = brechas["fase1"]["brecha"], brechas["fase2"]["brecha"]
    for fase, b in (("Fase 1", b1), ("Fase 2", b2)):
        sentido = ("a favor de los renders" if b > 0 else "a favor de las fotos")
        print(f"\n  {fase}: brecha = {b:+.4f}  ({sentido}, |brecha| = {abs(b):.4f})")

    # Lo que se compara entre fases es el TAMAÑO de la brecha, no su valor con signo: una
    # brecha de -0.08 está tan lejos de la paridad entre dominios como una de +0.08.
    if abs(b2) < abs(b1):
        print(f"\n  El fine-tuning ACERCÓ los dos dominios: |brecha| bajó "
              f"{abs(b1) - abs(b2):.4f} ({abs(b1):.4f} -> {abs(b2):.4f}).")
    else:
        print(f"\n  El fine-tuning SEPARÓ los dos dominios: |brecha| subió "
              f"{abs(b2) - abs(b1):.4f} ({abs(b1):.4f} -> {abs(b2):.4f}).")
        print("  Lectura: el fine-tuning se especializó en el dominio mayoritario (fotos).")
        print("  Para el producto eso es CORRECTO —la app solo ve fotos— pero conviene")
        print("  decirlo explícitamente: el modelo v9 ya no es un buen clasificador de PDFs.")

    # --- 5.6 El riesgo de §4: la confusión extrema 0<->2 ---
    print("\n" + "=" * 78)
    print("  RIESGO DE SUGERENCIAS §4 — ¿el moiré empuja las diapos humanas hacia 'saturada'?")
    print("=" * 78)
    for fase in ("fase1", "fase2"):
        a = agr[fase]["test"]
        print(f"  {fase}: 0->2 = {a['sin_ia_como_saturada']['media']:.1f} casos   ·   "
              f"2->0 = {a['saturada_como_sin_ia']['media']:.1f} casos   ·   "
              f"total extremos = {a['extremos_total']['media']:.1f}")
    print("  (0->2 es el falso positivo del producto: una diapo humana acusada de IA.")
    print("   Si sube al pasar a fotos, la simulación de moiré del augmentation no alcanzó")
    print("   y hay que subir su intensidad en documentacion/aug_pantalla.py.)")

    # --- 5.7 Matrices y figuras ---
    for fase in ("fase1", "fase2"):
        cm = np.array(agr[fase]["test"]["cm_sumada"])
        print(f"\nMatriz ACUMULADA — {fase} sobre TEST DE FOTOS:")
        print(cm)
        metricas.reporte_por_clase(cm, clases)

    print("\nGenerando figuras...")
    metricas.plot_matriz(np.array(agr["fase1"]["test"]["cm_sumada"]), clases,
                         "Matriz v9 — Fase 1 congelado (test de fotos)",
                         AQUI / "Figure_matriz_fase1_v9.png")
    metricas.plot_matriz(np.array(agr["fase2"]["test"]["cm_sumada"]), clases,
                         "Matriz v9 — Fase 2 fine-tuning (test de fotos)",
                         AQUI / "Figure_matriz_fase2_v9.png")
    metricas.plot_calibracion(tests["test"][1], res_f2["test"][0]["y_prob"],
                              f"Calibración v9 — fase 2 (semilla {SEMILLAS_FT[0]})",
                              AQUI / "Figure_calibracion_v9.png")
    curvas, micro, macro = metricas.curvas_roc_ovr(
        tests["test"][1], res_f2["test"][0]["y_prob"], num_clases)
    metricas.plot_roc(curvas, micro, macro, clases,
                      "ROC One-vs-Rest v9 — fase 2 (test de fotos)",
                      AQUI / "Figure_roc_v9.png")
    metricas.plot_semillas(res_f2["test"], ["accuracy", "macro_f1", "qwk", "ece"],
                           "Dispersión entre semillas — v9 fase 2 (TensorFlow)",
                           AQUI / "Figure_semillas_v9.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    ancho = 0.35
    ax.bar(x - ancho / 2, [brechas["fase1"]["fotos"], brechas["fase1"]["renders"]],
           ancho, label="Fase 1 (congelado)", color="#1F3864")
    ax.bar(x + ancho / 2, [brechas["fase2"]["fotos"], brechas["fase2"]["renders"]],
           ancho, label="Fase 2 (fine-tuning)", color="#B45309")
    ax.set_xticks(x)
    ax.set_xticklabels(["TEST fotos\n(accuracy comercial)", "TEST renders\n(laboratorio)"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Brecha de dominio — TensorFlow v9\nla diferencia entre las barras es lo "
                 "que inflaba la métrica de v1-v8",
                 fontweight="bold", color="#1F3864")
    for i, fase in enumerate(("fase1", "fase2")):
        for j, k in enumerate(("fotos", "renders")):
            v = brechas[fase][k]
            ax.text(j + (i - 0.5) * ancho, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(AQUI / "Figure_brecha_dominio_v9.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  guardado Figure_brecha_dominio_v9.png")

    # --- 5.8 Guardar resultados y el modelo de producción ---
    def limpiar(agregado):
        return {k: agregado[k] for k in metricas.CLAVES_ESCALARES + ["cm_sumada", "n_semillas"]}

    with open(AQUI / "resultados_v9.json", "w", encoding="utf-8") as f:
        json.dump({
            "framework": "tensorflow",
            "particion": particion["meta"],
            "semillas_fase1": SEMILLAS,
            "semillas_fase2": SEMILLAS_FT,
            "params_congelados_v7": PARAMS,
            "finetuning": {"capas_descongeladas": CAPAS_DESCONGELADAS, "lr": LR_FINETUNE,
                           "epocas": EPOCAS_FT, "batchnorm": "congeladas",
                           "batch_size": BATCH_FT},
            "vistas": {"n_aug": N_VISTAS_AUG, "seed_vistas": SEED_VISTAS},
            "fase1": {k: limpiar(v) for k, v in agr["fase1"].items()},
            "fase2": {k: limpiar(v) for k, v in agr["fase2"].items()},
            "comparacion_pareada": {k: limpiar(v) for k, v in agr_pareado.items()},
            "veredictos": veredictos,
            "brecha_dominio": brechas,
        }, f, indent=2, ensure_ascii=False)
    print("  guardado resultados_v9.json")

    # Modelo de PRODUCCIÓN: imágenes 0-255 -> softmax. Es el archivo que consume la
    # exportación a TensorFlow.js y a ONNX, y el que termina dentro de la app Ionic.
    mejor_semilla = max(SEMILLAS_FT,
                        key=lambda s: next(r["accuracy"] for r in res_f2["test"]
                                           if r["semilla"] == s))
    modelo_prod = tf.keras.Model(
        modelos_ft[mejor_semilla].input,
        tf.keras.layers.Softmax(name="probabilidades")(modelos_ft[mejor_semilla].output))
    modelo_prod.save(AQUI / "modelotf_v9_finetune.keras")
    print(f"  guardado modelotf_v9_finetune.keras  (fase 2, semilla {mejor_semilla})")

    # Metadatos que la app necesita para no adivinar nada del modelo.
    with open(AQUI / "modelo_v9_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v9", "framework": "tensorflow", "arquitectura": "MobileNetV3-Small",
            "clases": clases, "entrada": {"alto": SIZE[0], "ancho": SIZE[1], "canales": 3,
                                          "rango": "0-255", "orden": "NHWC"},
            "preprocesamiento": "ninguno en el cliente: la normalización ImageNet viaja "
                                "dentro del grafo (include_preprocessing=True)",
            "salida": "softmax de 3 clases, en el orden de 'clases'",
            "semilla": mejor_semilla,
            "accuracy_test_fotos": float(
                next(r["accuracy"] for r in res_f2["test"] if r["semilla"] == mejor_semilla)),
        }, f, indent=2, ensure_ascii=False)
    print("  guardado modelo_v9_meta.json")

    print(f"\nTiempo total: {(time.time() - t_inicio) / 60:.1f} min")
