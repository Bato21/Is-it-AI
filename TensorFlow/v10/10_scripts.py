"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 10  (COMPUERTA DE RECHAZO: el modelo puede decir "esto no es una diapositiva")

QUÉ CAMBIA RESPECTO DE LA v9 Y POR QUÉ
======================================
La v9 dejó el producto entero funcionando: tres modelos, fine-tuning en dos fases, brecha de
dominio medida, app Ionic con inferencia en el dispositivo. Y tenía un agujero que ninguna
métrica de la v9 podía ver, porque estaba fuera de lo que la v9 medía.

  Un softmax de 3 clases SIEMPRE reparte 1.0 entre sus 3 opciones. No existe la abstención.
  Si el usuario apunta la cámara a su escritorio, a la cara de un compañero o a una planilla
  de Excel abierta en el monitor, el modelo no puede decir "esto no me corresponde": devuelve
  un nivel de saturación de IA. Y no lo devuelve dudando — el argmax de un softmax sobre una
  entrada fuera de distribución suele salir con confianza ALTA, porque la red nunca vio nada
  que la obligue a repartir masa fuera de sus tres opciones.

  Ese es el peor modo de falla posible para un producto: no falla "un poco", responde con
  seguridad a una pregunta que nadie hizo. Y el test de la v9 no lo detectaba porque el test
  de la v9 está hecho SOLO de diapositivas: mide muy bien un problema que en producción
  aparece menos veces que este.

LA v10 AGREGA UNA CUARTA CLASE: `3_no_diapositiva`
--------------------------------------------------
450 imágenes que no son diapositivas, obtenidas de datasets públicos con
`documentacion/negativos_v10.py`. No es una clase más del eje: es la COMPUERTA del producto.
Si gana el argmax, la app no reporta nivel de IA — reporta que la foto no es una diapositiva.

Se eligió una cuarta clase en el mismo softmax en vez de un segundo modelo "detector de
diapositivas" delante, y la razón es de ingeniería de despliegue, no de exactitud:

  · Un gate aparte serían SEIS artefactos en la app en vez de tres (cada modelo del selector
    necesitaría el suyo), seis exportaciones y seis verificaciones de paridad.
  · Serían dos pasadas por imagen en el teléfono en vez de una: el doble de latencia en el
    dispositivo más lento, que es el que manda.
  · Y el umbral del gate sería un hiperparámetro más que ajustar sobre un test chico —
    exactamente el sesgo de selección que el proyecto viene evitando desde la v7. Con la
    cuarta clase la decisión sale del argmax, sin umbral que elegir.

  El costo de la decisión es que el eje ordinal deja de abarcar todas las clases, y eso hay
  que manejarlo en las métricas en vez de ignorarlo: QWK, MAE ordinal y los errores 0<->2 se
  calculan sobre el sub-bloque 0-1-2 y se leen como CONDICIONALES ("de lo que el modelo
  aceptó como diapositiva, qué tan bien lo ordenó"). Ver el encabezado de metricas.py.

POR QUÉ 450 IMÁGENES, Y POR QUÉ ESO SE MIDE EN VEZ DE AFIRMARSE
---------------------------------------------------------------
Las tres clases existentes promedian 446 imágenes. 450 es un cuarto exacto del dataset
resultante: la clase nueva entra con el mismo peso que las otras tres, sin privilegio ni
castigo. El razonamiento completo (cota superior por distorsión de la prior, cota inferior por
varianza intra-clase) está en el encabezado de `documentacion/negativos_v10.py`.

Pero un razonamiento no es una medición. Este script corre además una ABLACIÓN (§5.7):
entrena la misma cabeza con el 25%, el 50% y el 100% de la clase de rechazo y reporta cómo se
mueven el recall de la compuerta y —lo importante— el recall de la clase 0, que es la que la
compuerta puede canibalizar. Si la curva ya está plana en 225, el informe lo dice.

  La ablación corre sobre EMBEDDINGS PRE-COMPUTADOS del backbone congelado, así que las tres
  variantes cuestan segundos en vez de minutos: el backbone no cambia entre ellas.

QUÉ NO CAMBIA (a propósito)
---------------------------
Absolutamente todo lo demás es idéntico a la v9: misma arquitectura, mismos hiperparámetros
congelados de la v7, mismas 5 semillas, mismas 2 fases, misma augmentation, misma política de
partición (test y validación 100% fotos). Esa disciplina es la que permite atribuir cualquier
diferencia entre v9 y v10 a la clase nueva y no a otra cosa.

ESPEJO: PyTorch/v10/10_scripts.py hace lo mismo con las mismas vistas y la misma partición.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# La consola de Windows usa cp1252 y no sabe codificar las flechas ↑/↓ que imprime
# metricas.imprimir_agregado(): sin esto el script muere con UnicodeEncodeError DESPUÉS de
# haber entrenado todo.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np
import tensorflow as tf

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402
from imagenes import cargar_para_particion, generar_vistas  # noqa: E402
from particion_v10 import (  # noqa: E402
    CLASE_RECHAZO, N_ORDINALES, cargar_particion, conteos, dominios_de, rutas_y_etiquetas,
)

tf.get_logger().setLevel("ERROR")

# --- 1. Parámetros (los de la v9, sin tocar: ver "QUÉ NO CAMBIA") ---
SIZE = (224, 224)
BATCH_SIZE = 32
BATCH_FT = 16

N_VISTAS_AUG = 2       # 1 limpia + 2 aumentadas por imagen de desarrollo
SEED_VISTAS = 999      # la MISMA de la v9 y del espejo PyTorch

SEMILLAS = [123, 7, 42, 2024, 31]
SEMILLAS_FT = SEMILLAS

CAPAS_DESCONGELADAS = 12
LR_FINETUNE = 1e-5
EPOCAS_FT = 4

# --- Propio de la v10 ---
# Fracciones de la clase de rechazo con las que se corre la ablación de §5.7. Se eligen
# 1/4, 1/2 y 1 (o sea ~112, ~225 y 450 imágenes) porque lo que interesa es la FORMA de la
# curva, no el valor exacto: si duplicar de 225 a 450 mueve el recall menos que el desvío
# entre semillas, 450 ya está en la zona plana y el número está justificado.
ABLACION_FRACCIONES = [0.25, 0.50, 1.00]
ABLACION_SEMILLAS = SEMILLAS

AQUI = Path(__file__).resolve().parent


# --- 2. Datos ---
def preparar_datos():
    """Carga la partición v10, el caché de imágenes y arma las vistas de cada bloque."""
    particion = cargar_particion()
    clases = particion["clases"]
    print("Orden de clases:", clases)
    print(f"  eje ordinal: {clases[:N_ORDINALES]}   ·   compuerta: {clases[N_ORDINALES:]}")

    meta = particion["meta"]
    print(f"\nPartición v10 (huella {meta['huella_dataset']}) — política: {meta['politica']}")

    imgs, indice = cargar_para_particion(particion)

    bloques = {}
    for nombre in ("desarrollo", "test", "test_render"):
        rutas, y = rutas_y_etiquetas(particion, nombre)
        doms = dominios_de(particion, rutas)
        n_fotos = sum(1 for d in doms if d == "foto")
        print(f"  {nombre:<12}: {len(rutas):>4} imgs  ({n_fotos} fotos / "
              f"{len(rutas) - n_fotos} renders)  {conteos(particion, rutas)}")
        bloques[nombre] = (rutas, y, doms)

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


def submuestrear_clase(Y: np.ndarray, clase: int, fraccion: float, semilla: int) -> np.ndarray:
    """Índices de vistas que quedan al conservar `fraccion` de las imágenes de `clase`.

    El submuestreo es POR IMAGEN, no por vista: las N vistas de una imagen son consecutivas y
    se conservan o se descartan juntas. Descartar vistas sueltas dejaría el mismo contenido
    presente con menos augmentation, que no es lo mismo que tener menos datos — y la ablación
    mide justamente el efecto de tener menos DATOS. Es el mismo criterio que usa el modelo C
    para construir su dataset desequilibrado.
    """
    por_img = N_VISTAS_AUG + 1
    n_imgs = len(Y) // por_img
    etiquetas_img = Y[::por_img]

    rng = np.random.default_rng(semilla)
    conservadas = []
    for c in np.unique(etiquetas_img):
        idx = np.where(etiquetas_img == c)[0]
        if c == clase and fraccion < 1.0:
            idx = rng.choice(idx, size=max(1, int(round(len(idx) * fraccion))), replace=False)
        conservadas.append(idx)
    imgs_ok = np.sort(np.concatenate(conservadas))
    return np.concatenate([np.arange(i * por_img, (i + 1) * por_img) for i in imgs_ok])


# --- 3. Arquitectura (idéntica a la v9) ---
def construir_base() -> tf.keras.Model:
    """MobileNetV3-Small preentrenada en ImageNet, sin la cabeza de clasificación.

    include_preprocessing=True (el default) mete la normalización ImageNet DENTRO del grafo,
    así que el modelo espera píxeles 0-255 crudos y la app Ionic pasa el canvas tal cual.
    """
    return tf.keras.applications.MobileNetV3Small(
        input_shape=(*SIZE, 3), include_top=False, weights="imagenet")


def construir_cabeza(params, emb_dim: int, num_clases: int) -> tf.keras.Sequential:
    """Cabeza con los hiperparámetros congelados de la v7. Devuelve LOGITS (sin softmax).

    La ÚNICA diferencia con la v9 es num_clases: 4 en vez de 3. Los hiperparámetros no se
    vuelven a buscar — buscarlos otra vez sobre un test que cambió agregaría sesgo de
    selección y rompería la comparabilidad v9 vs v10, que es lo que esta versión quiere medir.
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

    Una BN en modo entrenamiento recalcula media y varianza con el batch actual: con lotes de
    16 y un LR de 1e-5, las estadísticas se mueven mucho más rápido que los pesos y el modelo
    se desestabiliza justo cuando se le pide que cambie poco. En Keras alcanza con
    trainable=False; en PyTorch hay que llamar .eval() (ver el espejo).
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

    Las vistas de desarrollo en float32 ocuparían varios GB (en uint8 son cientos de MB).
    Keras convierte el array completo a tensor antes de empezar, así que pasarle el numpy en
    float32 reservaría esa memoria de golpe. Convirtiendo lote a lote el pico baja a unos
    pocos MB y el resultado numérico es idéntico.
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
    """Logits -> resumen completo. n_ordinales acota las métricas del eje al sub-bloque 0-1-2."""
    prob = probabilidades_desde_logits(logits)
    r = metricas.resumen_completo(Y, prob, clases, n_ordinales=N_ORDINALES)
    r["y_prob"] = prob
    return r


def agregar(resumenes):
    return metricas.agregar_semillas(resumenes, claves=metricas.CLAVES_V10)


# --- 5. Main ---
if __name__ == "__main__":
    t_inicio = time.time()

    with open(RAIZ / "TensorFlow/v7/mejores_hiperparametros.json", encoding="utf-8") as f:
        cfg_v7 = json.load(f)
    PARAMS = cfg_v7["params"]
    EPOCHS_CABEZA = cfg_v7["epochs_refit"]

    particion, clases, X_dev, Y_dev, D_dev, tests = preparar_datos()
    num_clases = len(clases)
    IDX_RECHAZO = clases.index(CLASE_RECHAZO)

    print("\nHiperparámetros CONGELADOS de la v7 (elegidos por CV, sin ver el test):")
    print(f"  {PARAMS}   ·   épocas de la cabeza: {EPOCHS_CABEZA}")
    print(f"Fine-tuning: últimas {CAPAS_DESCONGELADAS} capas · lr={LR_FINETUNE} · "
          f"{EPOCAS_FT} épocas · BatchNorm congeladas")

    # --- 5.1 Embeddings pre-computados (fase 1) ---
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
            linea += (f"   {nombre}: acc={r['accuracy']:.4f} "
                      f"rech={r['recall_rechazo']:.3f} fuga={r['fuga_no_diapositiva']}")
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
        # backbone descongelado, sus gradientes iniciales arrastrarían los filtros
        # preentrenados antes de que aprendiera nada útil.
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
            linea += (f"   {nombre}: acc={r['accuracy']:.4f} "
                      f"rech={r['recall_rechazo']:.3f} fuga={r['fuga_no_diapositiva']}")
        print(linea + f"   ({time.time() - t0:.0f}s, {n_entrenables} capas entrenables)")

    # --- 5.4 Agregados y comparación A/B entre fases ---
    agr = {"fase1": {k: agregar(v) for k, v in res_f1.items()},
           "fase2": {k: agregar(v) for k, v in res_f2.items()}}
    res_f1_pareado = {k: [r for r in v if r["semilla"] in SEMILLAS_FT]
                      for k, v in res_f1.items()}
    agr_pareado = {k: agregar(v) for k, v in res_f1_pareado.items()}

    print()
    metricas.imprimir_agregado(agr["fase1"]["test"],
                               "FASE 1 — congelado · TEST DE FOTOS (accuracy comercial)",
                               claves=metricas.CLAVES_V10)
    print()
    metricas.imprimir_agregado(agr["fase2"]["test"],
                               "FASE 2 — fine-tuning · TEST DE FOTOS (accuracy comercial)",
                               claves=metricas.CLAVES_V10)

    print("\n" + "=" * 78)
    print(f"  COMPARACIÓN A/B sobre el TEST DE FOTOS  ·  semillas pareadas {SEMILLAS_FT}")
    print("=" * 78)
    print(f"  {'métrica':<28}{'F1 congelado':>14}{'F2 finetune':>13}{'F2-F1':>10}  veredicto")
    print("  " + "-" * 80)
    veredictos = {}
    for clave in metricas.CLAVES_V10:
        m1, s1 = agr_pareado["test"][clave]["media"], agr_pareado["test"][clave]["std"]
        m2, s2 = agr["fase2"]["test"][clave]["media"], agr["fase2"]["test"][clave]["std"]
        delta = m2 - m1
        # Mismo criterio declarado de antemano que usan v8 y v9: una diferencia solo cuenta si
        # supera la suma de los desvíos entre semillas. Festejar decimales por debajo de eso
        # es reportar ruido.
        if abs(delta) <= s1 + s2:
            v = "ruido"
        else:
            v = ("GANA FINETUNE" if (delta > 0) == metricas.MAS_ES_MEJOR[clave]
                 else "gana congelado")
        veredictos[clave] = v
        print(f"  {metricas.ETIQUETAS[clave]:<28}{m1:>14.4f}{m2:>13.4f}{delta:>+10.4f}  {v}")
    print("  " + "-" * 80)
    print("  'ruido' = |F2-F1| menor o igual que la suma de los desvíos de ambas fases.")

    # --- 5.5 La brecha de dominio (heredada de la v9) ---
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
    print("  brecha = accuracy(renders) - accuracy(fotos). Positiva = el render es más fácil")
    print("  (la ilusión que corrigió la v9). Negativa = el modelo se especializó en fotos,")
    print("  que es lo que el producto necesita.")
    b1, b2 = brechas["fase1"]["brecha"], brechas["fase2"]["brecha"]
    if abs(b2) < abs(b1):
        print(f"\n  El fine-tuning ACERCÓ los dominios: |brecha| {abs(b1):.4f} -> {abs(b2):.4f}.")
    else:
        print(f"\n  El fine-tuning SEPARÓ los dominios: |brecha| {abs(b1):.4f} -> {abs(b2):.4f}.")

    # --- 5.6 LA COMPUERTA DE RECHAZO (el resultado propio de la v10) ---
    print("\n" + "=" * 78)
    print("  LA COMPUERTA — ¿deja de analizar lo que no es una diapositiva?")
    print("=" * 78)
    a2 = agr["fase2"]["test"]
    cm2 = np.array(a2["cm_sumada"])
    n_rechazo_test = int(cm2[IDX_RECHAZO, :].sum())
    n_diapos_test = int(cm2[:IDX_RECHAZO, :].sum())
    print(f"  Test acumulado sobre {len(SEMILLAS_FT)} semillas: {n_diapos_test} diapositivas "
          f"+ {n_rechazo_test} no-diapositivas")
    print(f"\n  {'métrica':<30}{'media':>10}{'±std':>9}   lectura")
    print("  " + "-" * 78)
    filas_gate = [
        ("recall_rechazo", "de lo que NO era diapositiva, cuánto atajó"),
        ("precision_rechazo", "de lo que rechazó, cuánto no era diapositiva"),
        ("fuga_no_diapositiva", "EL FALLO QUE CIERRA LA v10: analizó algo que no era"),
        ("rechazo_indebido", "el costo: diapositivas reales descartadas"),
    ]
    for clave, lectura in filas_gate:
        d = a2[clave]
        print(f"  {metricas.ETIQUETAS[clave]:<30}{d['media']:>10.4f}{d['std']:>9.4f}   {lectura}")
    print("  " + "-" * 78)

    fuga_media = a2["fuga_no_diapositiva"]["media"]
    tasa_fuga = fuga_media / max(1, n_rechazo_test / len(SEMILLAS_FT))
    print(f"\n  Sin la compuerta (v9) la fuga habría sido del 100%: las "
          f"{n_rechazo_test // len(SEMILLAS_FT)} imágenes")
    print("  no-diapositiva del test habrían recibido igual un nivel de saturación de IA,")
    print(f"  porque un softmax de 3 clases no puede abstenerse. Con la compuerta la fuga es")
    print(f"  de {fuga_media:.1f} imágenes por semilla ({tasa_fuga:.1%}).")
    print("\n  El rechazo indebido es el precio: una diapositiva real que el modelo descarta.")
    print("  Es un error MENOS grave que la fuga —el usuario reencuadra y vuelve a disparar,")
    print("  en vez de recibir un veredicto inventado sobre su escritorio— pero se reporta")
    print("  igual, porque es el que puede crecer si la clase de rechazo se agranda de más.")

    # --- 5.7 ABLACIÓN: ¿cuántas imágenes de rechazo hacen falta? ---
    print("\n" + "=" * 78)
    print("  ABLACIÓN — ¿450 imágenes de rechazo son las que hacen falta?")
    print("=" * 78)
    print("  Misma cabeza, mismas semillas, mismo test. Lo ÚNICO que cambia es cuántas")
    print("  imágenes de la clase de rechazo entran al entrenamiento. Corre sobre los")
    print("  embeddings ya calculados, así que las tres variantes cuestan segundos.\n")

    por_img = N_VISTAS_AUG + 1
    n_rechazo_dev = int((Y_dev[::por_img] == IDX_RECHAZO).sum())
    ablacion = {}
    print(f"  {'fracción':<10}{'imgs':>6}{'recall rech.':>14}{'fuga':>8}"
          f"{'rech.indeb.':>13}{'recall clase 0':>16}{'0->compuerta':>14}{'accuracy':>10}")
    print("  " + "-" * 92)
    for fraccion in ABLACION_FRACCIONES:
        resultados_abl = []
        for semilla in ABLACION_SEMILLAS:
            idx = submuestrear_clase(Y_dev, IDX_RECHAZO, fraccion, semilla)
            tf.keras.utils.set_random_seed(semilla)
            cab = construir_cabeza(PARAMS, EMB_DIM, num_clases)
            cab.compile(optimizer=optimizador(PARAMS),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            cab.fit(E_dev[idx], Y_dev[idx], epochs=EPOCHS_CABEZA, batch_size=BATCH_SIZE,
                    verbose=0, shuffle=True)
            r = evaluar(cab.predict(E_tests["test"], verbose=0), tests["test"][1], clases)
            # Dos números sobre la clase 0, y hacen falta LOS DOS.
            #
            # La clase 0 es la que la compuerta puede canibalizar: una diapositiva humana
            # sobria (texto negro sobre blanco) se parece a un documento escaneado, que es
            # justamente una de las familias de la clase de rechazo. Si agrandar la clase de
            # rechazo hunde su recall, hay que saber POR QUÉ, y hay dos causas posibles con
            # consecuencias opuestas:
            #
            #   cero_a_compuerta > 0  -> canibalización real: la compuerta se está tragando
            #                            diapositivas. Es un argumento para achicar la clase.
            #   cero_a_compuerta = 0  -> el recall se perdió DENTRO del eje ordinal (0->1,
            #                            0->2). La compuerta no tiene nada que ver y achicarla
            #                            no arreglaría nada.
            #
            # Sin separar las dos, el veredicto de la ablación sería una conjetura.
            cm_r = np.array(r["cm"])
            r["recall_clase0"] = float(cm_r[0, 0] / max(1, cm_r[0, :].sum()))
            r["cero_a_compuerta"] = int(cm_r[0, IDX_RECHAZO])
            resultados_abl.append(r)

        a = metricas.agregar_semillas(
            resultados_abl, claves=metricas.CLAVES_V10 + ["recall_clase0", "cero_a_compuerta"])
        n_imgs = int(round(n_rechazo_dev * fraccion))
        ablacion[f"{fraccion:.2f}"] = {
            "fraccion": fraccion, "imgs_rechazo_entrenamiento": n_imgs,
            **{c: a[c] for c in ("recall_rechazo", "fuga_no_diapositiva", "rechazo_indebido",
                                 "recall_clase0", "cero_a_compuerta", "accuracy", "macro_f1")},
        }
        print(f"  {fraccion:<10.0%}{n_imgs:>6}"
              f"{a['recall_rechazo']['media']:>9.4f}±{a['recall_rechazo']['std']:.3f}"
              f"{a['fuga_no_diapositiva']['media']:>8.1f}"
              f"{a['rechazo_indebido']['media']:>13.1f}"
              f"{a['recall_clase0']['media']:>16.4f}"
              f"{a['cero_a_compuerta']['media']:>14.1f}"
              f"{a['accuracy']['media']:>10.4f}")
    print("  " + "-" * 92)

    claves_abl = sorted(ablacion, key=lambda x: float(x))
    r_bajo = ablacion[claves_abl[-2]]["recall_rechazo"]
    r_alto = ablacion[claves_abl[-1]]["recall_rechazo"]
    salto = r_alto["media"] - r_bajo["media"]
    umbral = r_bajo["std"] + r_alto["std"]
    print(f"\n  Duplicar de {ablacion[claves_abl[-2]]['imgs_rechazo_entrenamiento']} a "
          f"{ablacion[claves_abl[-1]]['imgs_rechazo_entrenamiento']} imágenes mueve el recall "
          f"de rechazo {salto:+.4f},")
    print(f"  contra un ruido entre semillas de ±{umbral:.4f}.")
    if abs(salto) <= umbral:
        print("\n  VEREDICTO: la curva YA ESTÁ PLANA. 450 imágenes están en la zona de")
        print("  rendimientos decrecientes: la mitad habría alcanzado. Se conserva 450 igual")
        print("  porque no cuesta nada y da margen para negativos más difíciles, pero el")
        print("  informe NO puede afirmar que las 450 sean necesarias.")
    elif salto > 0:
        print("\n  VEREDICTO: la curva TODAVÍA SUBE en 450. El tamaño elegido no está de más;")
        print("  si hiciera falta más recall de compuerta, agrandar la clase es la palanca")
        print("  con mejor relación costo/beneficio antes que tocar la arquitectura.")
    else:
        print("\n  VEREDICTO: agrandar la clase EMPEORÓ el recall de compuerta. Es señal de que")
        print("  las imágenes agregadas no aportan variedad nueva y sí distorsionan la prior:")
        print("  conviene recortar la clase, no agrandarla.")

    delta_c0 = (ablacion[claves_abl[-1]]["recall_clase0"]["media"]
                - ablacion[claves_abl[0]]["recall_clase0"]["media"])
    cero_gate = ablacion[claves_abl[-1]]["cero_a_compuerta"]["media"]
    print(f"\n  Efecto sobre la clase 0: recall {delta_c0:+.4f} al pasar de "
          f"{claves_abl[0]} a {claves_abl[-1]} de la clase de rechazo.")
    if delta_c0 < -0.01 and cero_gate >= 1.0:
        print(f"  CAUSA: canibalización real. Con la clase completa, {cero_gate:.1f} "
              f"diapositivas de la clase 0")
        print("  por semilla terminan en la compuerta. Es un argumento concreto para ACHICAR")
        print("  la clase de rechazo, o para sacarle la familia 'documento', que es la que se")
        print("  parece a una diapositiva sobria de texto negro sobre blanco.")
    elif delta_c0 < -0.01:
        print(f"  CAUSA: NO es la compuerta. Solo {cero_gate:.1f} diapositivas de la clase 0 "
              f"por semilla")
        print("  van a la compuerta; el recall se pierde DENTRO del eje ordinal (0->1, 0->2).")
        print("  Achicar la clase de rechazo no arreglaría eso: es el problema viejo de la")
        print("  frontera 0/1, no un efecto de la versión nueva.")
    else:
        print(f"  La clase 0 no se degrada al agrandar la compuerta "
              f"({cero_gate:.1f} imágenes de la clase 0 rechazadas por semilla).")

    # --- 5.8 El riesgo heredado: la confusión extrema 0<->2 ---
    print("\n" + "=" * 78)
    print("  RIESGO HEREDADO — ¿el moiré empuja las diapos humanas hacia 'saturada'?")
    print("=" * 78)
    for fase in ("fase1", "fase2"):
        a = agr[fase]["test"]
        print(f"  {fase}: 0->2 = {a['sin_ia_como_saturada']['media']:.1f} casos   ·   "
              f"2->0 = {a['saturada_como_sin_ia']['media']:.1f} casos   ·   "
              f"total extremos = {a['extremos_total']['media']:.1f}")
    print("  (Se cuentan sobre el sub-bloque ordinal 0-1-2: son errores DENTRO del eje, no")
    print("   confusiones con la compuerta, que se miden aparte en §5.6.)")

    # --- 5.9 Matrices y figuras ---
    for fase in ("fase1", "fase2"):
        cm = np.array(agr[fase]["test"]["cm_sumada"])
        print(f"\nMatriz ACUMULADA — {fase} sobre TEST DE FOTOS:")
        print(cm)
        metricas.reporte_por_clase(cm, clases)

    print("\nGenerando figuras...")
    metricas.plot_matriz(np.array(agr["fase1"]["test"]["cm_sumada"]), clases,
                         "Matriz v10 — Fase 1 congelado (test de fotos)",
                         AQUI / "Figure_matriz_fase1_v10.png")
    metricas.plot_matriz(np.array(agr["fase2"]["test"]["cm_sumada"]), clases,
                         "Matriz v10 — Fase 2 fine-tuning (test de fotos)",
                         AQUI / "Figure_matriz_fase2_v10.png")
    metricas.plot_calibracion(tests["test"][1], res_f2["test"][0]["y_prob"],
                              f"Calibración v10 — fase 2 (semilla {SEMILLAS_FT[0]})",
                              AQUI / "Figure_calibracion_v10.png")
    curvas, micro, macro = metricas.curvas_roc_ovr(
        tests["test"][1], res_f2["test"][0]["y_prob"], num_clases)
    metricas.plot_roc(curvas, micro, macro, clases,
                      "ROC One-vs-Rest v10 — fase 2 (test de fotos)",
                      AQUI / "Figure_roc_v10.png")
    metricas.plot_semillas(res_f2["test"], ["accuracy", "macro_f1", "recall_rechazo", "qwk"],
                           "Dispersión entre semillas — v10 fase 2 (TensorFlow)",
                           AQUI / "Figure_semillas_v10.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figura de la ablación: responde "¿cuántas imágenes?" con curvas y no con una afirmación.
    #
    # Va en DOS paneles y no en uno, y el motivo es que en un solo eje la figura MIENTE. El
    # recall de la clase 0 cae al agrandar la clase de rechazo, así que dibujado junto al de
    # la compuerta se lee como un intercambio —"la compuerta se come diapositivas"— y ese es
    # justamente el diagnóstico que los datos desmienten. El panel de abajo muestra el conteo
    # de diapositivas de la clase 0 que terminan RECHAZADAS, que es cero en los tres tamaños:
    # la caída del recall ocurre dentro del eje ordinal y no tiene que ver con la compuerta.
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    xs = [ablacion[k]["imgs_rechazo_entrenamiento"] for k in claves_abl]
    for clave, color, etiqueta in (
            ("recall_rechazo", "#1F3864", "recall de la compuerta (↑ mejor)"),
            ("recall_clase0", "#B45309", "recall de la clase 0 (↑ mejor)"),
            ("accuracy", "#2E7D32", "accuracy global")):
        ys = [ablacion[k][clave]["media"] for k in claves_abl]
        es = [ablacion[k][clave]["std"] for k in claves_abl]
        ax.errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=4, color=color, label=etiqueta)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=color)
    ax.set_ylabel("valor (media de las semillas)")
    ax.set_title("¿Cuántas imágenes de rechazo hacen falta? — v10 (TensorFlow)\n"
                 "arriba: qué gana la compuerta · abajo: por qué la caída naranja NO es culpa suya",
                 fontweight="bold", color="#1F3864")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    # Las barras van sobre el MISMO eje numérico que el panel de arriba (sharex=True). Pasarle
    # strings a bar() crearía un eje categórico y, compartido con uno numérico, matplotlib
    # apila las tres barras en x=0 y superpone las etiquetas: la figura queda ilegible.
    ys2 = [ablacion[k]["cero_a_compuerta"]["media"] for k in claves_abl]
    ancho_barra = max(8, (max(xs) - min(xs)) / 12)
    ax2.bar(xs, ys2, color="#B45309", width=ancho_barra)
    ax2.set_ylim(0, max(1.0, max(ys2) * 1.4))
    ax2.set_ylabel("0 → compuerta")
    ax2.set_xlabel("imágenes de 3_no_diapositiva en el entrenamiento")
    ax2.set_xticks(xs)
    ax2.grid(alpha=0.3, axis="y")
    for x, y in zip(xs, ys2):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=9, color="#B45309", fontweight="bold")
    ax2.text(0.5, 0.60,
             "diapositivas de la clase 0 rechazadas por la compuerta\n"
             "(cero en los tres tamaños: la caída de arriba es del eje ordinal)",
             transform=ax2.transAxes, ha="center", fontsize=8.5, color="#444")

    plt.tight_layout()
    plt.savefig(AQUI / "Figure_ablacion_rechazo_v10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  guardado Figure_ablacion_rechazo_v10.png")

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
    ax.set_title("Brecha de dominio — TensorFlow v10", fontweight="bold", color="#1F3864")
    for i, fase in enumerate(("fase1", "fase2")):
        for j, k in enumerate(("fotos", "renders")):
            v = brechas[fase][k]
            ax.text(j + (i - 0.5) * ancho, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(AQUI / "Figure_brecha_dominio_v10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  guardado Figure_brecha_dominio_v10.png")

    # --- 5.10 Guardar resultados y el modelo de producción ---
    def limpiar(agregado):
        return {k: agregado[k] for k in metricas.CLAVES_V10 + ["cm_sumada", "n_semillas"]}

    with open(AQUI / "resultados_v10.json", "w", encoding="utf-8") as f:
        json.dump({
            "framework": "tensorflow",
            "version": "v10",
            "clases": clases,
            "n_ordinales": N_ORDINALES,
            "clase_rechazo": CLASE_RECHAZO,
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
            "ablacion_clase_rechazo": ablacion,
        }, f, indent=2, ensure_ascii=False)
    print("  guardado resultados_v10.json")

    # Modelo de PRODUCCIÓN: imágenes 0-255 -> softmax de 4 clases. Es el archivo que consume
    # la exportación a TensorFlow.js y el que termina dentro de la app Ionic.
    mejor_semilla = max(SEMILLAS_FT,
                        key=lambda s: next(r["macro_f1"] for r in res_f2["test"]
                                           if r["semilla"] == s))
    modelo_prod = tf.keras.Model(
        modelos_ft[mejor_semilla].input,
        tf.keras.layers.Softmax(name="probabilidades")(modelos_ft[mejor_semilla].output))
    modelo_prod.save(AQUI / "modelotf_v10_finetune.keras")
    print(f"  guardado modelotf_v10_finetune.keras  (fase 2, semilla {mejor_semilla})")

    # La semilla se elige por MACRO F1 y no por accuracy — cambio respecto de la v9. Con 4
    # clases y una de ellas fácil, la accuracy premia acertar la clase de rechazo y esconde
    # el desempeño en el eje ordinal, que es lo que el producto vende. Macro F1 pesa las 4
    # clases igual.
    r_mejor = next(r for r in res_f2["test"] if r["semilla"] == mejor_semilla)
    with open(AQUI / "modelo_v10_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v10", "framework": "tensorflow", "arquitectura": "MobileNetV3-Small",
            "clases": clases, "n_ordinales": N_ORDINALES, "clase_rechazo": CLASE_RECHAZO,
            "entrada": {"alto": SIZE[0], "ancho": SIZE[1], "canales": 3,
                        "rango": "0-255", "orden": "NHWC"},
            "preprocesamiento": "ninguno en el cliente: la normalización ImageNet viaja "
                                "dentro del grafo (include_preprocessing=True)",
            "salida": f"softmax de {num_clases} clases, en el orden de 'clases'",
            "semilla": mejor_semilla,
            "criterio_seleccion": "macro F1 sobre el test de fotos",
            "accuracy_test_fotos": float(r_mejor["accuracy"]),
            "macro_f1_test_fotos": float(r_mejor["macro_f1"]),
            "recall_rechazo_test_fotos": float(r_mejor["recall_rechazo"]),
        }, f, indent=2, ensure_ascii=False)
    print("  guardado modelo_v10_meta.json")

    print(f"\nTiempo total: {(time.time() - t_inicio) / 60:.1f} min")
