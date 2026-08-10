"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 8  (DESTILACIÓN EN CADENA vs. ENTRENAMIENTO DIRECTO)

Propósito experimental:
  Todo el proyecto avanzó agregando UNA técnica por versión: v1 CNN pelada, v2 le sumó
  augmentation, v3 convergencia, v4 transfer learning, v6/v7 hiperparámetros buscados.
  El modelo final (v7) recibe todas esas técnicas DE GOLPE: se entrena una sola vez con
  la receta completa.

  La v8 pregunta otra cosa: ¿y si en vez de entrenar una sola vez con todo, se fuera
  TRANSFIRIENDO el conocimiento de versión en versión? Entrenar la v1, destilarla hacia
  un estudiante que además tiene la augmentation de la v2, destilar ESO hacia uno que
  además converge como la v3, y así hasta el final.

  Se comparan DOS BRAZOS que terminan en la MISMA arquitectura y se evalúan sobre el
  MISMO test apartado de la v7:

    BRAZO A — DIRECTO   : MobileNetV3-Small congelada + cabeza (hiperparámetros de la
                          v7), entrenada de una con cross-entropy. Es la receta v7.

    BRAZO B — CADENA    : cinco eslabones encadenados, cada uno destilando del anterior
                          y agregando lo que aportaba la versión correspondiente:
                            E1  CNN desde cero, sin nada          (como v1)  <- profesor raíz
                            E2  CNN + augmentation                (como v2)  <- destila de E1
                            E3  CNN + augmentation + converger    (como v3)  <- destila de E2
                            E4  MobileNetV3 + cabeza tipo v4      (como v4)  <- destila de E3
                            E5  cabeza con hiperparámetros v7     (como v7)  <- destila de E4

  El eslabón E4 es el interesante: el profesor es una CNN propia de 180x180 y el
  estudiante una MobileNetV3 de 224x224. Arquitecturas distintas, MISMO espacio de
  salida (3 clases) => es exactamente el Escenario 1 del script del profe
  (destilacion_cross_arch.py): "mismo vocabulario, KL divergence sobre los logits".
  Allá el vocabulario compartido eran los 50.257 tokens de GPT-2; acá son nuestras 3
  clases. La mecánica es idéntica.

LA HIPÓTESIS, DECLARADA ANTES DE CORRER (esto importa):
  No se espera que la cadena gane en accuracy. Arranca desde un profesor deliberadamente
  malo (la v1 sobreajusta a propósito) y cada eslabón hereda los sesgos del anterior. Lo
  que SÍ se espera, y es el motivo real de destilar, es mejor CALIBRACIÓN: los targets
  blandos del profesor transmiten "cuánto duda" y no solo "qué contesta", así que el
  estudiante suele terminar menos exceso-confiado. Por eso la v7 agregó ECE: sin esa
  métrica este efecto sería invisible y la v8 parecería un fracaso.

EL PROBLEMA DEL PROFESOR SATURADO (y cómo se resuelve acá):
  Un profesor sobreajustado no enseña casi nada. La v1 llega a train_acc ~1.0: sobre las
  imágenes que memorizó, su softmax es prácticamente one-hot, el término KL degenera en
  cross-entropy y la destilación no transfiere "conocimiento oscuro" ninguno. El
  conocimiento oscuro solo existe donde el profesor DUDA.

  Solución: destilar sobre VISTAS AUMENTADAS. El profesor nunca vio esas rotaciones y
  cambios de brillo, así que ahí sí produce una distribución genuinamente blanda. Y
  encaja con el relato: lo aumentado es justamente lo que aporta la v2.

  Para que además siga siendo barato, las vistas aumentadas se generan UNA sola vez con
  semilla fija (no se re-sortean por época). Eso permite pre-computar los embeddings de
  MobileNet y los logits del profesor sobre TODAS las vistas, y que los eslabones E4/E5
  entrenen en segundos. Se pierde algo de diversidad de augmentation a cambio de que el
  experimento corra en minutos y sea 100% reproducible: es un tradeoff consciente.

CONTRASTE DE FRAMEWORKS (espejo de PyTorch/v8/08_scripts.py):
  La destilación es el contraste MÁS FUERTE de todo el proyecto, por dos razones:

  1. LOGITS vs PROBABILIDADES. Para aplicar temperatura hay que dividir los LOGITS por T.
     En PyTorch el modelo ya devuelve logits crudos y no hay nada que hacer. En Keras la
     costumbre del proyecto (v1-v7) era terminar la cabeza en softmax, y una vez aplicado
     el softmax la temperatura ya no se puede aplicar. Por eso acá las cabezas se
     construyen SIN activación final (devuelven logits) y el softmax se aplica a mano en
     `probabilidades()`. Es un cambio estructural que la destilación OBLIGA a hacer.

  2. CÓMO SE LE PASAN LOS LOGITS DEL PROFESOR A LA PÉRDIDA. En PyTorch la pérdida es una
     función común y se le pasa lo que uno quiera. En Keras, `loss(y_true, y_pred)` solo
     recibe esos dos tensores, así que hay que CONTRABANDEAR los logits del profesor
     dentro de `y_true`: se arma un y_true de ancho 1+3 = [etiqueta, logits_profesor] y
     la pérdida lo desempaqueta. Es el idiom estándar de KD en Keras y una diferencia de
     diseño real entre los dos frameworks.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silenciar logs de TF (solo errores)

import numpy as np
import tensorflow as tf

# Módulos COMPARTIDOS con PyTorch (numpy puro, sin dependencias de framework).
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402
from particion_datos import cargar_particion, conteos, rutas_y_etiquetas  # noqa: E402

tf.get_logger().setLevel("ERROR")

# --- 1. Parámetros ---
DATA_DIR = RAIZ / "dataset"
if not DATA_DIR.is_dir():
    raise SystemExit(f"No existe la carpeta de datos: {DATA_DIR}")

SIZE_CNN = (180, 180)      # entrada de la CNN propia (v1-v3)
SIZE_MNET = (224, 224)     # entrada de MobileNetV3 (v4+)
BATCH_SIZE = 32
SEED = 123

N_VISTAS_AUG = 2           # vistas aumentadas por imagen (+1 limpia = 3 por imagen)
SEED_VISTAS = 999          # semilla propia de las vistas: fija para TODAS las semillas
                           # de entrenamiento, así los dos brazos ven EXACTAMENTE los
                           # mismos datos y la única diferencia es cómo se entrena.

# Hiperparámetros de destilación. Son los del script del profe (destilacion_cross_arch.py,
# Escenario 1): TEMPERATURA=4.0, ALPHA=0.7. Se FIJAN en vez de buscarlos con Optuna a
# propósito: si el brazo B pudiera buscar dos hiperparámetros extra que el brazo A no
# tiene, ganaría por tener más búsqueda, no por destilar.
TEMPERATURA = 4.0
ALPHA = 0.7

SEMILLAS = [123, 7, 42, 2024, 31]   # las MISMAS de la v7, para poder comparar

# Épocas por eslabón. Fijas y declaradas de antemano: dentro de la cadena no hay un
# conjunto de validación con el que hacer early stopping sin quitarle datos a alguno de
# los dos brazos, así que el "converger" de la v3 se representa con un presupuesto de
# épocas mayor (40) frente al recorte arbitrario de la v1 (10).
EPOCAS = {"E1": 10, "E2": 30, "E3": 40, "E4": 30}
LR_CNN = 1e-3


# --- 2. Vistas: 1 limpia + N aumentadas por imagen, generadas UNA vez ---
particion = cargar_particion()
class_names = particion["clases"]
print("Orden de clases:", class_names)
num_classes = len(class_names)

rutas_dev, y_dev_lista = rutas_y_etiquetas(particion, "desarrollo")
rutas_test, y_test_lista = rutas_y_etiquetas(particion, "test")
print(f"\nPartición canónica (huella {particion['meta']['huella_dataset']}):")
print(f"  desarrollo : {len(rutas_dev):>3} imgs  {conteos(particion, rutas_dev)}")
print(f"  test       : {len(rutas_test):>3} imgs  {conteos(particion, rutas_test)}   "
      f"<- el MISMO test apartado de la v7")

# Augmentation domain-aware, la MISMA de v2/v3/v4 en su versión Keras.
# CONTRASTE: acá son CAPAS de Keras; en PyTorch son transforms del DataLoader.
aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05, seed=SEED_VISTAS),      # ±18°
    tf.keras.layers.RandomZoom(0.1, seed=SEED_VISTAS),
    tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEED_VISTAS),
    tf.keras.layers.RandomContrast(0.2, seed=SEED_VISTAS),
], name="augmentation")


def construir_extractor() -> tf.keras.Model:
    """MobileNetV3-Small congelada + GAP -> 576-d. trainable=False ya pone las BN en
    modo inferencia (en PyTorch hay que forzar eval() a mano: gotcha de la v4)."""
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*SIZE_MNET, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(*SIZE_MNET, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, x, name="extractor_features")


extractor = construir_extractor()
EMB_DIM = extractor.output_shape[-1]


def preparar_vistas(rutas, etiquetas, n_aug, semilla):
    """Genera las vistas y devuelve (imgs 180px uint8, embeddings 576-d, etiquetas).

    Cada imagen produce 1 vista LIMPIA + n_aug vistas AUMENTADAS. La augmentation se
    aplica sobre la imagen original y recién después se hacen los DOS resize (180 para
    la CNN, 224 para MobileNet), así los dos tamaños comparten la misma instancia de
    augmentation: el profesor CNN y el estudiante MobileNet ven la MISMA vista, que es
    lo que exige la destilación (si vieran vistas distintas, los logits del profesor no
    corresponderían a la entrada del estudiante).

    Las imágenes de 180px se guardan en uint8 y se normalizan dentro del modelo
    (capa Rescaling), igual que en v1-v3.
    """
    tf.keras.utils.set_random_seed(semilla)
    imgs180, embs, ys = [], [], []
    buffer_mnet = []

    def volcar_buffer():
        if not buffer_mnet:
            return
        lote = tf.stack(buffer_mnet)
        embs.append(extractor.predict(lote, verbose=0))
        buffer_mnet.clear()

    for ruta, etiqueta in zip(rutas, etiquetas):
        crudo = tf.io.decode_image(tf.io.read_file(str(DATA_DIR / ruta)),
                                   channels=3, expand_animations=False)
        original = tf.cast(crudo, tf.float32)
        for v in range(n_aug + 1):
            img = original if v == 0 else aug(tf.expand_dims(original, 0),
                                              training=True)[0]
            img = tf.clip_by_value(img, 0.0, 255.0)
            imgs180.append(tf.cast(tf.image.resize(img, SIZE_CNN), tf.uint8).numpy())
            # 0-255 crudo: MobileNetV3Small trae su propio preprocesamiento ImageNet
            # dentro (include_preprocessing=True), igual que en v4/v6/v7.
            buffer_mnet.append(tf.image.resize(img, SIZE_MNET))
            ys.append(etiqueta)
            if len(buffer_mnet) >= BATCH_SIZE:
                volcar_buffer()
    volcar_buffer()

    return (np.stack(imgs180), np.concatenate(embs),
            np.array(ys, dtype=np.int32))


print(f"\nGenerando vistas (1 limpia + {N_VISTAS_AUG} aumentadas por imagen, "
      f"semilla fija {SEED_VISTAS})...")
t0 = time.time()
X180_dev, XE_dev, Y_dev = preparar_vistas(rutas_dev, y_dev_lista, N_VISTAS_AUG, SEED_VISTAS)
# El TEST no se aumenta: se evalúa sobre la imagen limpia, como en toda versión anterior.
X180_test, XE_test, Y_test = preparar_vistas(rutas_test, y_test_lista, 0, SEED_VISTAS)
print(f"  dev : {X180_dev.shape} a 180px  ·  {XE_dev.shape} embeddings")
print(f"  test: {X180_test.shape} a 180px  ·  {XE_test.shape} embeddings")
print(f"  ({time.time() - t0:.0f}s)")

# Máscara de las vistas LIMPIAS de dev (el eslabón E1 no usa augmentation, como la v1).
mascara_limpias = np.zeros(len(Y_dev), dtype=bool)
mascara_limpias[::N_VISTAS_AUG + 1] = True


# --- 3. Modelos ---
# TODAS las cabezas devuelven LOGITS (sin softmax final). Es lo que permite dividir por
# la temperatura. Ver el contraste de frameworks en el encabezado.
def baseline_cnn() -> tf.keras.Sequential:
    """La MISMA CNN de v1/v2/v3, pero devolviendo logits en vez de softmax.

    Rescaling(1/255) -> Conv2D(16) -> Pool -> Conv2D(32) -> Pool -> Flatten ->
    Dense(64) -> Dense(3). Padding 'valid' por defecto: 180 -> 178 -> 89 -> 87 -> 43.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*SIZE_CNN, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes),  # LOGITS
    ])


def cabeza_v4() -> tf.keras.Sequential:
    """Cabeza tipo v4: réplica del classifier de MobileNetV3-Small sobre el embedding."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(EMB_DIM,)),
        tf.keras.layers.Dense(1024, activation="hard_swish"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes),  # LOGITS
    ])


def cabeza_v7(params) -> tf.keras.Sequential:
    """Cabeza con los hiperparámetros que Optuna eligió en la v7 (por CV, sin ver el test)."""
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(EMB_DIM,))])
    for _ in range(params["n_capas"]):
        m.add(tf.keras.layers.Dense(params["units"], activation="relu"))
        m.add(tf.keras.layers.Dropout(params["dropout"]))
    m.add(tf.keras.layers.Dense(num_classes))  # LOGITS
    return m


def optimizador_v7(params):
    lr = params["lr"]
    return {"adam": tf.keras.optimizers.Adam(lr),
            "rmsprop": tf.keras.optimizers.RMSprop(lr),
            "sgd": tf.keras.optimizers.SGD(lr, momentum=0.9)}[params["optimizer"]]


# --- 4. LA PÉRDIDA DE DESTILACIÓN (el corazón de la v8) ---
def hacer_perdida_destilacion(T: float, alpha: float):
    """Devuelve la pérdida alpha*KL(T)*T^2 + (1-alpha)*CE, lista para compile().

    Calcada del `perdida_kl` del script del profe (destilacion_cross_arch.py, Escenario 1),
    adaptada de un vocabulario de 50.257 tokens a nuestras 3 clases.

    EL TRUCO DE KERAS: compile(loss=fn) solo le pasa a fn (y_true, y_pred). Los logits
    del profesor no entran por ningún lado. La solución idiomática es CONTRABANDEARLOS
    dentro de y_true: se entrena con un y_true de ancho 1+num_classes, donde la columna 0
    es la etiqueta y el resto son los logits del profesor. La pérdida lo desempaqueta.
    En PyTorch nada de esto hace falta: la pérdida es una función común.

    Por qué cada pieza:
      - Dividir por T APLASTA la distribución y saca a la luz las probabilidades chicas.
        Si el profesor dice [0.98, 0.02, 0.00], con T=4 dice algo como [0.55, 0.28, 0.17]:
        ahí aparece que "esta diapo, si no fuera 0, sería más 1 que 2". Eso es el
        conocimiento oscuro, y es lo que una etiqueta dura nunca transmite.
      - Multiplicar por T^2 compensa que los gradientes del término KL escalan como 1/T^2.
      - El término CE mantiene al estudiante anclado a las etiquetas reales.
    """
    def perdida(y_true, logits_est):
        etiquetas = tf.cast(y_true[:, 0], tf.int32)
        logits_prof = y_true[:, 1:]
        # KL(profesor || estudiante), sumada sobre clases y promediada sobre el batch:
        # es exactamente el reduction="batchmean" de F.kl_div en PyTorch.
        log_p_est = tf.nn.log_softmax(logits_est / T, axis=1)
        log_p_prof = tf.nn.log_softmax(logits_prof / T, axis=1)
        p_prof = tf.exp(log_p_prof)
        kl = tf.reduce_mean(tf.reduce_sum(p_prof * (log_p_prof - log_p_est), axis=1))
        kl = kl * (T ** 2)
        ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            etiquetas, logits_est, from_logits=True))
        return alpha * kl + (1 - alpha) * ce
    return perdida


# --- 5. Entrenamiento de un eslabón ---
def entrenar_eslabon(modelo, X, Y, logits_prof, epochs, optimizador, semilla,
                     T=TEMPERATURA, alpha=ALPHA):
    """Entrena un eslabón. Si logits_prof es None entrena con cross-entropy pura.

    `logits_prof` son los logits del profesor YA PRE-COMPUTADOS sobre las mismas vistas,
    alineados por índice con X. Pre-computarlos (en vez de correr el profesor dentro del
    bucle) es la razón de que la cadena entera corra en minutos: el profesor está
    congelado, así que sus logits no cambian entre épocas.
    """
    tf.keras.utils.set_random_seed(semilla)
    if logits_prof is None:
        modelo.compile(optimizer=optimizador,
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        modelo.fit(X, Y, epochs=epochs, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
    else:
        # Empaquetado [etiqueta | logits_profesor] -> ver hacer_perdida_destilacion.
        y_packed = np.concatenate([Y.reshape(-1, 1).astype(np.float32),
                                   logits_prof.astype(np.float32)], axis=1)
        modelo.compile(optimizer=optimizador, loss=hacer_perdida_destilacion(T, alpha))
        modelo.fit(X, y_packed, epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
                   shuffle=True)
    return modelo


def logits_de(modelo, X) -> np.ndarray:
    """Logits del modelo sobre todas las vistas (las cabezas ya devuelven logits)."""
    return modelo.predict(X, verbose=0)


def probabilidades(modelo, X) -> np.ndarray:
    """Vector COMPLETO de probabilidades softmax (regla del proyecto desde la v2).

    CONTRASTE: hasta la v7 la cabeza de Keras terminaba en softmax y predict() ya
    devolvía probabilidades. Acá devuelve LOGITS (para poder aplicar temperatura), así
    que el softmax se aplica a mano — igual que se venía haciendo en PyTorch.
    """
    return tf.nn.softmax(logits_de(modelo, X), axis=1).numpy()


# --- 6. Los dos brazos ---
def brazo_directo(params, epochs, semilla):
    """BRAZO A — la receta de la v7 entrenada de una: cabeza Optuna + cross-entropy.

    Usa las MISMAS vistas que la cadena (limpias + aumentadas) para que la única
    diferencia entre los dos brazos sea el procedimiento de entrenamiento, no los datos.
    Por eso su número puede diferir un poco del de la v7, que entrenaba solo sobre las
    vistas limpias.
    """
    return entrenar_eslabon(cabeza_v7(params), XE_dev, Y_dev, None, epochs,
                            optimizador_v7(params), semilla)


def brazo_cadena(params, epochs_final, semilla, verbose=True):
    """BRAZO B — los cinco eslabones, cada uno destilando del anterior.

    Regla clave: el estudiante se RE-INICIALIZA en cada eslabón (pesos random en E1-E3,
    pesos de ImageNet en E4-E5). Si heredara los pesos del profesor esto sería
    fine-tuning con otro nombre, no destilación.
    """
    historial = []

    def registrar(nombre, modelo, X_ev):
        prob = probabilidades(modelo, X_ev)
        r = metricas.resumen_completo(Y_test, prob, class_names)
        historial.append({"eslabon": nombre, "accuracy": r["accuracy"], "qwk": r["qwk"],
                          "ece": r["ece"], "extremos_total": r["extremos_total"]})
        if verbose:
            print(f"    {nombre}: test_acc={r['accuracy']:.4f}  QWK={r['qwk']:.4f}  "
                  f"ECE={r['ece']:.4f}  0<->2={r['extremos_total']}")

    # --- E1: CNN desde cero, SOLO vistas limpias, sin profesor (es la raíz) ---
    e1 = entrenar_eslabon(baseline_cnn(), X180_dev[mascara_limpias],
                          Y_dev[mascara_limpias], None, EPOCAS["E1"],
                          tf.keras.optimizers.Adam(LR_CNN), semilla)
    registrar("E1 (v1: CNN pelada)", e1, X180_test)

    # --- E2: CNN re-inicializada + augmentation, destilando de E1 ---
    # Acá es donde el profesor sobreajustado SÍ enseña: E1 memorizó las vistas limpias,
    # pero nunca vio las aumentadas, así que sobre ellas produce distribuciones blandas.
    e2 = entrenar_eslabon(baseline_cnn(), X180_dev, Y_dev, logits_de(e1, X180_dev),
                          EPOCAS["E2"], tf.keras.optimizers.Adam(LR_CNN), semilla)
    registrar("E2 (v2: + augmentation)", e2, X180_test)

    # --- E3: CNN re-inicializada + más épocas (converger), destilando de E2 ---
    e3 = entrenar_eslabon(baseline_cnn(), X180_dev, Y_dev, logits_de(e2, X180_dev),
                          EPOCAS["E3"], tf.keras.optimizers.Adam(LR_CNN), semilla)
    registrar("E3 (v3: + converger)", e3, X180_test)

    # --- E4: CROSS-ARCHITECTURE. Profesor CNN 180px -> estudiante MobileNetV3 224px ---
    # ESTE es el Escenario 1 del script del profe: arquitecturas distintas, mismo espacio
    # de salida (3 clases), así que los logits se comparan directamente con KL.
    e4 = entrenar_eslabon(cabeza_v4(), XE_dev, Y_dev, logits_de(e3, X180_dev),
                          EPOCAS["E4"], tf.keras.optimizers.Adam(1e-3), semilla)
    registrar("E4 (v4: -> MobileNetV3)", e4, XE_test)

    # --- E5: cabeza con los hiperparámetros de la v7, destilando de E4 ---
    logits_e4 = logits_de(e4, XE_dev)
    e5 = entrenar_eslabon(cabeza_v7(params), XE_dev, Y_dev, logits_e4, epochs_final,
                          optimizador_v7(params), semilla)
    registrar("E5 (v7: cabeza Optuna)", e5, XE_test)

    # Se devuelven también los logits de E4: el barrido de T/alpha los reutiliza en vez
    # de reconstruir la cadena entera (ahorra una cadena completa de cómputo).
    return e5, historial, logits_e4


# --- 7. Barrido de sensibilidad de T y alpha (sobre CV, NUNCA sobre el test) ---
def barrido_T_alpha(params, epochs_final, semilla, logits_prof):
    """¿Cuánto dependen los resultados de la temperatura y del peso del término KL?

    Se mide sobre VALIDACIÓN CRUZADA dentro de desarrollo, no sobre el test: el barrido
    es información, no un mecanismo de selección. Si se eligiera T y alpha mirando el
    test, se repetiría exactamente el pecado que la v7 vino a corregir.
    """
    n_folds = particion["meta"]["n_folds"]
    pos = {r: i for i, r in enumerate(rutas_dev)}
    # Cada imagen aporta N_VISTAS_AUG+1 vistas CONSECUTIVAS, así que la vista limpia de
    # la imagen i está en la posición i*(N_VISTAS_AUG+1).
    idx_folds_vistas = [
        np.array([pos[r] * (N_VISTAS_AUG + 1)
                  for r in rutas_y_etiquetas(particion, "validacion", fold=k)[0]])
        for k in range(n_folds)
    ]

    resultados = []
    for T in [2.0, 4.0, 8.0]:
        for a in [0.0, 0.3, 0.5, 0.7, 0.9]:
            accs = []
            for k in range(n_folds):
                idx_va = idx_folds_vistas[k]
                mask = np.ones(len(Y_dev), dtype=bool)
                # Se excluyen TODAS las vistas de las imágenes de validación, no solo la
                # limpia: si una vista aumentada de una imagen quedara en train y su vista
                # limpia en validación, habría fuga (es la misma diapositiva).
                for b in idx_va:
                    mask[b:b + N_VISTAS_AUG + 1] = False
                cab = entrenar_eslabon(cabeza_v7(params), XE_dev[mask], Y_dev[mask],
                                       logits_prof[mask], epochs_final,
                                       optimizador_v7(params), semilla, T, a)
                prob = probabilidades(cab, XE_dev[idx_va])
                accs.append(float((prob.argmax(1) == Y_dev[idx_va]).mean()))
            resultados.append({"T": T, "alpha": a, "cv_accuracy": float(np.mean(accs))})
            print(f"    T={T:<4} alpha={a:<4}  cv_accuracy={np.mean(accs):.4f}")
    return resultados


# --- 8. Main ---
if __name__ == "__main__":
    AQUI = Path(__file__).resolve().parent

    with open(RAIZ / "TensorFlow/v7/mejores_hiperparametros.json", encoding="utf-8") as f:
        cfg_v7 = json.load(f)
    PARAMS = cfg_v7["params"]
    EPOCHS_FINAL = cfg_v7["epochs_refit"]
    print(f"\nHiperparámetros heredados de la v7 (elegidos por CV, sin ver el test):")
    print(f"  {PARAMS}   ·   épocas del eslabón final: {EPOCHS_FINAL}")
    print(f"Destilación: T={TEMPERATURA} · alpha(KL)={ALPHA}  "
          f"(los valores del script del profe)")

    # --- Los dos brazos, semilla por semilla ---
    print("\n" + "=" * 78)
    print(f"  BRAZO A (directo)  vs  BRAZO B (cadena)  ·  {len(SEMILLAS)} semillas")
    print("=" * 78)

    res_a, res_b, historiales = [], [], []
    modelos_b, logits_e4 = [], None
    for semilla in SEMILLAS:
        print(f"\n  --- semilla {semilla} ---")
        t0 = time.time()

        modelo_a = brazo_directo(PARAMS, EPOCHS_FINAL, semilla)
        prob_a = probabilidades(modelo_a, XE_test)
        ra = metricas.resumen_completo(Y_test, prob_a, class_names)
        ra["semilla"] = semilla; ra["y_prob"] = prob_a
        res_a.append(ra)
        print(f"    BRAZO A (directo): test_acc={ra['accuracy']:.4f}  "
              f"QWK={ra['qwk']:.4f}  ECE={ra['ece']:.4f}  0<->2={ra['extremos_total']}")

        modelo_b, historial, logits_e4 = brazo_cadena(PARAMS, EPOCHS_FINAL, semilla)
        prob_b = probabilidades(modelo_b, XE_test)
        rb = metricas.resumen_completo(Y_test, prob_b, class_names)
        rb["semilla"] = semilla; rb["y_prob"] = prob_b
        res_b.append(rb)
        modelos_b.append(modelo_b)
        historiales.append(historial)
        print(f"    ({time.time() - t0:.0f}s)")

    # --- Agregado y comparación ---
    agr_a = metricas.agregar_semillas(res_a)
    agr_b = metricas.agregar_semillas(res_b)

    print()
    metricas.imprimir_agregado(agr_a, "BRAZO A — DIRECTO (TensorFlow v8)")
    print()
    metricas.imprimir_agregado(agr_b, "BRAZO B — CADENA DE DESTILACIÓN (TensorFlow v8)")

    print("\n" + "=" * 78)
    print("  COMPARACIÓN A/B  —  ¿la cadena gana, pierde o empata?")
    print("=" * 78)
    print(f"  {'métrica':<24}{'A directo':>13}{'B cadena':>13}{'B - A':>10}  veredicto")
    print("  " + "-" * 74)
    veredictos = {}
    for clave in metricas.CLAVES_ESCALARES:
        ma, sa = agr_a[clave]["media"], agr_a[clave]["std"]
        mb, sb = agr_b[clave]["media"], agr_b[clave]["std"]
        delta = mb - ma
        # Criterio declarado de antemano: una diferencia solo cuenta si supera la suma
        # de los desvíos de los dos brazos. Con 63 imágenes de test, cualquier cosa por
        # debajo de eso es ruido y decirlo es más honesto que festejar decimales.
        umbral = sa + sb
        if abs(delta) <= umbral:
            v = "ruido"
        else:
            mejor_b = (delta > 0) == metricas.MAS_ES_MEJOR[clave]
            v = "GANA CADENA" if mejor_b else "gana directo"
        veredictos[clave] = v
        print(f"  {metricas.ETIQUETAS[clave]:<24}{ma:>13.4f}{mb:>13.4f}{delta:>+10.4f}  {v}")
    print("  " + "-" * 74)
    print("  'ruido' = |B-A| menor o igual que la suma de los desvíos de ambos brazos.")

    # --- Progresión eslabón por eslabón (promediada entre semillas) ---
    print("\n" + "=" * 78)
    print("  PROGRESIÓN DE LA CADENA, ESLABÓN POR ESLABÓN (media entre semillas)")
    print("=" * 78)
    nombres_esl = [h["eslabon"] for h in historiales[0]]
    print(f"  {'eslabón':<28}{'test_acc':>10}{'QWK':>9}{'ECE':>9}{'0<->2':>8}")
    print("  " + "-" * 62)
    prog = {}
    for i, nombre in enumerate(nombres_esl):
        acc = np.mean([h[i]["accuracy"] for h in historiales])
        qw = np.mean([h[i]["qwk"] for h in historiales])
        ec = np.mean([h[i]["ece"] for h in historiales])
        ex = np.mean([h[i]["extremos_total"] for h in historiales])
        prog[nombre] = {"accuracy": float(acc), "qwk": float(qw), "ece": float(ec),
                        "extremos_total": float(ex)}
        print(f"  {nombre:<28}{acc:>10.4f}{qw:>9.4f}{ec:>9.4f}{ex:>8.1f}")

    # --- Barrido de sensibilidad (sobre CV) ---
    print("\n" + "=" * 78)
    print("  SENSIBILIDAD A T Y ALPHA (medida sobre CV de desarrollo, NUNCA sobre test)")
    print("=" * 78)
    print("  alpha=0.0 equivale a NO destilar: es la línea base dentro del mismo barrido.")
    barrido = barrido_T_alpha(PARAMS, EPOCHS_FINAL, SEMILLAS[0], logits_e4)
    mejor = max(barrido, key=lambda r: r["cv_accuracy"])
    print(f"\n  mejor combinación por CV: T={mejor['T']} alpha={mejor['alpha']} "
          f"-> {mejor['cv_accuracy']:.4f}")
    print(f"  (NO se usa para elegir el modelo final: la v8 se reporta con los valores")
    print(f"   fijos del script del profe, T={TEMPERATURA} alpha={ALPHA}.)")

    # --- Matrices y figuras ---
    cm_a = np.array(agr_a["cm_sumada"])
    cm_b = np.array(agr_b["cm_sumada"])
    print(f"\nMatriz ACUMULADA — BRAZO A (directo), {len(SEMILLAS)} semillas:")
    print(cm_a)
    metricas.reporte_por_clase(cm_a, class_names)
    print(f"\nMatriz ACUMULADA — BRAZO B (cadena), {len(SEMILLAS)} semillas:")
    print(cm_b)
    metricas.reporte_por_clase(cm_b, class_names)

    print("\nGenerando figuras...")
    metricas.plot_matriz(cm_a, class_names,
                         f"Matriz v8 — BRAZO A directo (test, {len(SEMILLAS)} semillas)",
                         AQUI / "Figure_matriz_directo_v8.png")
    metricas.plot_matriz(cm_b, class_names,
                         f"Matriz v8 — BRAZO B cadena (test, {len(SEMILLAS)} semillas)",
                         AQUI / "Figure_matriz_cadena_v8.png")
    metricas.plot_calibracion(Y_test, res_a[0]["y_prob"],
                              f"Calibración — BRAZO A directo (semilla {SEMILLAS[0]})",
                              AQUI / "Figure_calibracion_directo_v8.png")
    metricas.plot_calibracion(Y_test, res_b[0]["y_prob"],
                              f"Calibración — BRAZO B cadena (semilla {SEMILLAS[0]})",
                              AQUI / "Figure_calibracion_cadena_v8.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Progresión de la cadena: accuracy y ECE eslabón a eslabón.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    xs = range(len(nombres_esl))
    etiquetas_cortas = [n.split(" ")[0] for n in nombres_esl]
    ax1.plot(xs, [prog[n]["accuracy"] for n in nombres_esl], "o-", lw=2, ms=8,
             color="#2E75B6", label="cadena")
    ax1.axhline(agr_a["accuracy"]["media"], ls="--", lw=2, color="#1F3864",
                label=f"brazo A directo ({agr_a['accuracy']['media']:.3f})")
    ax1.set_xticks(list(xs)); ax1.set_xticklabels(etiquetas_cortas)
    ax1.set_ylabel("accuracy en test"); ax1.set_title("Accuracy eslabón a eslabón")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(xs, [prog[n]["ece"] for n in nombres_esl], "o-", lw=2, ms=8,
             color="#2E75B6", label="cadena")
    ax2.axhline(agr_a["ece"]["media"], ls="--", lw=2, color="#1F3864",
                label=f"brazo A directo ({agr_a['ece']['media']:.3f})")
    ax2.set_xticks(list(xs)); ax2.set_xticklabels(etiquetas_cortas)
    ax2.set_ylabel("ECE (más bajo = mejor)"); ax2.set_title("Calibración eslabón a eslabón")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle("Cadena de destilación — TensorFlow v8 (media de 5 semillas)",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(AQUI / "Figure_progresion_v8.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  guardado Figure_progresion_v8.png")

    # Barrido T/alpha.
    fig, ax = plt.subplots(figsize=(8, 5))
    for T in sorted({r["T"] for r in barrido}):
        pts = [r for r in barrido if r["T"] == T]
        ax.plot([p["alpha"] for p in pts], [p["cv_accuracy"] for p in pts], "o-",
                lw=2, ms=7, label=f"T={T}")
    ax.set_xlabel("alpha (peso del término KL) — 0.0 = sin destilar")
    ax.set_ylabel("accuracy de CV en desarrollo")
    ax.set_title("Sensibilidad a temperatura y peso de la destilación\n"
                 "medido sobre CV, nunca sobre el test",
                 fontweight="bold", color="#1F3864")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(AQUI / "Figure_barrido_T_alpha_v8.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  guardado Figure_barrido_T_alpha_v8.png")

    metricas.plot_semillas(res_b, ["accuracy", "macro_f1", "qwk", "ece"],
                           "Dispersión entre semillas — BRAZO B cadena (TensorFlow v8)",
                           AQUI / "Figure_semillas_cadena_v8.png")

    # --- Guardar resultados ---
    with open(AQUI / "resultados_v8.json", "w", encoding="utf-8") as f:
        json.dump({
            "framework": "tensorflow",
            "particion": particion["meta"],
            "semillas": SEMILLAS,
            "params_heredados_v7": PARAMS,
            "destilacion": {"temperatura": TEMPERATURA, "alpha": ALPHA,
                            "vistas_aumentadas": N_VISTAS_AUG, "seed_vistas": SEED_VISTAS},
            "epocas_eslabones": EPOCAS,
            "brazo_a_directo": {k: agr_a[k] for k in metricas.CLAVES_ESCALARES},
            "brazo_b_cadena": {k: agr_b[k] for k in metricas.CLAVES_ESCALARES},
            "veredictos": veredictos,
            "cm_a": agr_a["cm_sumada"],
            "cm_b": agr_b["cm_sumada"],
            "progresion_cadena": prog,
            "barrido_T_alpha": barrido,
        }, f, indent=2, ensure_ascii=False)
    print("  guardado resultados_v8.json")

    # Modelo de inferencia de punta a punta: imágenes 0-255 -> backbone -> cabeza ->
    # softmax. Se agrega el softmax explícito porque la cabeza devuelve logits.
    inputs = tf.keras.Input(shape=(*SIZE_MNET, 3))
    salida = tf.keras.layers.Softmax()(modelos_b[0](extractor(inputs)))
    tf.keras.Model(inputs, salida).save(AQUI / "modelotf_v8_cadena.keras")
    print(f"  guardado modelotf_v8_cadena.keras  (brazo B, semilla {SEMILLAS[0]})")
