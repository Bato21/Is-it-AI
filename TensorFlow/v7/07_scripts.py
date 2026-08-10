"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 7  (PROTOCOLO DE EVALUACIÓN HONESTO)

Propósito experimental:
  La v6 buscó hiperparámetros con Optuna y reportó 93.3% de val_accuracy (best trial
  96.7%). Ese número tiene un problema metodológico serio, y la v7 existe para medirlo:

    Optuna probó 30 combinaciones MAXIMIZANDO val_accuracy sobre 60 imágenes, y
    después se reentrenó y se reportó la accuracy SOBRE ESAS MISMAS 60 imágenes.
    Es "el mejor de 30 intentos sobre el mismo examen". El ganador de 30 intentos
    sobre un examen chico gana en parte por mérito y en parte por suerte, y no hay
    forma de separar las dos cosas si nunca se lo evalúa sobre un examen nuevo.

  La v7 NO cambia el modelo: mismo backbone MobileNetV3-Small congelado, misma cabeza,
  MISMO espacio de búsqueda de Optuna que la v6. Lo único que cambia es CÓMO se mide.
  Así el A/B contra la v6 aísla una sola pregunta: ¿cuánto del 93.3% era real y cuánto
  era sesgo de selección?

Los cuatro cambios de protocolo:

  1. TEST APARTADO (20%, 63 imgs). Se genera con documentacion/particion_datos.py y no
     se toca hasta el reporte final. Optuna NUNCA lo ve. Es el único número honesto.

  2. VALIDACIÓN CRUZADA 5-FOLD sobre el 80% de desarrollo. El objetivo de Optuna ya no
     es la accuracy de UNA validación de 60 imágenes, sino la MEDIA de 5 validaciones
     de ~50. Promediar 5 mediciones ruidosas reduce el ruido ~sqrt(5) y hace que TPE
     persiga señal en vez de perseguir suerte.

  3. SPLIT ESTRATIFICADO Y COMPARTIDO con PyTorch. Hasta la v6, TF partía con
     image_dataset_from_directory (que además dejaba 24/18/18 en vez de soportes
     parejos) y PyTorch con random_split: splits distintos, así que comparar 80.0%
     (TF v4) contra 90.0% (PT v4) tenía un asterisco. Ahora los dos leen
     documentacion/particion.json. El asterisco desaparece.

  4. MULTI-SEMILLA (5 corridas). Con 63 imágenes de test, UNA imagen vale 1.6 puntos de
     accuracy. Reportar "93.3%" de una sola corrida finge una precisión que el dataset
     no soporta. Acá se reporta media ± desvío, que es lo que permite decir si una
     diferencia entre dos modelos es real o es ruido.

Nuevo respecto a la v6:
  - Partición canónica compartida (documentacion/particion_datos.py).
  - Optuna optimiza la media de 5-fold CV, con pruning por fold.
  - Refit sobre los 250 de desarrollo y evaluación ÚNICA sobre el test de 63.
  - Métricas nuevas: QWK y MAE ordinal (las clases son ordinales) y ECE (calibración),
    todas en documentacion/metricas.py — el mismo código que usa el lado PyTorch.
  - CURVAS ROC del lado TensorFlow: hasta ahora existían SOLO en PyTorch (6 figuras),
    así que la paridad "espejo" del proyecto estaba rota. Con el módulo compartido,
    los dos frameworks las generan con el mismo código.
  - Resultados agregados sobre 5 semillas (media ± std) + figura de dispersión.

CONTRASTE DE FRAMEWORKS (espejo de PyTorch/v7/07_scripts.py):
  El protocolo, la partición, el espacio de búsqueda y las métricas son IDÉNTICOS —
  literalmente el mismo JSON de partición y el mismo módulo de métricas. Lo que sigue
  difiriendo es la mecánica:
    - Carga de datos: acá tf.data.Dataset.from_tensor_slices sobre la LISTA de rutas
      del manifiesto + map(decode_jpeg/png); en PyTorch, un Dataset propio con PIL.
      Ninguno de los dos usa ya el barrido automático de carpetas.
    - Entrenamiento: acá Sequential + compile + fit con el callback EarlyStopping; en
      PyTorch, bucle manual con copia del state_dict a mano.
    - Preprocesamiento ImageNet: acá vive DENTRO del modelo (MobileNetV3Small viene con
      include_preprocessing=True, así que recibe imágenes 0-255 crudas y normaliza sola);
      en PyTorch vive en el transform, fuera del modelo. Es la diferencia estructural
      que el proyecto viene marcando desde la v4.
    - BatchNorm: acá trainable=False ya pone las BN en modo inferencia; en PyTorch hay
      que forzar el backbone a eval() a mano (el gotcha documentado en la v4).
"""

import json
import os
import statistics
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silenciar logs de TF (solo errores)

import numpy as np
import optuna
import tensorflow as tf

# Módulos COMPARTIDOS con PyTorch (numpy puro, sin dependencias de framework).
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402
from particion_datos import cargar_particion, conteos, rutas_y_etiquetas  # noqa: E402

tf.get_logger().setLevel("ERROR")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # solo el resumen, no cada trial

# --- 1. Parámetros ---
DATA_DIR = RAIZ / "dataset"
if not DATA_DIR.is_dir():
    raise SystemExit(
        f"No existe la carpeta de datos: {DATA_DIR}\n"
        "Colocá el dataset ahí (subcarpetas 0_sin_ia/ 1_rastro_ia/ 2_saturada_ia/), o\n"
        "generá datos sintéticos para probar el flujo de punta a punta:\n"
        "    python documentacion/crear_datos_prueba.py --por-clase 30"
    )

IMG_SIZE = (224, 224)      # requisito de MobileNetV3, igual que v4/v6
BATCH_SIZE = 32
SEED = 123                 # semilla base (la misma de v1-v6)

N_TRIALS = 30              # igual que la v6, para que el A/B sea limpio
EPOCHS_TRIAL = 40          # techo por fold (el early stopping corta antes)
PATIENCE = 8
SEMILLAS = [123, 7, 42, 2024, 31]   # 5 corridas del refit; la 1a es la de siempre


# --- 2. Carga de datos desde la partición canónica ---
particion = cargar_particion()
class_names = particion["clases"]
print("Orden de clases:", class_names)  # ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']
num_classes = len(class_names)
N_FOLDS = particion["meta"]["n_folds"]

rutas_dev, y_dev = rutas_y_etiquetas(particion, "desarrollo")
rutas_test, y_test = rutas_y_etiquetas(particion, "test")
print(f"\nPartición canónica (documentacion/particion.json, huella "
      f"{particion['meta']['huella_dataset']}):")
print(f"  desarrollo : {len(rutas_dev):>3} imgs  {conteos(particion, rutas_dev)}")
print(f"  test       : {len(rutas_test):>3} imgs  {conteos(particion, rutas_test)}   "
      f"<- APARTADO, se usa una sola vez")
print(f"  folds      : {N_FOLDS} (validación cruzada dentro de desarrollo)")

y_dev = np.array(y_dev, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)


def dataset_de_rutas(rutas: list[str], etiquetas: np.ndarray) -> tf.data.Dataset:
    """tf.data a partir de una LISTA EXPLÍCITA de rutas, no de una carpeta.

    CONTRASTE con v1-v6: ahí se usaba image_dataset_from_directory, que barre el
    directorio y arma el split por su cuenta. Acá las rutas vienen de particion.json,
    que es el mismo archivo que lee PyTorch. Ese cambio es lo que garantiza que los dos
    frameworks entrenen y evalúen exactamente sobre las mismas imágenes.

    decode_image con channels=3 cubre PNG con alfa y JPEG en el mismo grafo (el
    equivalente del convert("RGB") de PIL del lado PyTorch). expand_animations=False es
    obligatorio: sin eso, decode_image devuelve rango dinámico desconocido y el resize
    falla al construir el grafo.
    """
    completas = [str(DATA_DIR / r) for r in rutas]

    def cargar(ruta, etiqueta):
        img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        # Se dejan las imágenes en 0-255 CRUDAS a propósito: el backbone trae su propia
        # capa de preprocesamiento ImageNet (include_preprocessing=True), igual que v4/v6.
        return tf.cast(img, tf.float32), etiqueta

    return (tf.data.Dataset.from_tensor_slices((completas, etiquetas))
            .map(cargar, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))


# --- 3. Backbone congelado + PRE-CÓMPUTO de embeddings ---
# El backbone no cambia nunca (feature extraction), así que sus features se calculan
# UNA vez para las 313 imágenes y todos los trials, folds y semillas entrenan una MLP
# chica sobre esos vectores. Es lo que hace que 30 trials x 5 folds x 5 semillas corra
# en minutos en CPU en vez de en horas.
def construir_extractor() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False  # feature extraction; BN pasa a modo inferencia sola
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # (N, 576)
    return tf.keras.Model(inputs, x, name="extractor_features")


print("\nPre-computando embeddings del backbone congelado (una sola vez)...")
extractor = construir_extractor()
EMB_DIM = extractor.output_shape[-1]


def embeddings_de(rutas: list[str], etiquetas: np.ndarray) -> np.ndarray:
    xs = []
    for images, _ in dataset_de_rutas(rutas, etiquetas):
        xs.append(extractor.predict(images, verbose=0))
    return np.concatenate(xs)


X_dev = embeddings_de(rutas_dev, y_dev)
X_test = embeddings_de(rutas_test, y_test)
print(f"  desarrollo: {X_dev.shape}  ·  test: {X_test.shape}  (dim {EMB_DIM})")

# Índices de cada fold DENTRO de X_dev: rutas_dev es la concatenación de los folds en
# orden, así que basta con recorrer los folds y mapear ruta -> posición.
pos_de_ruta = {r: i for i, r in enumerate(rutas_dev)}
idx_folds = [
    np.array([pos_de_ruta[r] for r in rutas_y_etiquetas(particion, "validacion", fold=k)[0]])
    for k in range(N_FOLDS)
]


# --- 4. Cabeza y entrenamiento ---
# ESPACIO DE BÚSQUEDA IDÉNTICO AL DE LA v6 (a propósito: si cambiara el espacio, el A/B
# v6 vs v7 mezclaría "mejor protocolo" con "mejor búsqueda" y no se podría atribuir la
# diferencia a nada). Los hiperparámetros nuevos (weight_decay, label_smoothing) entran
# recién en la v8, junto con la destilación.
def construir_cabeza(params) -> tf.keras.Sequential:
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(EMB_DIM,))])
    for _ in range(params["n_capas"]):
        model.add(tf.keras.layers.Dense(params["units"], activation="relu"))
        model.add(tf.keras.layers.Dropout(params["dropout"]))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    opts = {
        "adam": tf.keras.optimizers.Adam(params["lr"]),
        "rmsprop": tf.keras.optimizers.RMSprop(params["lr"]),
        "sgd": tf.keras.optimizers.SGD(params["lr"], momentum=0.9),
    }
    model.compile(optimizer=opts[params["optimizer"]],
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def entrenar_cabeza(params, X_tr, y_tr, X_va, y_va, epochs, paciencia, semilla):
    """Entrena la cabeza. Devuelve (modelo, mejor val_acc, mejor época).

    La "mejor época" se devuelve porque el refit final la necesita: cuando se
    reentrena sobre TODO desarrollo ya no queda validación con la cual cortar, así
    que se usa la mediana de las mejores épocas de los 5 folds (receta estándar de
    refit tras validación cruzada).

    Si X_va es None se entrena a ciegas por `epochs` épocas: ese es exactamente el
    caso del refit final.

    CONTRASTE: acá el early stopping con restore_best_weights=True es un callback de
    una línea; en PyTorch hay que llevar a mano el contador de épocas sin mejora y una
    copia profunda del state_dict.
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(semilla)
    modelo = construir_cabeza(params)

    if X_va is None:  # refit a ciegas: sin validación, sin early stopping
        modelo.fit(X_tr, y_tr, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
        return modelo, 0.0, epochs

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=paciencia, restore_best_weights=True, verbose=0,
    )
    hist = modelo.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
                      batch_size=BATCH_SIZE, callbacks=[early], verbose=0)
    val_accs = hist.history["val_accuracy"]
    mejor_epoca = int(np.argmax(val_accs)) + 1
    return modelo, float(max(val_accs)), mejor_epoca


def probabilidades(modelo, X) -> np.ndarray:
    """Vector COMPLETO de probabilidades softmax (no solo el argmax).

    Es una regla del proyecto desde la v2 —toda inferencia devuelve el vector— y acá
    además es un requisito técnico: ROC-AUC y ECE necesitan el score continuo, no la
    decisión dura. La cabeza ya termina en softmax, así que predict() devuelve
    probabilidades directamente (en PyTorch hay que aplicar torch.softmax a mano).
    """
    return modelo.predict(X, verbose=0)


# --- 5. Validación cruzada: el objetivo REAL de Optuna ---
def validacion_cruzada(params, semilla=SEED, trial=None) -> tuple[float, list[int], list[float]]:
    """Entrena y evalúa la cabeza en los 5 folds. Devuelve (media, mejores épocas, accs).

    Este es el cambio central de la v7. En la v6, objective() devolvía la accuracy de
    UNA validación de 60 imágenes: con 30 trials compitiendo, el ganador era en buena
    parte el que tuvo suerte con esas 60. Acá el trial se juzga por la MEDIA de 5
    validaciones independientes, que es una estimación mucho más estable.

    El pruning ahora tiene un "paso" natural: el fold. Se reporta la media parcial
    después de cada fold y, si ya va por debajo de la mediana histórica, MedianPruner
    corta el trial sin gastar los folds restantes.
    """
    accs, epocas = [], []
    for k in range(N_FOLDS):
        idx_va = idx_folds[k]
        mascara = np.ones(len(X_dev), dtype=bool)
        mascara[idx_va] = False
        _, acc, epoca = entrenar_cabeza(params, X_dev[mascara], y_dev[mascara],
                                        X_dev[idx_va], y_dev[idx_va],
                                        EPOCHS_TRIAL, PATIENCE, semilla)
        accs.append(acc)
        epocas.append(max(epoca, 1))

        if trial is not None:
            trial.report(float(np.mean(accs)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(accs)), epocas, accs


def objective(trial):
    params = {
        "n_capas": trial.suggest_int("n_capas", 1, 2),
        "units": trial.suggest_categorical("units", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6, step=0.05),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }
    media, _, accs = validacion_cruzada(params, semilla=SEED, trial=trial)
    trial.set_user_attr("std_folds", float(np.std(accs)))
    return media


def ejecutar_estudio() -> optuna.Study:
    print("\n" + "=" * 78)
    print(f"  Optuna sobre la cabeza  ·  {N_TRIALS} trials  ·  objetivo = media de "
          f"{N_FOLDS}-fold CV")
    print("=" * 78)
    print(f"{'Trial':>6}{'cv_media':>10}{'±std':>8}{'mejor':>8}   parámetros")
    print("-" * 78)
    sampler = optuna.samplers.TPESampler(seed=SEED)   # bayesiano, igual que la v6
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name="iahuella_cabeza_cv")

    def callback(study, trial):
        p = trial.params
        # GOTCHA DE OPTUNA 4.x: un trial PODADO NO tiene value=None. Optuna le deja el
        # ÚLTIMO valor intermedio reportado (la media parcial de los folds que alcanzó
        # a correr). Filtrar por `value is not None` los cuenta como completados y
        # ensucia las estadísticas del estudio con medias de 2-3 folds. Hay que mirar
        # trial.state, que es el dato autoritativo.
        if trial.state == optuna.trial.TrialState.PRUNED:
            print(f"{trial.number:>6}{'podado':>10}{'':>8}{study.best_value:>8.4f}   "
                  f"capas={p.get('n_capas')} units={p.get('units')} "
                  f"drop={p.get('dropout'):.2f} opt={p.get('optimizer')} lr={p.get('lr', 0):.1e}")
            return
        std = trial.user_attrs.get("std_folds", 0.0)
        print(f"{trial.number:>6}{trial.value:>10.4f}{std:>8.4f}{study.best_value:>8.4f}   "
              f"capas={p.get('n_capas')} units={p.get('units')} "
              f"drop={p.get('dropout'):.2f} opt={p.get('optimizer')} lr={p.get('lr', 0):.1e}")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback], show_progress_bar=False)
    return study


# --- 6. Figuras de Optuna (idénticas a la v6, sobre el nuevo objetivo) ---
def plot_historia(study, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    nums = [t.number for t in trials]
    vals = [t.value for t in trials]
    mejores = [max(vals[: i + 1]) for i in range(len(vals))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(nums, vals, color="#2E75B6", alpha=0.75, width=0.7)
    ax1.plot(nums, mejores, "r-o", ms=4, lw=2, label="mejor acumulado")
    ax1.set_xlabel("trial"); ax1.set_ylabel(f"media de {N_FOLDS}-fold CV")
    ax1.set_title("accuracy de CV por trial"); ax1.legend(); ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax2.hist(vals, bins=10, color="#2E75B6", alpha=0.75, edgecolor="white")
    ax2.axvline(max(vals), color="green", ls="--", lw=2, label=f"mejor {max(vals):.3f}")
    ax2.axvline(np.mean(vals), color="orange", ls="--", lw=2, label=f"media {np.mean(vals):.3f}")
    ax2.set_xlabel(f"media de {N_FOLDS}-fold CV"); ax2.set_ylabel("nº trials")
    ax2.set_title("distribución de resultados"); ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    plt.suptitle(f"Optuna v7 — {len(trials)} trials completados · TPE · objetivo = CV",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {salida.name}")


def plot_importancia(study, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"  (importancia no disponible: {e})")
        return
    nombres = list(imp.keys())[::-1]
    valores = list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(nombres, valores, color="#2E75B6", edgecolor="white", height=0.6)
    for i, v in enumerate(valores):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlabel("importancia (fANOVA)")
    ax.set_title("Importancia de hiperparámetros — v7\nmayor = más impacto en la CV",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {salida.name}")


# --- 7. Main ---
if __name__ == "__main__":
    AQUI = Path(__file__).resolve().parent

    study = ejecutar_estudio()
    # Solo los COMPLETE cuentan para las estadísticas: los podados traen una media
    # parcial (2-3 folds) que no es comparable con una de 5 folds.
    completados = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    podados = len(study.trials) - len(completados)

    print("\n" + "=" * 78)
    print("  MEJORES HIPERPARÁMETROS (elegidos por validación cruzada, sin ver el test)")
    print("=" * 78)
    print(f"  media de CV: {study.best_value:.4f}  ({study.best_value:.1%})")
    for k, v in study.best_params.items():
        print(f"    {k:<10}: {v}")
    print(f"\n  trials completados : {len(completados)}/{N_TRIALS}  ({podados} podados)")
    print(f"  media cv_accuracy  : {np.mean([t.value for t in completados]):.4f}")
    print(f"  std  cv_accuracy   : {np.std([t.value for t in completados]):.4f}")
    print(f"  peor / mejor       : {min(t.value for t in completados):.4f} / "
          f"{max(t.value for t in completados):.4f}")

    # Época de refit: mediana de las mejores épocas de los 5 folds con los mejores
    # hiperparámetros. Al reentrenar sobre TODO desarrollo no queda validación para
    # cortar, así que se fija de antemano un nº de épocas derivado de la CV.
    _, epocas_cv, accs_cv = validacion_cruzada(study.best_params, semilla=SEED)
    EPOCHS_REFIT = int(statistics.median(epocas_cv))
    print(f"\n  accuracy por fold  : {[round(a, 4) for a in accs_cv]}")
    print(f"  mejores épocas     : {epocas_cv}  ->  refit con {EPOCHS_REFIT} épocas")

    # --- Refit + evaluación sobre TEST, una semilla por vez ---
    print("\n" + "-" * 78)
    print(f"  Refit sobre desarrollo ({len(rutas_dev)} imgs) y evaluación sobre TEST "
          f"({len(rutas_test)} imgs)")
    print(f"  {len(SEMILLAS)} semillas: {SEMILLAS}")
    print("-" * 78)

    resumenes, cabezas = [], []
    for semilla in SEMILLAS:
        cabeza, _, _ = entrenar_cabeza(study.best_params, X_dev, y_dev, None, None,
                                       EPOCHS_REFIT, PATIENCE, semilla)
        y_prob = probabilidades(cabeza, X_test)
        r = metricas.resumen_completo(y_test, y_prob, class_names)
        r["semilla"] = semilla
        r["y_prob"] = y_prob
        resumenes.append(r)
        cabezas.append(cabeza)
        print(f"  semilla {semilla:>5}  test_acc={r['accuracy']:.4f}  "
              f"F1={r['macro_f1']:.4f}  QWK={r['qwk']:.4f}  "
              f"0<->2={r['extremos_total']}  ECE={r['ece']:.4f}")

    # --- Agregado media ± std ---
    print()
    agregado = metricas.agregar_semillas(resumenes)
    metricas.imprimir_agregado(agregado, "RESULTADOS SOBRE TEST — TensorFlow v7")

    cm_sumada = np.array(agregado["cm_sumada"])
    print(f"\nMatriz de confusión ACUMULADA sobre {len(SEMILLAS)} semillas "
          f"(filas = real, columnas = predicho):")
    print(cm_sumada)
    print(f"\nReporte por clase (acumulado, {int(cm_sumada.sum())} predicciones):")
    metricas.reporte_por_clase(cm_sumada, class_names)
    ext = metricas.errores_extremos(cm_sumada)
    print(f"\nError caro (esquina 0<->2): {ext['extremos_total']} de {int(cm_sumada.sum())}")
    print(f"  saturada clasificada como sin_ia (2->0): {ext['saturada_como_sin_ia']}")
    print(f"  sin_ia clasificada como saturada (0->2): {ext['sin_ia_como_saturada']}")

    # --- Comparación explícita contra el número de la v6 ---
    acc_v7 = agregado["accuracy"]["media"]
    print("\n" + "=" * 78)
    print("  LO QUE COSTABA EL PROTOCOLO VIEJO")
    print("=" * 78)
    print(f"  v6 (TensorFlow): 0.9333 de val_accuracy (best trial 0.9667), pero ese val")
    print(f"                   es el MISMO conjunto que Optuna maximizó durante 30")
    print(f"                   trials -> sesgo de selección.")
    print(f"  v7 (TensorFlow): {acc_v7:.4f} ± {agregado['accuracy']['std']:.4f} sobre un test")
    print(f"                   de {len(rutas_test)} imágenes que la búsqueda NUNCA vio.")
    print(f"  Diferencia     : {acc_v7 - 0.9333:+.4f}")

    # --- Figuras ---
    print("\nGenerando figuras...")
    plot_historia(study, AQUI / "Figure_optuna_historia_v7.png")
    plot_importancia(study, AQUI / "Figure_optuna_importancia_v7.png")
    metricas.plot_matriz(cm_sumada, class_names,
                         f"Matriz de confusión v7 (test, {len(SEMILLAS)} semillas)",
                         AQUI / "Figure_matriz_v7.png")

    # ROC y calibración de la PRIMERA semilla (no de la mejor: elegir la mejor por su
    # resultado en el test volvería a contaminar el test, que es justo el error que
    # esta versión vino a corregir).
    y_prob_0 = resumenes[0]["y_prob"]
    curvas, micro, macro = metricas.curvas_roc_ovr(y_test, y_prob_0, num_classes)
    print(f"\nROC-AUC One-vs-Rest (test, semilla {SEMILLAS[0]}):")
    for i in range(num_classes):
        print(f"  {class_names[i]:<16} AUC = {curvas[i][2]:.4f}")
    print(f"  {'macro-promedio':<16} AUC = {macro:.4f}")
    print(f"  {'micro-promedio':<16} AUC = {micro[2]:.4f}")

    metricas.plot_roc(curvas, micro, macro, class_names,
                      f"Curvas ROC One-vs-Rest — v7 (test, semilla {SEMILLAS[0]})",
                      AQUI / "Figure_roc_v7.png")
    metricas.plot_calibracion(y_test, y_prob_0,
                              f"Calibración — v7 (test, semilla {SEMILLAS[0]})",
                              AQUI / "Figure_calibracion_v7.png")
    metricas.plot_semillas(resumenes, ["accuracy", "macro_f1", "qwk", "ece"],
                           f"Dispersión entre semillas — TensorFlow v7 (test, "
                           f"{len(rutas_test)} imgs)",
                           AQUI / "Figure_semillas_v7.png")

    # --- Guardar resultados e hiperparámetros ---
    with open(AQUI / "mejores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump({
            "cv_accuracy": study.best_value,
            "params": study.best_params,
            "epochs_refit": EPOCHS_REFIT,
            "accuracy_por_fold": accs_cv,
        }, f, indent=2, ensure_ascii=False)
    print("\n  guardado mejores_hiperparametros.json")

    with open(AQUI / "resultados_v7.json", "w", encoding="utf-8") as f:
        json.dump({
            "framework": "tensorflow",
            "particion": particion["meta"],
            "semillas": SEMILLAS,
            "params": study.best_params,
            "epochs_refit": EPOCHS_REFIT,
            "cv_accuracy": study.best_value,
            "test": {k: agregado[k] for k in metricas.CLAVES_ESCALARES},
            "cm_sumada": agregado["cm_sumada"],
            "por_semilla": [{k: v for k, v in r.items() if k != "y_prob"} for r in resumenes],
        }, f, indent=2, ensure_ascii=False)
    print("  guardado resultados_v7.json")

    # --- Modelo de inferencia de punta a punta (imágenes 0-255 -> backbone -> cabeza) ---
    # Se guarda la cabeza de la PRIMERA semilla, decidido de antemano. Guardar "la
    # mejor de las 5 según el test" sería volver a elegir mirando el test.
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    modelo_full = tf.keras.Model(inputs, cabezas[0](extractor(inputs)))
    modelo_full.save(AQUI / "modelotf_v7_protocolo.keras")
    print(f"  guardado modelotf_v7_protocolo.keras  (semilla {SEMILLAS[0]}, "
          f"elegida de antemano)")
