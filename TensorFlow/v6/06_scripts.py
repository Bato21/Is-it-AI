"""
Clasificador de huella de IA en diapositivas — TensorFlow / Keras
Versión 6  (búsqueda de hiperparámetros de la CABEZA con Optuna)

Propósito experimental:
  La v4 fijó la arquitectura (MobileNetV3-Small congelada + cabeza GAP->Dropout(0.2)->
  Dense(3)) y entrenó con hiperparámetros elegidos A MANO (Adam por defecto, dropout 0.2,
  batch 32). La v6 hace UN solo cambio conceptual: en vez de elegir esos hiperparámetros
  a ojo, se BUSCAN automáticamente con Optuna, maximizando val_accuracy.

  MISMO backbone que la v4 (feature extraction: MobileNetV3-Small preentrenada, congelada),
  MISMO dataset balanceado (dataset/), MISMO split 80/20 y seed. Lo único que cambia es que
  la cabeza (dropout, capa oculta, learning rate, optimizador, batch) ya no es fija: es lo
  que Optuna explora trial a trial.

  Flujo, calcado del ejemplo del profe (documentacion/optuna_ejemplo_revisar.py) pero
  aplicado a NUESTRO problema de visión en vez de Iris:
    1. objective(trial): propone hiperparámetros, arma la cabeza, entrena pocas épocas
       sobre features del backbone congelado y devuelve el mejor val_accuracy.
    2. study.optimize: TPE (sampler bayesiano) + MedianPruner (corta trials malos).
    3. Se reentrena con los MEJORES hiperparámetros y se evalúa con matriz de confusión
       + reporte por clase (las métricas que se defienden en la interrogación oral).

  OPTIMIZACIÓN CLAVE (por qué esto corre en minutos y no en horas):
    El backbone está congelado, así que sus features NO cambian entre trials. Se calculan
    UNA vez (pre-cómputo de embeddings 576-dim) y todos los trials entrenan una MLP chica
    sobre esos vectores. Buscar hiperparámetros de la cabeza != re-pasar imágenes por la
    CNN 30 veces. Este es el patrón correcto de Optuna sobre feature extraction.

Nuevo respecto a la v4:
  - Búsqueda automática de hiperparámetros con Optuna (TPE + MedianPruner).
  - Pre-cómputo de embeddings del backbone congelado (features fijas).
  - Figuras optuna_historia.png (val_acc por trial) y optuna_importancia.png.
  - Se guardan los mejores hiperparámetros en mejores_hiperparametros.json.

CONTRASTE DE FRAMEWORKS (espejo de PyTorch/v6/06_scripts.py):
  El ESPACIO de búsqueda y la lógica de Optuna son idénticos (Optuna no depende del
  framework: recibe un trial y un número a maximizar). Lo que cambia es cómo se arma y
  entrena la cabeza dentro de objective(): en Keras es Sequential + compile + fit; en
  PyTorch es nn.Sequential + bucle manual de entrenamiento. El pre-cómputo de embeddings
  también difiere (model.predict del backbone vs. forward con torch.no_grad()).
"""

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silenciar logs de TF (solo errores)

import numpy as np
import tensorflow as tf
import optuna
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # solo el resumen, no cada trial

# --- 1. Parámetros ---
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset")
if not Path(DATA_DIR).is_dir():
    raise SystemExit(
        f"No existe la carpeta de datos: {DATA_DIR}\n"
        "Colocá el dataset ahí (subcarpetas 0_sin_ia/ 1_rastro_ia/ 2_saturada_ia/), o\n"
        "generá datos sintéticos para probar el flujo de punta a punta:\n"
        "    python documentacion/crear_datos_prueba.py --por-clase 30"
    )
IMG_SIZE = (224, 224)      # requisito de MobileNetV3, igual que la v4
BATCH_SIZE = 32
SEED = 123                 # MISMO seed/split que v2-v5

N_TRIALS = 30              # combinaciones a probar (igual que el ejemplo del profe)
EPOCHS_TRIAL = 40          # épocas máx por trial (con early stopping cortan antes)
EPOCHS_FINAL = 80          # épocas del reentrenamiento con los mejores hiperparámetros
PATIENCE = 8

# --- 2. Carga de datos (split 80/20 IDÉNTICO a v4, mismo seed) ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
class_names = train_ds.class_names
print("Orden de clases:", class_names)  # ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']
num_classes = len(class_names)


# --- 3. Backbone congelado + PRE-CÓMPUTO de embeddings ---
# El backbone no cambia entre trials => se calculan sus features UNA sola vez y todos
# los trials entrenan sobre esos vectores. Aquí SÍ incluimos el preprocesamiento
# ImageNet dentro del extractor (imágenes 0-255 crudas, igual que la v4).
def construir_extractor() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False  # feature extraction; BN en modo inferencia
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # (N, 576)
    return tf.keras.Model(inputs, x, name="extractor_features")


print("\nPre-computando embeddings del backbone congelado (una sola vez)...")
extractor = construir_extractor()
EMB_DIM = extractor.output_shape[-1]


def embeddings_de(ds):
    xs, ys = [], []
    for images, labels in ds:
        xs.append(extractor.predict(images, verbose=0))
        ys.append(labels.numpy())
    return np.concatenate(xs), np.concatenate(ys)


X_train, y_train = embeddings_de(train_ds)
X_val, y_val = embeddings_de(val_ds)
print(f"  embeddings train: {X_train.shape}  ·  val: {X_val.shape}  (dim {EMB_DIM})")


# --- 4. Reporte por clase (a mano desde la matriz de confusión, igual que v4/v5) ---
# cm[i, j] = real i, predicho j.
def reporte_por_clase(cm: np.ndarray, nombres: list[str]) -> None:
    total = cm.sum()
    print(f"{'clase':<16}{'precision':>10}{'recall':>9}{'f1':>7}{'soporte':>9}")
    print("-" * 51)
    precs, recs, f1s = [], [], []
    for i in range(len(nombres)):
        soporte = cm[i, :].sum()
        predichos = cm[:, i].sum()
        recall = cm[i, i] / soporte if soporte else 0.0
        precision = cm[i, i] / predichos if predichos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
        print(f"{nombres[i]:<16}{precision:>10.3f}{recall:>9.3f}{f1:>7.3f}{int(soporte):>9}")
    print("-" * 51)
    print(f"{'macro avg':<16}{np.mean(precs):>10.3f}{np.mean(recs):>9.3f}{np.mean(f1s):>7.3f}{int(total):>9}")
    acc = np.trace(cm) / total if total else 0.0
    print(f"accuracy: {acc:.3f}  ({int(np.trace(cm))}/{int(total)})")


# --- 5. Espacio de búsqueda + función objetivo ---
# ESPACIO (todo sobre la CABEZA; el backbone queda congelado):
#   n_capas    int  [1, 2]                capas densas ocultas antes del softmax
#   units      cat  {64, 128, 256}        neuronas por capa oculta
#   dropout    float[0.1, 0.6] step 0.05  regularización de la cabeza
#   optimizer  cat  {adam, rmsprop, sgd}
#   lr         float[1e-4, 1e-2] log      learning rate
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


def objective(trial):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    params = {
        "n_capas": trial.suggest_int("n_capas", 1, 2),
        "units": trial.suggest_categorical("units", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6, step=0.05),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }
    model = construir_cabeza(params)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=0,
    )
    hist = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS_TRIAL, batch_size=BATCH_SIZE, callbacks=[early], verbose=0,
    )
    return max(hist.history["val_accuracy"])


# --- 6. Estudio Optuna (TPE + MedianPruner, calcado del ejemplo del profe) ---
def ejecutar_estudio() -> optuna.Study:
    print("\n" + "=" * 66)
    print(f"  Optuna sobre la cabeza  ·  {N_TRIALS} trials  ·  máx {EPOCHS_TRIAL} épocas/trial")
    print("=" * 66)
    print(f"{'Trial':>6}{'val_acc':>9}{'mejor':>8}   parámetros")
    print("-" * 66)
    sampler = optuna.samplers.TPESampler(seed=SEED)  # bayesiano: aprende qué zonas explorar
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name="iahuella_cabeza")

    def callback(study, trial):
        val = trial.value if trial.value is not None else 0.0
        p = trial.params
        print(f"{trial.number:>6}{val:>9.4f}{study.best_value:>8.4f}   "
              f"capas={p.get('n_capas')} units={p.get('units')} "
              f"drop={p.get('dropout')} opt={p.get('optimizer')} lr={p.get('lr', 0):.1e}")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback], show_progress_bar=False)
    return study


# --- 7. Reentrenar con los mejores hiperparámetros + matriz de confusión ---
def reentrenar_mejor(best_params) -> tuple[tf.keras.Model, np.ndarray]:
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    cabeza = construir_cabeza(best_params)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE + 4, restore_best_weights=True, verbose=0,
    )
    cabeza.fit(X_train, y_train, validation_data=(X_val, y_val),
               epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE, callbacks=[early], verbose=2)

    y_pred = np.argmax(cabeza.predict(X_val, verbose=0), axis=1)
    cm = tf.math.confusion_matrix(y_val, y_pred, num_classes=num_classes).numpy()

    # Modelo de INFERENCIA de punta a punta: imágenes 0-255 -> backbone -> cabeza.
    # Se une el extractor (con su preprocesamiento ImageNet) y la cabeza óptima en un
    # solo .keras, para que el consumo (modelos/v6/) reciba imágenes crudas como en v4.
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    modelo_full = tf.keras.Model(inputs, cabeza(extractor(inputs)))
    return modelo_full, cm


# --- 8. Figuras de Optuna (val_acc por trial + importancia de hiperparámetros) ---
def plot_historia(study, salida):
    trials = [t for t in study.trials if t.value is not None]
    nums = [t.number for t in trials]
    vals = [t.value for t in trials]
    mejores = [max(vals[: i + 1]) for i in range(len(vals))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(nums, vals, color="#2E75B6", alpha=0.75, width=0.7)
    ax1.plot(nums, mejores, "r-o", ms=4, lw=2, label="mejor acumulado")
    ax1.set_xlabel("trial"); ax1.set_ylabel("val_accuracy")
    ax1.set_title("val_accuracy por trial"); ax1.legend(); ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax2.hist(vals, bins=10, color="#2E75B6", alpha=0.75, edgecolor="white")
    ax2.axvline(max(vals), color="green", ls="--", lw=2, label=f"mejor {max(vals):.3f}")
    ax2.axvline(np.mean(vals), color="orange", ls="--", lw=2, label=f"media {np.mean(vals):.3f}")
    ax2.set_xlabel("val_accuracy"); ax2.set_ylabel("nº trials")
    ax2.set_title("distribución de resultados"); ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    plt.suptitle(f"Optuna — {len(trials)} trials · TPE · cabeza sobre MobileNetV3-Small",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {salida.name}")


def plot_importancia(study, salida):
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:  # fANOVA necesita >= algunos trials completos
        print(f"  (importancia no disponible: {e})")
        return
    nombres = list(imp.keys())[::-1]
    valores = list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(nombres, valores, color="#2E75B6", edgecolor="white", height=0.6)
    for i, v in enumerate(valores):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlabel("importancia (fANOVA)")
    ax.set_title("Importancia de hiperparámetros\nmayor = más impacto en val_accuracy",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {salida.name}")


# --- 9. Main ---
if __name__ == "__main__":
    AQUI = Path(__file__).resolve().parent

    study = ejecutar_estudio()

    print("\n" + "=" * 66)
    print("  MEJORES HIPERPARÁMETROS")
    print("=" * 66)
    print(f"  val_accuracy: {study.best_value:.4f}  ({study.best_value:.1%})")
    for k, v in study.best_params.items():
        print(f"    {k:<10}: {v}")
    trials_ok = [t for t in study.trials if t.value is not None]
    print(f"\n  trials completados : {len(trials_ok)}/{N_TRIALS}")
    print(f"  media val_accuracy : {np.mean([t.value for t in trials_ok]):.4f}")
    print(f"  std  val_accuracy  : {np.std([t.value for t in trials_ok]):.4f}")
    print(f"  peor / mejor       : {min(t.value for t in trials_ok):.4f} / "
          f"{max(t.value for t in trials_ok):.4f}")

    # Guardar los mejores hiperparámetros (para reproducir sin re-buscar).
    with open(AQUI / "mejores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump({"val_accuracy": study.best_value, "params": study.best_params}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n  guardado mejores_hiperparametros.json")

    print("\n" + "-" * 66)
    print("  Reentrenando con los mejores hiperparámetros")
    print("-" * 66)
    modelo_full, cm = reentrenar_mejor(study.best_params)
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    print("\nReporte por clase:")
    reporte_por_clase(cm, class_names)

    print("\nGenerando figuras...")
    plot_historia(study, AQUI / "Figure_optuna_historia_v6.png")
    plot_importancia(study, AQUI / "Figure_optuna_importancia_v6.png")

    # Matriz de confusión del modelo final.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión v6 (val)")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(AQUI / "Figure_matriz_v6.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  guardado Figure_matriz_v6.png")

    modelo_full.save(AQUI / "modelotf_v6_optuna.keras")
    print("\nModelo final guardado en modelotf_v6_optuna.keras")
