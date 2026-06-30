"""
Clasificador de huella de IA en diapositivas — TensorFlow
Versión 2 — Transfer Learning con MobileNetV3Large

Esta v2 responde al sobreajuste de la v1 (CNN desde cero) con:
  - Transfer learning (MobileNetV3Large preentrenada en ImageNet)
  - Data augmentation como capas del modelo
  - Dropout en la cabeza
  - Entrenamiento en dos fases (cabeza congelada -> fine-tuning)

Sigue el mismo flujo que el ejemplo de la Unidad 2 (flores), adaptado a las
3 clases del proyecto:
  0_sin_ia      — sin huella de IA visible
  1_rastro_ia   — IA con clara intervención humana (rastro)
  2_saturada_ia — predominantemente generada por IA

Genera, en esta carpeta:
  modelo_diapositivas.keras  — modelo entrenado (lo usa consumidor.py)
  training_curves.png        — loss y accuracy por época (fase 1 y 2)
  confusion_matrix.png       — matriz de confusión en test
  roc_curves.png             — curvas ROC one-vs-rest + AUC por clase

Librerías: TensorFlow 2.15/2.16 · NumPy 1.26 · matplotlib 3.8
"""

import math
import sys
from pathlib import Path

# Consola UTF-8 (Windows usa cp1252 por defecto y rompe con ─, acentos, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. CONFIGURACIÓN ──────────────────────────────────────────────────
RAIZ          = Path(__file__).resolve().parents[2]
DATA_DIR      = str(RAIZ / "data")
IMG_SIZE      = (224, 224)     # entrada nativa de MobileNetV3Large
BATCH_SIZE    = 32
EPOCHS_FASE1  = 15             # cabeza — base congelada
EPOCHS_FASE2  = 10             # fine-tune — últimas capas descongeladas
FINE_TUNE_AT  = 200            # descongelar capas desde este índice
LR_FASE1      = 1e-3
LR_FASE2      = 1e-4           # LR más bajo para fine-tune
SEED          = 42

CLASES   = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
N_CLASES = len(CLASES)
COLORES  = ["#1E7B45", "#B45309", "#991B1B"]   # un color por clase

MODELO_OUT = Path(__file__).resolve().parent / "modelo_diapositivas.keras"


# ── 2. DATASET ────────────────────────────────────────────────────────
def cargar_dataset():
    """70% train · 15% val · 15% test, con orden de clases fijo."""
    data_dir = Path(DATA_DIR)
    total = sum(1 for c in CLASES for _ in (data_dir / c).glob("*")
                if _.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if total == 0:
        raise SystemExit(
            f"No hay imágenes en {data_dir}. Convierte presentaciones con "
            f"tools/deck_a_imagenes.py y clasifícalas en data/<clase>/."
        )
    print(f"Total imágenes: {total}")

    kwargs = dict(
        directory   = DATA_DIR,
        seed        = SEED,
        image_size  = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_names = CLASES,        # fija el orden 0,1,2 (no alfabético sorpresa)
        label_mode  = "categorical", # one-hot => necesario para ROC y CE
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=0.30, subset="training", **kwargs)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=0.30, subset="validation", **kwargs)

    # Dividir val_test en 50/50 a NIVEL DE EJEMPLO (no de batch). Con datasets
    # pequeños puede haber un solo batch; hacer skip() por batch dejaría test vacío.
    val_test_unb = val_test_ds.unbatch()
    n_vt = sum(int(tf.shape(y)[0]) for _, y in val_test_ds)  # total de ejemplos
    n_val = max(1, min(n_vt - 1, n_vt // 2))                  # garantiza >=1 en test
    val_ds  = val_test_unb.take(n_val).batch(BATCH_SIZE)
    test_ds = val_test_unb.skip(n_val).batch(BATCH_SIZE)
    print(f"  val/test (ejemplos): val={n_val}  test={n_vt - n_val}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


# ── 3. MODELO ─────────────────────────────────────────────────────────
def construir_modelo():
    """
    Augmentation => MobileNetV3Large (congelado) => GAP => Dense => Dropout => Softmax

    include_preprocessing=True => el modelo espera imágenes 0-255 (el escalado
    a [-1,1] ocurre dentro de la red). Esto simplifica el despliegue móvil.
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV3Large(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )
    base.trainable = False   # Fase 1: base congelada

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = augmentation(inputs)
    x       = base(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(N_CLASES, activation="softmax")(x)

    modelo = tf.keras.Model(inputs, outputs, name="mobilenetv3_diapositivas")
    return modelo, base


# ── 4. ENTRENAR (dos fases) ───────────────────────────────────────────
def entrenar(modelo, base, train_ds, val_ds):
    historial = {}

    print(f"\n{'─'*55}\nFASE 1 — Cabeza ({EPOCHS_FASE1} épocas), base congelada\n{'─'*55}")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FASE1),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    cb1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, verbose=1, min_lr=1e-6),
    ]
    h1 = modelo.fit(train_ds, epochs=EPOCHS_FASE1, validation_data=val_ds, callbacks=cb1)
    historial["fase1"] = h1.history

    print(f"\n{'─'*55}\nFASE 2 — Fine-tuning (desde índice {FINE_TUNE_AT})\n{'─'*55}")
    base.trainable = True
    for capa in base.layers[:FINE_TUNE_AT]:
        capa.trainable = False
    print(f"  Capas descongeladas: {sum(l.trainable for l in base.layers)}  |  LR {LR_FASE2}")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FASE2),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    cb2 = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                            restore_best_weights=True, verbose=1)]
    h2 = modelo.fit(train_ds, epochs=EPOCHS_FASE2, validation_data=val_ds, callbacks=cb2)
    historial["fase2"] = h2.history
    return historial


# ── 5. MÉTRICAS MANUALES (sin sklearn, para evitar choques de versiones) ──
def calcular_cm(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def metricas_por_clase(cm):
    m = {}
    for i in range(len(cm)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        m[i] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}
    return m


def roc_manual(y_true_bin, y_score):
    thr = np.sort(np.unique(y_score))[::-1]
    P = int(y_true_bin.sum()); N = len(y_true_bin) - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    fprs, tprs = [0.0], [0.0]
    for t in thr:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true_bin == 1)).sum())
        fp = int(((pred == 1) & (y_true_bin == 0)).sum())
        fprs.append(fp / N); tprs.append(tp / P)
    fprs.append(1.0); tprs.append(1.0)
    fpr, tpr = np.array(fprs), np.array(tprs)
    return fpr, tpr, abs(float(np.trapz(tpr, fpr)))


def predecir_test(modelo, test_ds):
    yt, yp = [], []
    for imgs, labels in test_ds:
        probs = modelo(imgs, training=False).numpy()
        yt.append(np.argmax(labels.numpy(), axis=1))
        yp.append(probs)
    y_true = np.concatenate(yt); y_prob = np.concatenate(yp)
    return y_true, np.argmax(y_prob, axis=1), y_prob


# ── 6. PLOTS ──────────────────────────────────────────────────────────
def plot_curvas(historial, out="training_curves.png"):
    h1, h2 = historial["fase1"], historial["fase2"]
    acc = h1["accuracy"] + h2["accuracy"]; val_acc = h1["val_accuracy"] + h2["val_accuracy"]
    loss = h1["loss"] + h2["loss"]; val_loss = h1["val_loss"] + h2["val_loss"]
    ep = range(1, len(acc) + 1); corte = len(h1["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, acc, "b-o", ms=4, label="Train Acc"); ax1.plot(ep, val_acc, "b--s", ms=4, label="Val Acc")
    ax1.axvline(corte + 0.5, color="gray", ls=":"); ax1.set_title("Accuracy por época", fontweight="bold")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.05)
    ax2.plot(ep, loss, "r-o", ms=4, label="Train Loss"); ax2.plot(ep, val_loss, "r--s", ms=4, label="Val Loss")
    ax2.axvline(corte + 0.5, color="gray", ls=":"); ax2.set_title("Loss por época", fontweight="bold")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Categorical Crossentropy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle("MobileNetV3Large — Curvas de entrenamiento\nFase 1: cabeza · Fase 2: fine-tuning",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")


def plot_cm(cm, met, acc, out="confusion_matrix.png"):
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], figure=fig)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N_CLASES)); ax.set_yticks(range(N_CLASES))
    ax.set_xticklabels(CLASES, rotation=30, ha="right"); ax.set_yticklabels(CLASES)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de Confusión — Test\nAccuracy: {acc:.1%}", pad=12)
    thr = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(N_CLASES):
        for j in range(N_CLASES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold",
                    color="white" if cm[i, j] > thr else "black")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1E7B45", lw=2.5))

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    headers = ["Clase", "Precision", "Recall", "F1", "N"]
    filas = [[CLASES[i], f"{met[i]['precision']:.3f}", f"{met[i]['recall']:.3f}",
              f"{met[i]['f1']:.3f}", str(met[i]['support'])] for i in range(N_CLASES)]
    macro = [np.mean([met[i][k] for i in range(N_CLASES)]) for k in ("precision", "recall", "f1")]
    filas.append(["macro avg", f"{macro[0]:.3f}", f"{macro[1]:.3f}", f"{macro[2]:.3f}", ""])
    tbl = ax2.table(cellText=filas, colLabels=headers, loc="center", bbox=[0.0, 0.3, 1.0, 0.6])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F3864"); cell.set_text_props(color="white", fontweight="bold")
        elif r == len(filas):
            cell.set_facecolor("#D6E4F0"); cell.set_text_props(fontweight="bold")
    plt.suptitle("MobileNetV3Large — Evaluación en test", fontweight="bold", color="#1F3864")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")


def plot_roc(y_true, y_prob, out="roc_curves.png"):
    y_bin = np.eye(N_CLASES)[y_true]
    paneles = N_CLASES + 1
    cols = 2; rows = math.ceil(paneles / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5 * rows))
    axes_flat = list(np.array(axes).flat)
    aucs = {}
    for i, (clase, color) in enumerate(zip(CLASES, COLORES)):
        ax = axes_flat[i]
        fpr, tpr, auc = roc_manual(y_bin[:, i], y_prob[:, i]); aucs[clase] = auc
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {clase}", fontweight="bold", color=color); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax_all = axes_flat[N_CLASES]
    for clase, color in zip(CLASES, COLORES):
        fpr, tpr, auc = roc_manual(y_bin[:, CLASES.index(clase)], y_prob[:, CLASES.index(clase)])
        ax_all.plot(fpr, tpr, color=color, lw=2, label=f"{clase} (AUC={auc:.3f})")
    ax_all.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_all.set_title(f"Todas las clases\nMacro-avg AUC = {np.mean(list(aucs.values())):.3f}", fontweight="bold")
    ax_all.set_xlabel("FPR"); ax_all.set_ylabel("TPR"); ax_all.legend(loc="lower right", fontsize=9); ax_all.grid(alpha=0.3)
    for k in range(paneles, len(axes_flat)):
        axes_flat[k].axis("off")
    plt.suptitle("MobileNetV3Large — Curvas ROC (One-vs-Rest)", fontweight="bold", color="#1F3864")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")
    return aucs


# ── 7. MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    tf.random.set_seed(SEED); np.random.seed(SEED)

    train_ds, val_ds, test_ds = cargar_dataset()
    print("\nConstruyendo modelo...")
    modelo, base = construir_modelo()
    print(f"  Parámetros totales    : {modelo.count_params():>12,}")

    historial = entrenar(modelo, base, train_ds, val_ds)

    print("\nEvaluando en test...")
    y_true, y_pred, y_prob = predecir_test(modelo, test_ds)
    cm  = calcular_cm(y_true, y_pred, N_CLASES)
    met = metricas_por_clase(cm)
    acc = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0.0

    print(f"\n{'─'*58}\n{'Clase':<14}{'Precision':>10}{'Recall':>8}{'F1':>8}{'N':>6}\n{'─'*58}")
    for i, clase in enumerate(CLASES):
        m = met[i]
        print(f"{clase:<14}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>6}")
    print(f"{'─'*58}\nAccuracy: {acc:.4f}  ({acc:.1%})")

    print("\nGenerando visualizaciones...")
    plot_curvas(historial); plot_cm(cm, met, acc); plot_roc(y_true, y_prob)

    modelo.save(MODELO_OUT)
    print(f"\nModelo guardado: {MODELO_OUT}")
    print("Completado")
