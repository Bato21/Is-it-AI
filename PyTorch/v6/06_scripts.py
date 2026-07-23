"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 6  (búsqueda de hiperparámetros de la CABEZA con Optuna)

Propósito experimental:
  La v4 fijó la arquitectura (MobileNetV3-Small congelada + cabeza Linear(576->3)) y
  entrenó con hiperparámetros elegidos A MANO (Adam por defecto, dropout 0.2, batch 32).
  La v6 hace UN solo cambio conceptual: en vez de elegir esos hiperparámetros a ojo, se
  BUSCAN automáticamente con Optuna, maximizando val_accuracy.

  MISMO backbone que la v4 (feature extraction: MobileNetV3-Small preentrenada, congelada),
  MISMO dataset balanceado (dataset/), MISMO split 80/20 y seed. Lo único que cambia es que
  la cabeza (dropout, capa oculta, learning rate, optimizador) ya no es fija: es lo que
  Optuna explora trial a trial.

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

CONTRASTE DE FRAMEWORKS (espejo de TensorFlow/v6/06_scripts.py):
  El ESPACIO de búsqueda y la lógica de Optuna son idénticos (Optuna no depende del
  framework: recibe un trial y un número a maximizar). Lo que cambia es cómo se arma y
  entrena la cabeza dentro de objective(): en PyTorch es nn.Sequential + bucle manual de
  entrenamiento; en Keras es Sequential + compile + fit. El pre-cómputo de embeddings
  también difiere: acá es un forward con torch.no_grad() sobre features+avgpool del
  backbone (OJO con el gotcha de BatchNorm: hay que poner el backbone en eval()).
"""

import collections
import copy
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

import matplotlib
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRIALS = 30              # combinaciones a probar (igual que el ejemplo del profe)
EPOCHS_TRIAL = 40          # épocas máx por trial (con early stopping cortan antes)
EPOCHS_FINAL = 80          # épocas del reentrenamiento con los mejores hiperparámetros
PATIENCE = 8

# Normalización ImageNet (igual que la v4): va en el TRANSFORM, fuera del modelo.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
transform_eval = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

# --- 2. Carga de datos (split 80/20 IDÉNTICO a v4, mismo seed) ---
torch.manual_seed(SEED)
base = datasets.ImageFolder(DATA_DIR)
class_names = base.classes
print("Orden de clases:", class_names)  # ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']
num_classes = len(class_names)

n_val = max(1, int(len(base) * 0.2))
n_train = len(base) - n_val
idx_train, idx_val = random_split(
    range(len(base)), [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)
idx_train, idx_val = list(idx_train), list(idx_val)

# NOTA: para pre-computar embeddings usamos SOLO transform_eval (sin augmentation): las
# features del backbone congelado tienen que ser deterministas para que Optuna compare
# trials sobre los mismos vectores. La augmentation se re-introduciría en un fine-tuning.
ds_full = datasets.ImageFolder(DATA_DIR, transform=transform_eval)
ds_train = Subset(ds_full, idx_train)
ds_val = Subset(ds_full, idx_val)


# --- 3. Backbone congelado + PRE-CÓMPUTO de embeddings ---
# GOTCHA DE BATCHNORM (contraste de frameworks, ver v4): hay que poner el backbone en
# eval() para que BatchNorm use sus estadísticas de ImageNet y no las recalcule con
# nuestros datos. En Keras esto es automático con trainable=False.
def construir_extractor() -> nn.Module:
    modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    extractor = nn.Sequential(modelo.features, modelo.avgpool, nn.Flatten())
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor.to(DEVICE).eval()


print("\nPre-computando embeddings del backbone congelado (una sola vez)...")
extractor = construir_extractor()


def embeddings_de(ds) -> tuple[torch.Tensor, torch.Tensor]:
    dl = DataLoader(ds, batch_size=BATCH_SIZE)
    xs, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            xs.append(extractor(x.to(DEVICE)).cpu())
            ys.append(y)
    return torch.cat(xs), torch.cat(ys)


X_train, y_train = embeddings_de(ds_train)
X_val, y_val = embeddings_de(ds_val)
EMB_DIM = X_train.shape[1]
print(f"  embeddings train: {tuple(X_train.shape)}  ·  val: {tuple(X_val.shape)}  (dim {EMB_DIM})")

# DataLoaders sobre los embeddings (MLP chica, ya no imágenes).
dl_train = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

# Soporte por clase (para leer los recalls en absolutos, como en v5).
val_counts = collections.Counter(int(v) for v in y_val)
print("Soporte val por clase:", {class_names[i]: val_counts[i] for i in range(num_classes)})


# --- 4. Reporte por clase (a mano desde la matriz de confusión, igual que v4/v5) ---
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
# ESPACIO idéntico al lado TF (Optuna no depende del framework):
#   n_capas    int  [1, 2]                capas densas ocultas antes del softmax
#   units      cat  {64, 128, 256}        neuronas por capa oculta
#   dropout    float[0.1, 0.6] step 0.05  regularización de la cabeza
#   optimizer  cat  {adam, rmsprop, sgd}
#   lr         float[1e-4, 1e-2] log      learning rate
def construir_cabeza(params) -> nn.Module:
    capas, entrada = [], EMB_DIM
    for _ in range(params["n_capas"]):
        capas += [nn.Linear(entrada, params["units"]), nn.ReLU(), nn.Dropout(params["dropout"])]
        entrada = params["units"]
    capas.append(nn.Linear(entrada, num_classes))
    return nn.Sequential(*capas).to(DEVICE)


def crear_optimizer(cabeza, params):
    lr = params["lr"]
    if params["optimizer"] == "adam":
        return torch.optim.Adam(cabeza.parameters(), lr=lr)
    if params["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(cabeza.parameters(), lr=lr)
    return torch.optim.SGD(cabeza.parameters(), lr=lr, momentum=0.9)


def evaluar(cabeza) -> tuple[float, np.ndarray]:
    cabeza.eval()
    correctos, total = 0, 0
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for x, y in dl_val:
            pred = cabeza(x.to(DEVICE)).argmax(1).cpu()
            correctos += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.numpy(), pred.numpy()):
                cm[t, p] += 1
    return correctos / total, cm


def entrenar_cabeza(params, epochs, paciencia):
    torch.manual_seed(SEED)
    cabeza = construir_cabeza(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = crear_optimizer(cabeza, params)
    mejor_acc, mejor_state, sin_mejora = 0.0, copy.deepcopy(cabeza.state_dict()), 0
    for _ in range(epochs):
        cabeza.train()
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(cabeza(x), y)
            loss.backward()
            optimizer.step()
        va_acc, _ = evaluar(cabeza)
        if va_acc > mejor_acc:
            mejor_acc, mejor_state, sin_mejora = va_acc, copy.deepcopy(cabeza.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    cabeza.load_state_dict(mejor_state)
    return cabeza, mejor_acc


def objective(trial):
    params = {
        "n_capas": trial.suggest_int("n_capas", 1, 2),
        "units": trial.suggest_categorical("units", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6, step=0.05),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }
    _, mejor_acc = entrenar_cabeza(params, EPOCHS_TRIAL, PATIENCE)
    return mejor_acc


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


# --- 7. Figuras de Optuna (val_acc por trial + importancia de hiperparámetros) ---
def plot_historia(study, salida):
    trials = [t for t in study.trials if t.value is not None]
    nums = [t.number for t in trials]
    vals = [t.value for t in trials]
    mejores = [max(vals[: i + 1]) for i in range(len(vals))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(nums, vals, color="#B45309", alpha=0.75, width=0.7)
    ax1.plot(nums, mejores, "r-o", ms=4, lw=2, label="mejor acumulado")
    ax1.set_xlabel("trial"); ax1.set_ylabel("val_accuracy")
    ax1.set_title("val_accuracy por trial"); ax1.legend(); ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax2.hist(vals, bins=10, color="#B45309", alpha=0.75, edgecolor="white")
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
    except Exception as e:
        print(f"  (importancia no disponible: {e})")
        return
    nombres = list(imp.keys())[::-1]
    valores = list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(nombres, valores, color="#B45309", edgecolor="white", height=0.6)
    for i, v in enumerate(valores):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlabel("importancia (fANOVA)")
    ax.set_title("Importancia de hiperparámetros\nmayor = más impacto en val_accuracy",
                 fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {salida.name}")


# --- 8. Main ---
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

    with open(AQUI / "mejores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump({"val_accuracy": study.best_value, "params": study.best_params}, f,
                  indent=2, ensure_ascii=False)
    print("\n  guardado mejores_hiperparametros.json")

    print("\n" + "-" * 66)
    print("  Reentrenando con los mejores hiperparámetros")
    print("-" * 66)
    cabeza, val_acc = entrenar_cabeza(study.best_params, EPOCHS_FINAL, PATIENCE + 4)
    _, cm = evaluar(cabeza)
    print(f"val_accuracy final: {val_acc:.4f}")
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    print("\nReporte por clase:")
    reporte_por_clase(cm, class_names)

    print("\nGenerando figuras...")
    plot_historia(study, AQUI / "Figure_optuna_historia_v6.png")
    plot_importancia(study, AQUI / "Figure_optuna_importancia_v6.png")

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

    # Guardar el modelo final de INFERENCIA: backbone congelado + cabeza óptima.
    # Se reconstruye el extractor a CPU para un checkpoint portable; el consumo replica
    # el mismo Normalize del transform (contraste con TF, que lo lleva dentro del modelo).
    MODEL_PATH = AQUI / "modelopt_v6_optuna.pt"
    torch.save(
        {
            "cabeza_state_dict": cabeza.state_dict(),
            "best_params": study.best_params,
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "emb_dim": EMB_DIM,
            "arch": "mobilenet_v3_small",
            "normalize_mean": NORMALIZE_MEAN,
            "normalize_std": NORMALIZE_STD,
        },
        MODEL_PATH,
    )
    print(f"\nModelo final guardado en {MODEL_PATH.name}")
