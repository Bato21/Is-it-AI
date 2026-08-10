"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 7  (PROTOCOLO DE EVALUACIÓN HONESTO)

Propósito experimental:
  La v6 buscó hiperparámetros con Optuna y reportó 93.3% de val_accuracy. Ese número
  tiene un problema metodológico serio, y la v7 existe para medirlo:

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

  3. SPLIT ESTRATIFICADO Y COMPARTIDO con TensorFlow. Hasta la v6, TF partía con
     image_dataset_from_directory y PyTorch con random_split: splits distintos, así que
     comparar 90.0% (PT v4) contra 80.0% (TF v4) tenía un asterisco. Ahora los dos leen
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
    todas en documentacion/metricas.py — el mismo código que usa el lado TensorFlow.
  - Resultados agregados sobre 5 semillas (media ± std) + figura de dispersión.

CONTRASTE DE FRAMEWORKS (espejo de TensorFlow/v7/07_scripts.py):
  El protocolo, la partición, el espacio de búsqueda y las métricas son IDÉNTICOS —
  literalmente el mismo JSON de partición y el mismo módulo de métricas. Lo que sigue
  difiriendo es la mecánica: acá la cabeza es nn.Sequential con bucle de entrenamiento
  manual y early stopping a mano (copia del state_dict); en Keras es Sequential +
  compile + fit con el callback EarlyStopping. El pre-cómputo de embeddings acá es un
  forward con torch.no_grad() sobre features+avgpool, con el backbone forzado a eval()
  por el gotcha de BatchNorm (ver v4); en Keras basta trainable=False.
"""

import copy
import json
import statistics
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

import numpy as np
import optuna
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Módulos COMPARTIDOS con TensorFlow (numpy puro, sin dependencias de framework).
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402
from particion_datos import cargar_particion, conteos, rutas_y_etiquetas  # noqa: E402

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRIALS = 30              # igual que la v6, para que el A/B sea limpio
EPOCHS_TRIAL = 40          # techo por fold (el early stopping corta antes)
PATIENCE = 8
SEMILLAS = [123, 7, 42, 2024, 31]   # 5 corridas del refit; la 1a es la de siempre

# Normalización ImageNet: va en el TRANSFORM (fuera del modelo), como en v4/v6.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Sin augmentation: las features del backbone congelado tienen que ser DETERMINISTAS
# para poder pre-computarlas una sola vez y que todos los trials comparen sobre los
# mismos vectores (mismo criterio que la v6).
transform_eval = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


# --- 2. Carga de datos desde la partición canónica ---
class ImagenesDeLista(Dataset):
    """Dataset a partir de una LISTA EXPLÍCITA de rutas, no de una carpeta.

    CONTRASTE con v1-v6: ahí se usaba ImageFolder, que barre el directorio y arma el
    orden por su cuenta. Acá las rutas vienen de particion.json, que es el mismo
    archivo que lee TensorFlow. Ese cambio es lo que garantiza que los dos frameworks
    entrenen y evalúen exactamente sobre las mismas imágenes.
    """

    def __init__(self, raiz: Path, rutas: list[str], etiquetas: list[int], transform):
        self.raiz, self.rutas, self.etiquetas, self.transform = raiz, rutas, etiquetas, transform

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, i):
        # convert("RGB") es obligatorio: hay PNGs con canal alfa en el dataset y el
        # backbone espera 3 canales. Sin esto, revienta con imágenes RGBA.
        img = Image.open(self.raiz / self.rutas[i]).convert("RGB")
        return self.transform(img), self.etiquetas[i]


particion = cargar_particion()
class_names = particion["clases"]
print("Orden de clases:", class_names)  # ['0_sin_ia', '1_rastro_ia', '2_saturada_ia']
num_classes = len(class_names)
N_FOLDS = particion["meta"]["n_folds"]

rutas_dev, y_dev = rutas_y_etiquetas(particion, "desarrollo")
rutas_test, y_test_lista = rutas_y_etiquetas(particion, "test")
print(f"\nPartición canónica (documentacion/particion.json, huella "
      f"{particion['meta']['huella_dataset']}):")
print(f"  desarrollo : {len(rutas_dev):>3} imgs  {conteos(particion, rutas_dev)}")
print(f"  test       : {len(rutas_test):>3} imgs  {conteos(particion, rutas_test)}   "
      f"<- APARTADO, se usa una sola vez")
print(f"  folds      : {N_FOLDS} (validación cruzada dentro de desarrollo)")


# --- 3. Backbone congelado + PRE-CÓMPUTO de embeddings ---
# El backbone no cambia nunca (feature extraction), así que sus features se calculan
# UNA vez para las 313 imágenes y todos los trials, folds y semillas entrenan una MLP
# chica sobre esos vectores. Es lo que hace que 30 trials x 5 folds x 5 semillas corra
# en minutos en CPU en vez de en horas.
def construir_extractor() -> nn.Module:
    modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    extractor = nn.Sequential(modelo.features, modelo.avgpool, nn.Flatten())
    for p in extractor.parameters():
        p.requires_grad = False
    # GOTCHA DE BATCHNORM (ver v4): eval() para que BatchNorm use las estadísticas de
    # ImageNet y no las recalcule con nuestros datos. En Keras es automático con
    # trainable=False; acá hay que pedirlo explícitamente.
    return extractor.to(DEVICE).eval()


print("\nPre-computando embeddings del backbone congelado (una sola vez)...")
extractor = construir_extractor()


def embeddings_de(rutas: list[str], etiquetas: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    ds = ImagenesDeLista(DATA_DIR, rutas, etiquetas, transform_eval)
    dl = DataLoader(ds, batch_size=BATCH_SIZE)
    xs, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            xs.append(extractor(x.to(DEVICE)).cpu())
            ys.append(y)
    return torch.cat(xs), torch.cat(ys)


X_dev, y_dev_t = embeddings_de(rutas_dev, y_dev)
X_test, y_test_t = embeddings_de(rutas_test, y_test_lista)
EMB_DIM = X_dev.shape[1]
print(f"  desarrollo: {tuple(X_dev.shape)}  ·  test: {tuple(X_test.shape)}  (dim {EMB_DIM})")

# Índices de cada fold DENTRO de X_dev: rutas_dev es la concatenación de los folds en
# orden, así que basta con recorrer los folds y mapear ruta -> posición.
pos_de_ruta = {r: i for i, r in enumerate(rutas_dev)}
idx_folds = [
    np.array([pos_de_ruta[r] for r in rutas_y_etiquetas(particion, "validacion", fold=k)[0]])
    for k in range(N_FOLDS)
]


# --- 4. Cabeza, optimizador y bucle de entrenamiento ---
# ESPACIO DE BÚSQUEDA IDÉNTICO AL DE LA v6 (a propósito: si cambiara el espacio, el A/B
# v6 vs v7 mezclaría "mejor protocolo" con "mejor búsqueda" y no se podría atribuir la
# diferencia a nada). Los hiperparámetros nuevos (weight_decay, label_smoothing) entran
# recién en la v8, junto con la destilación.
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


def evaluar_acc(cabeza, X, y) -> float:
    cabeza.eval()
    with torch.no_grad():
        pred = cabeza(X.to(DEVICE)).argmax(1).cpu()
    return float((pred == y).float().mean())


def probabilidades(cabeza, X) -> np.ndarray:
    """Vector COMPLETO de probabilidades softmax (no solo el argmax).

    Es una regla del proyecto desde la v2 —toda inferencia devuelve el vector— y acá
    además es un requisito técnico: ROC-AUC y ECE necesitan el score continuo, no la
    decisión dura.
    """
    cabeza.eval()
    with torch.no_grad():
        return torch.softmax(cabeza(X.to(DEVICE)), dim=1).cpu().numpy()


def entrenar_cabeza(params, X_tr, y_tr, X_va, y_va, epochs, paciencia, semilla):
    """Entrena la cabeza con early stopping manual sobre val_accuracy.

    Devuelve (cabeza con los mejores pesos restaurados, mejor val_acc, mejor época).
    La "mejor época" se devuelve porque el refit final la necesita: cuando se
    reentrena sobre TODO desarrollo ya no queda validación con la cual cortar, así
    que se usa la mediana de las mejores épocas de los 5 folds (receta estándar de
    refit tras validación cruzada).

    Si X_va es None se entrena a ciegas por `epochs` épocas: ese es exactamente el
    caso del refit final.
    """
    torch.manual_seed(semilla)
    cabeza = construir_cabeza(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = crear_optimizer(cabeza, params)
    dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(semilla))

    mejor_acc, mejor_state, mejor_epoca, sin_mejora = 0.0, copy.deepcopy(cabeza.state_dict()), 0, 0
    for epoca in range(1, epochs + 1):
        cabeza.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(cabeza(x), y).backward()
            optimizer.step()
        if X_va is None:
            continue  # refit a ciegas: sin validación, sin early stopping
        va_acc = evaluar_acc(cabeza, X_va, y_va)
        if va_acc > mejor_acc:
            mejor_acc, mejor_epoca, sin_mejora = va_acc, epoca, 0
            mejor_state = copy.deepcopy(cabeza.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if X_va is not None:
        cabeza.load_state_dict(mejor_state)
    return cabeza, mejor_acc, mejor_epoca


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
        X_tr, y_tr = X_dev[mascara], y_dev_t[mascara]
        X_va, y_va = X_dev[idx_va], y_dev_t[idx_va]

        _, acc, epoca = entrenar_cabeza(params, X_tr, y_tr, X_va, y_va,
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
    import matplotlib.pyplot as plt
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    nums = [t.number for t in trials]
    vals = [t.value for t in trials]
    mejores = [max(vals[: i + 1]) for i in range(len(vals))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(nums, vals, color="#B45309", alpha=0.75, width=0.7)
    ax1.plot(nums, mejores, "r-o", ms=4, lw=2, label="mejor acumulado")
    ax1.set_xlabel("trial"); ax1.set_ylabel(f"media de {N_FOLDS}-fold CV")
    ax1.set_title("accuracy de CV por trial"); ax1.legend(); ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax2.hist(vals, bins=10, color="#B45309", alpha=0.75, edgecolor="white")
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
    import matplotlib.pyplot as plt
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
    for i, semilla in enumerate(SEMILLAS):
        cabeza, _, _ = entrenar_cabeza(study.best_params, X_dev, y_dev_t, None, None,
                                       EPOCHS_REFIT, PATIENCE, semilla)
        y_prob = probabilidades(cabeza, X_test)
        r = metricas.resumen_completo(y_test_t.numpy(), y_prob, class_names)
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
    metricas.imprimir_agregado(agregado, "RESULTADOS SOBRE TEST — PyTorch v7")

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
    print(f"  v6 (PyTorch): 0.9333 de val_accuracy, pero ese val es el MISMO conjunto")
    print(f"                que Optuna maximizó durante 30 trials -> sesgo de selección.")
    print(f"  v7 (PyTorch): {acc_v7:.4f} ± {agregado['accuracy']['std']:.4f} sobre un test")
    print(f"                de {len(rutas_test)} imágenes que la búsqueda NUNCA vio.")
    print(f"  Diferencia  : {acc_v7 - 0.9333:+.4f}  <- esto es lo que el protocolo viejo")
    print(f"                estaba regalando (o no, si sale positivo: entonces el modelo")
    print(f"                era genuinamente bueno y la v6 no estaba tan inflada).")

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
    curvas, micro, macro = metricas.curvas_roc_ovr(y_test_t.numpy(), y_prob_0, num_classes)
    print(f"\nROC-AUC One-vs-Rest (test, semilla {SEMILLAS[0]}):")
    for i in range(num_classes):
        print(f"  {class_names[i]:<16} AUC = {curvas[i][2]:.4f}")
    print(f"  {'macro-promedio':<16} AUC = {macro:.4f}")
    print(f"  {'micro-promedio':<16} AUC = {micro[2]:.4f}")

    metricas.plot_roc(curvas, micro, macro, class_names,
                      f"Curvas ROC One-vs-Rest — v7 (test, semilla {SEMILLAS[0]})",
                      AQUI / "Figure_roc_v7.png")
    metricas.plot_calibracion(y_test_t.numpy(), y_prob_0,
                              f"Calibración — v7 (test, semilla {SEMILLAS[0]})",
                              AQUI / "Figure_calibracion_v7.png")
    metricas.plot_semillas(resumenes, ["accuracy", "macro_f1", "qwk", "ece"],
                           f"Dispersión entre semillas — PyTorch v7 (test, "
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
            "framework": "pytorch",
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

    # --- Checkpoint del modelo de inferencia (backbone congelado + cabeza) ---
    # Se guarda la cabeza de la PRIMERA semilla, decidido de antemano. Guardar "la
    # mejor de las 5 según el test" sería volver a elegir mirando el test.
    MODEL_PATH = AQUI / "modelopt_v7_protocolo.pt"
    torch.save({
        "cabeza_state_dict": cabezas[0].state_dict(),
        "best_params": study.best_params,
        "semilla": SEMILLAS[0],
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "emb_dim": EMB_DIM,
        "arch": "mobilenet_v3_small",
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
        "huella_particion": particion["meta"]["huella_dataset"],
    }, MODEL_PATH)
    print(f"  guardado {MODEL_PATH.name}  (semilla {SEMILLAS[0]}, elegida de antemano)")
