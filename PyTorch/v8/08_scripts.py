"""
Clasificador de huella de IA en diapositivas — PyTorch
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
  Un profesor sobreajustado no enseña casi nada. La v1 llega a train_acc 0.996: sobre
  las imágenes que memorizó, su softmax es prácticamente one-hot, el término KL degenera
  en cross-entropy y la destilación no transfiere "conocimiento oscuro" ninguno. El
  conocimiento oscuro solo existe donde el profesor DUDA.

  Solución: destilar sobre VISTAS AUMENTADAS. El profesor nunca vio esas rotaciones y
  cambios de brillo, así que ahí sí produce una distribución genuinamente blanda. Y
  encaja con el relato: lo aumentado es justamente lo que aporta la v2.

  Para que además siga siendo barato, las vistas aumentadas se generan UNA sola vez con
  semilla fija (no se re-sortean por época). Eso permite pre-computar los embeddings de
  MobileNet y los logits del profesor sobre TODAS las vistas, y que los eslabones E4/E5
  entrenen en segundos. Se pierde algo de diversidad de augmentation a cambio de que el
  experimento corra en minutos y sea 100% reproducible: es un tradeoff consciente.

Nuevo respecto a la v7:
  - Pérdida de destilación KL + CE con temperatura (perdida_destilacion).
  - Conjunto de vistas fijo (1 limpia + 2 aumentadas por imagen).
  - Cadena de 5 eslabones con profesores encadenados.
  - Barrido de sensibilidad de T y alpha, evaluado SOBRE CV (nunca sobre el test).
  - Comparación A/B de los dos brazos, 5 semillas, sobre el test apartado de la v7.

CONTRASTE DE FRAMEWORKS (espejo de TensorFlow/v8/08_scripts.py):
  La pérdida de destilación es el contraste más jugoso de todo el proyecto. Acá se arma
  con F.kl_div sobre log_softmax(estudiante/T) y softmax(profesor/T), y el modelo
  devuelve LOGITS crudos, así que dividir por la temperatura es directo. En Keras la
  cabeza termina en softmax, así que para poder aplicar temperatura hay que construirla
  SIN la activación final y aplicar el softmax a mano, o recuperar los logits con
  tf.math.log. Es la diferencia estructural "logits vs probabilidades" que el proyecto
  viene marcando desde la v1 y que acá, por fin, tiene consecuencias reales.
"""

import copy
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Módulos COMPARTIDOS con TensorFlow (numpy puro, sin dependencias de framework).
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402
from particion_datos import cargar_particion, conteos, rutas_y_etiquetas  # noqa: E402

# --- 1. Parámetros ---
DATA_DIR = RAIZ / "dataset"
if not DATA_DIR.is_dir():
    raise SystemExit(f"No existe la carpeta de datos: {DATA_DIR}")

SIZE_CNN = (180, 180)      # entrada de la CNN propia (v1-v3)
SIZE_MNET = (224, 224)     # entrada de MobileNetV3 (v4+)
BATCH_SIZE = 32
SEED = 123
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_VISTAS_AUG = 2           # vistas aumentadas por imagen (+1 limpia = 3 por imagen)
SEED_VISTAS = 999          # semilla propia de las vistas: fija para TODAS las semillas
                           # de entrenamiento, así los dos brazos ven EXACTAMENTE los
                           # mismos datos y la única diferencia es cómo se entrena.

# Hiperparámetros de destilación. Son los del script del profe (destilacion_cross_arch.py,
# Escenario 1): TEMPERATURA=4.0, ALPHA=0.7. Se FIJAN en vez de buscarlos con Optuna a
# propósito: si el brazo B pudiera buscar dos hiperparámetros extra que el brazo A no
# tiene, ganaría por tener más búsqueda, no por destilar. El barrido de sensibilidad
# (§6) explora T y alpha, pero se mide sobre CV y NO se usa para elegir el modelo final.
TEMPERATURA = 4.0
ALPHA = 0.7

SEMILLAS = [123, 7, 42, 2024, 31]   # las MISMAS de la v7, para poder comparar

# Épocas por eslabón. Fijas y declaradas de antemano: dentro de la cadena no hay un
# conjunto de validación con el que hacer early stopping sin quitarle datos a alguno de
# los dos brazos, así que el "converger" de la v3 se representa con un presupuesto de
# épocas mayor (40) frente al recorte arbitrario de la v1 (10). Es una simplificación
# consciente y está anotada en explicacion.md.
EPOCAS = {"E1": 10, "E2": 30, "E3": 40, "E4": 30}
LR_CNN = 1e-3


# --- 2. Vistas: 1 limpia + N aumentadas por imagen, generadas UNA vez ---
# La augmentation es la MISMA de v2/v3/v4 (RandomAffine 18° + scale 0.9-1.1, ColorJitter
# brillo/contraste 0.2). Sin flip: una diapositiva espejada tiene el texto en espejo.
aug = transforms.Compose([
    transforms.RandomAffine(degrees=18, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
a_tensor_mnet = transforms.Compose([
    transforms.Resize(SIZE_MNET),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

particion = cargar_particion()
class_names = particion["clases"]
print("Orden de clases:", class_names)
num_classes = len(class_names)

rutas_dev, y_dev = rutas_y_etiquetas(particion, "desarrollo")
rutas_test, y_test = rutas_y_etiquetas(particion, "test")
print(f"\nPartición canónica (huella {particion['meta']['huella_dataset']}):")
print(f"  desarrollo : {len(rutas_dev):>3} imgs  {conteos(particion, rutas_dev)}")
print(f"  test       : {len(rutas_test):>3} imgs  {conteos(particion, rutas_test)}   "
      f"<- el MISMO test apartado de la v7")


def construir_extractor() -> nn.Module:
    """MobileNetV3-Small congelada, en eval() por el gotcha de BatchNorm (ver v4)."""
    modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    ext = nn.Sequential(modelo.features, modelo.avgpool, nn.Flatten())
    for p in ext.parameters():
        p.requires_grad = False
    return ext.to(DEVICE).eval()


extractor = construir_extractor()


def preparar_vistas(rutas, etiquetas, n_aug, semilla):
    """Genera las vistas y devuelve (imgs 180px uint8, embeddings 576-d, etiquetas).

    Cada imagen produce 1 vista LIMPIA + n_aug vistas AUMENTADAS. La augmentation se
    aplica sobre la imagen original y recién después se hacen los DOS resize (180 para
    la CNN, 224 para MobileNet), así los dos tamaños comparten la misma instancia de
    augmentation: el profesor CNN y el estudiante MobileNet ven la MISMA vista, que es
    lo que exige la destilación (si vieran vistas distintas, los logits del profesor no
    corresponderían a la entrada del estudiante).

    Las imágenes de 180px se guardan en uint8 (73 MB para 750 vistas) y se normalizan
    por batch: en float32 serían 292 MB y esta máquina tiene poca RAM libre.
    """
    torch.manual_seed(semilla)
    imgs180, embs, ys = [], [], []
    buffer_mnet = []

    def volcar_buffer():
        if not buffer_mnet:
            return
        lote = torch.stack(buffer_mnet).to(DEVICE)
        with torch.no_grad():
            embs.append(extractor(lote).cpu())
        buffer_mnet.clear()

    for ruta, etiqueta in zip(rutas, etiquetas):
        original = Image.open(DATA_DIR / ruta).convert("RGB")
        for v in range(n_aug + 1):
            img = original if v == 0 else aug(original)
            # 180px uint8 para la CNN (v1-v3 solo hacen /255, sin normalización ImageNet)
            # np.array (no np.asarray): asarray devuelve una vista de solo lectura del
            # buffer de PIL y torch.from_numpy avisa que el tensor no es escribible.
            arr = np.array(img.resize(SIZE_CNN[::-1], Image.BILINEAR), dtype=np.uint8)
            imgs180.append(torch.from_numpy(arr).permute(2, 0, 1))
            buffer_mnet.append(a_tensor_mnet(img))
            ys.append(etiqueta)
            if len(buffer_mnet) >= BATCH_SIZE:
                volcar_buffer()
    volcar_buffer()

    return torch.stack(imgs180), torch.cat(embs), torch.tensor(ys, dtype=torch.long)


print(f"\nGenerando vistas (1 limpia + {N_VISTAS_AUG} aumentadas por imagen, "
      f"semilla fija {SEED_VISTAS})...")
t0 = time.time()
X180_dev, XE_dev, Y_dev = preparar_vistas(rutas_dev, y_dev, N_VISTAS_AUG, SEED_VISTAS)
# El TEST no se aumenta: se evalúa sobre la imagen limpia, como en toda versión anterior.
X180_test, XE_test, Y_test = preparar_vistas(rutas_test, y_test, 0, SEED_VISTAS)
EMB_DIM = XE_dev.shape[1]
print(f"  dev : {tuple(X180_dev.shape)} a 180px  ·  {tuple(XE_dev.shape)} embeddings")
print(f"  test: {tuple(X180_test.shape)} a 180px  ·  {tuple(XE_test.shape)} embeddings")
print(f"  ({time.time() - t0:.0f}s)")

# Máscara de las vistas LIMPIAS de dev (el eslabón E1 no usa augmentation, como la v1).
mascara_limpias = torch.zeros(len(Y_dev), dtype=torch.bool)
mascara_limpias[::N_VISTAS_AUG + 1] = True


# --- 3. Modelos ---
class BaselineCNN(nn.Module):
    """La MISMA CNN de v1/v2/v3, sin tocar una línea.

    Conv(16)->Pool->Conv(32)->Pool->Flatten->Dense(64)->Dense(3). Con entrada 180x180 el
    flatten da 32*43*43 = 59168. Se reusa tal cual para que los eslabones E1-E3 de la
    cadena sean literalmente los modelos de las versiones que representan.
    """

    def __init__(self, num_clases: int):
        super().__init__()
        # Sin padding, igual que v1/v2/v3 (y que el padding='valid' por defecto de Keras):
        # 180 -> conv 178 -> pool 89 -> conv 87 -> pool 43  =>  32*43*43 = 59168.
        self.red = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 43 * 43, 64), nn.ReLU(),
            nn.Linear(64, num_clases),
        )

    def forward(self, x):
        return self.red(x)


def cabeza_v4() -> nn.Module:
    """Cabeza tipo v4: réplica del classifier de MobileNetV3-Small sobre el embedding.

    MobileNetV3-Small trae Linear(576,1024) -> Hardswish -> Dropout(0.2) -> Linear(1024,3).
    Se reproduce igual para que el eslabón E4 sea de verdad "la v4" y no una cabeza nueva.
    """
    return nn.Sequential(
        nn.Linear(EMB_DIM, 1024), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1024, num_classes)
    ).to(DEVICE)


def cabeza_v7(params) -> nn.Module:
    """Cabeza con los hiperparámetros que Optuna eligió en la v7 (por CV, sin ver el test)."""
    capas, entrada = [], EMB_DIM
    for _ in range(params["n_capas"]):
        capas += [nn.Linear(entrada, params["units"]), nn.ReLU(), nn.Dropout(params["dropout"])]
        entrada = params["units"]
    capas.append(nn.Linear(entrada, num_classes))
    return nn.Sequential(*capas).to(DEVICE)


def optimizador_v7(modelo, params):
    lr = params["lr"]
    if params["optimizer"] == "adam":
        return torch.optim.Adam(modelo.parameters(), lr=lr)
    if params["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(modelo.parameters(), lr=lr)
    return torch.optim.SGD(modelo.parameters(), lr=lr, momentum=0.9)


# --- 4. LA PÉRDIDA DE DESTILACIÓN (el corazón de la v8) ---
def perdida_destilacion(logits_est, logits_prof, y, T: float, alpha: float):
    """alpha * KL(profesor || estudiante) * T^2  +  (1-alpha) * CE(estudiante, etiquetas).

    Calcada del `perdida_kl` del script del profe (destilacion_cross_arch.py, Escenario 1),
    adaptada de un vocabulario de 50.257 tokens a nuestras 3 clases. Dos diferencias:
      - Allá había que desplazar los logits (shift) porque es predicción del token
        siguiente; acá es clasificación de imagen, no hay secuencia que desplazar.
      - Allá la CE ignoraba el token de padding; acá no hay padding.

    Por qué cada pieza:
      - Dividir por T APLASTA la distribución y saca a la luz las probabilidades chicas.
        Si el profesor dice [0.98, 0.02, 0.00], con T=4 dice algo como [0.55, 0.28, 0.17]:
        ahí aparece que "esta diapo, si no fuera 0, sería más 1 que 2". Eso es el
        conocimiento oscuro, y es lo que una etiqueta dura nunca transmite.
      - Multiplicar por T^2 compensa que los gradientes del término KL escalan como 1/T^2;
        sin eso, subir la temperatura apagaría la destilación sin querer.
      - El término CE mantiene al estudiante anclado a las etiquetas reales: si el
        profesor se equivoca, la verdad del dataset sigue tirando en la dirección correcta.
    """
    log_p_est = F.log_softmax(logits_est / T, dim=1)
    p_prof = F.softmax(logits_prof / T, dim=1)
    kl = F.kl_div(log_p_est, p_prof, reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(logits_est, y)
    return alpha * kl + (1 - alpha) * ce


# --- 5. Entrenamiento de un eslabón ---
def normalizar180(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] -> float [0,1]. Es el equivalente del ToTensor() de v1-v3."""
    return x_uint8.float().div_(255.0)


def entrenar_eslabon(modelo, X, Y, logits_prof, epochs, optimizador, es_cnn, semilla,
                     T=TEMPERATURA, alpha=ALPHA):
    """Entrena un eslabón. Si logits_prof es None entrena con cross-entropy pura.

    `logits_prof` son los logits del profesor YA PRE-COMPUTADOS sobre las mismas vistas,
    alineados por índice con X. Pre-computarlos (en vez de correr el profesor dentro del
    bucle) es la razón de que la cadena entera corra en minutos: el profesor está
    congelado, así que sus logits no cambian entre épocas.
    """
    torch.manual_seed(semilla)
    idx = torch.arange(len(Y))
    tensores = [idx, Y] if logits_prof is None else [idx, Y, logits_prof]
    dl = DataLoader(TensorDataset(*tensores), batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(semilla))

    for _ in range(epochs):
        modelo.train()
        for lote in dl:
            i, y = lote[0], lote[1].to(DEVICE)
            x = normalizar180(X[i]).to(DEVICE) if es_cnn else X[i].to(DEVICE)
            optimizador.zero_grad()
            out = modelo(x)
            if logits_prof is None:
                loss = F.cross_entropy(out, y)
            else:
                loss = perdida_destilacion(out, lote[2].to(DEVICE), y, T, alpha)
            loss.backward()
            optimizador.step()
    return modelo


@torch.no_grad()
def logits_de(modelo, X, es_cnn) -> torch.Tensor:
    """Logits del modelo sobre todas las vistas, por lotes (para no reventar la RAM)."""
    modelo.eval()
    salidas = []
    for i in range(0, len(X), BATCH_SIZE):
        x = normalizar180(X[i:i + BATCH_SIZE]) if es_cnn else X[i:i + BATCH_SIZE]
        salidas.append(modelo(x.to(DEVICE)).cpu())
    return torch.cat(salidas)


def probabilidades(modelo, X, es_cnn) -> np.ndarray:
    """Vector COMPLETO de probabilidades softmax (regla del proyecto desde la v2)."""
    return torch.softmax(logits_de(modelo, X, es_cnn), dim=1).numpy()


# --- 6. Los dos brazos ---
def brazo_directo(params, epochs, semilla):
    """BRAZO A — la receta de la v7 entrenada de una: cabeza Optuna + cross-entropy.

    Usa las MISMAS vistas que la cadena (limpias + aumentadas) para que la única
    diferencia entre los dos brazos sea el procedimiento de entrenamiento, no los datos.
    Por eso su número puede diferir un poco del de la v7, que entrenaba solo sobre las
    vistas limpias.
    """
    modelo = cabeza_v7(params)
    return entrenar_eslabon(modelo, XE_dev, Y_dev, None, epochs,
                            optimizador_v7(modelo, params), es_cnn=False, semilla=semilla)


def brazo_cadena(params, epochs_final, semilla, verbose=True):
    """BRAZO B — los cinco eslabones, cada uno destilando del anterior.

    Regla clave: el estudiante se RE-INICIALIZA en cada eslabón (pesos random en E1-E3,
    pesos de ImageNet en E4-E5). Si heredara los pesos del profesor esto sería
    fine-tuning con otro nombre, no destilación.
    """
    historial = []

    def registrar(nombre, modelo, es_cnn, X_ev):
        prob = probabilidades(modelo, X_ev, es_cnn)
        r = metricas.resumen_completo(Y_test.numpy(), prob, class_names)
        historial.append({"eslabon": nombre, "accuracy": r["accuracy"], "qwk": r["qwk"],
                          "ece": r["ece"], "extremos_total": r["extremos_total"]})
        if verbose:
            print(f"    {nombre}: test_acc={r['accuracy']:.4f}  QWK={r['qwk']:.4f}  "
                  f"ECE={r['ece']:.4f}  0<->2={r['extremos_total']}")

    # --- E1: CNN desde cero, SOLO vistas limpias, sin profesor (es la raíz) ---
    X1, Y1 = X180_dev[mascara_limpias], Y_dev[mascara_limpias]
    e1 = BaselineCNN(num_classes).to(DEVICE)
    e1 = entrenar_eslabon(e1, X1, Y1, None, EPOCAS["E1"],
                          torch.optim.Adam(e1.parameters(), lr=LR_CNN), True, semilla)
    registrar("E1 (v1: CNN pelada)", e1, True, X180_test)

    # --- E2: CNN re-inicializada + augmentation, destilando de E1 ---
    # Acá es donde el profesor sobreajustado SÍ enseña: E1 memorizó las vistas limpias,
    # pero nunca vio las aumentadas, así que sobre ellas produce distribuciones blandas.
    logits_e1 = logits_de(e1, X180_dev, True)
    e2 = BaselineCNN(num_classes).to(DEVICE)
    e2 = entrenar_eslabon(e2, X180_dev, Y_dev, logits_e1, EPOCAS["E2"],
                          torch.optim.Adam(e2.parameters(), lr=LR_CNN), True, semilla)
    registrar("E2 (v2: + augmentation)", e2, True, X180_test)

    # --- E3: CNN re-inicializada + más épocas (converger), destilando de E2 ---
    logits_e2 = logits_de(e2, X180_dev, True)
    e3 = BaselineCNN(num_classes).to(DEVICE)
    e3 = entrenar_eslabon(e3, X180_dev, Y_dev, logits_e2, EPOCAS["E3"],
                          torch.optim.Adam(e3.parameters(), lr=LR_CNN), True, semilla)
    registrar("E3 (v3: + converger)", e3, True, X180_test)

    # --- E4: CROSS-ARCHITECTURE. Profesor CNN 180px -> estudiante MobileNetV3 224px ---
    # ESTE es el Escenario 1 del script del profe: arquitecturas distintas, mismo espacio
    # de salida (3 clases), así que los logits se comparan directamente con KL.
    logits_e3 = logits_de(e3, X180_dev, True)
    e4 = cabeza_v4()
    e4 = entrenar_eslabon(e4, XE_dev, Y_dev, logits_e3, EPOCAS["E4"],
                          torch.optim.Adam(e4.parameters(), lr=1e-3), False, semilla)
    registrar("E4 (v4: -> MobileNetV3)", e4, False, XE_test)

    # --- E5: cabeza con los hiperparámetros de la v7, destilando de E4 ---
    logits_e4 = logits_de(e4, XE_dev, False)
    e5 = cabeza_v7(params)
    e5 = entrenar_eslabon(e5, XE_dev, Y_dev, logits_e4, epochs_final,
                          optimizador_v7(e5, params), False, semilla)
    registrar("E5 (v7: cabeza Optuna)", e5, False, XE_test)

    # Se devuelven también los logits de E4: el barrido de T/alpha los reutiliza en vez
    # de reconstruir la cadena entera (ahorra una cadena completa de cómputo).
    return e5, historial, logits_e4


# --- 7. Barrido de sensibilidad de T y alpha (sobre CV, NUNCA sobre el test) ---
def barrido_T_alpha(params, epochs_final, semilla, logits_prof):
    """¿Cuánto dependen los resultados de la temperatura y del peso del término KL?

    Se mide sobre VALIDACIÓN CRUZADA dentro de desarrollo, no sobre el test: el barrido
    es información, no un mecanismo de selección. Si se eligiera T y alpha mirando el
    test, se repetiría exactamente el pecado que la v7 vino a corregir.

    Para que sea barato, el barrido toca solo el ÚLTIMO eslabón, reutilizando los logits
    del profesor E4 que la cadena ya calculó (`logits_prof`).
    """
    n_folds = particion["meta"]["n_folds"]
    pos = {r: i for i, r in enumerate(rutas_dev)}
    # Cada imagen aporta N_VISTAS_AUG+1 vistas CONSECUTIVAS, así que la vista limpia de
    # la imagen i está en la posición i*(N_VISTAS_AUG+1).
    idx_folds_vistas = []
    for k in range(n_folds):
        rutas_k = rutas_y_etiquetas(particion, "validacion", fold=k)[0]
        idx_folds_vistas.append(
            np.array([pos[r] * (N_VISTAS_AUG + 1) for r in rutas_k])  # solo la vista LIMPIA
        )

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
                mask_t = torch.from_numpy(mask)
                cab = cabeza_v7(params)
                cab = entrenar_eslabon(cab, XE_dev[mask_t], Y_dev[mask_t],
                                       logits_prof[mask_t], epochs_final,
                                       optimizador_v7(cab, params), False, semilla, T, a)
                prob = probabilidades(cab, XE_dev[torch.from_numpy(idx_va)], False)
                accs.append(float((prob.argmax(1) == Y_dev[idx_va].numpy()).mean()))
            resultados.append({"T": T, "alpha": a, "cv_accuracy": float(np.mean(accs))})
            print(f"    T={T:<4} alpha={a:<4}  cv_accuracy={np.mean(accs):.4f}")
    return resultados


# --- 8. Main ---
if __name__ == "__main__":
    AQUI = Path(__file__).resolve().parent

    with open(RAIZ / "PyTorch/v7/mejores_hiperparametros.json", encoding="utf-8") as f:
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
    modelos_b = []
    for semilla in SEMILLAS:
        print(f"\n  --- semilla {semilla} ---")
        t0 = time.time()

        modelo_a = brazo_directo(PARAMS, EPOCHS_FINAL, semilla)
        prob_a = probabilidades(modelo_a, XE_test, False)
        ra = metricas.resumen_completo(Y_test.numpy(), prob_a, class_names)
        ra["semilla"] = semilla; ra["y_prob"] = prob_a
        res_a.append(ra)
        print(f"    BRAZO A (directo): test_acc={ra['accuracy']:.4f}  "
              f"QWK={ra['qwk']:.4f}  ECE={ra['ece']:.4f}  0<->2={ra['extremos_total']}")

        modelo_b, historial, logits_e4 = brazo_cadena(PARAMS, EPOCHS_FINAL, semilla)
        prob_b = probabilidades(modelo_b, XE_test, False)
        rb = metricas.resumen_completo(Y_test.numpy(), prob_b, class_names)
        rb["semilla"] = semilla; rb["y_prob"] = prob_b
        res_b.append(rb)
        modelos_b.append(modelo_b)
        historiales.append(historial)
        print(f"    ({time.time() - t0:.0f}s)")

    # --- Agregado y comparación ---
    agr_a = metricas.agregar_semillas(res_a)
    agr_b = metricas.agregar_semillas(res_b)

    print()
    metricas.imprimir_agregado(agr_a, "BRAZO A — DIRECTO (PyTorch v8)")
    print()
    metricas.imprimir_agregado(agr_b, "BRAZO B — CADENA DE DESTILACIÓN (PyTorch v8)")

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
    # Reutiliza el profesor E4 de la ÚLTIMA cadena entrenada (semilla SEMILLAS[-1]).
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
    metricas.plot_calibracion(Y_test.numpy(), res_a[0]["y_prob"],
                              f"Calibración — BRAZO A directo (semilla {SEMILLAS[0]})",
                              AQUI / "Figure_calibracion_directo_v8.png")
    metricas.plot_calibracion(Y_test.numpy(), res_b[0]["y_prob"],
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
             color="#B45309", label="cadena")
    ax1.axhline(agr_a["accuracy"]["media"], ls="--", lw=2, color="#1F3864",
                label=f"brazo A directo ({agr_a['accuracy']['media']:.3f})")
    ax1.set_xticks(list(xs)); ax1.set_xticklabels(etiquetas_cortas)
    ax1.set_ylabel("accuracy en test"); ax1.set_title("Accuracy eslabón a eslabón")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(xs, [prog[n]["ece"] for n in nombres_esl], "o-", lw=2, ms=8,
             color="#B45309", label="cadena")
    ax2.axhline(agr_a["ece"]["media"], ls="--", lw=2, color="#1F3864",
                label=f"brazo A directo ({agr_a['ece']['media']:.3f})")
    ax2.set_xticks(list(xs)); ax2.set_xticklabels(etiquetas_cortas)
    ax2.set_ylabel("ECE (más bajo = mejor)"); ax2.set_title("Calibración eslabón a eslabón")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle("Cadena de destilación — PyTorch v8 (media de 5 semillas)",
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
                           "Dispersión entre semillas — BRAZO B cadena (PyTorch v8)",
                           AQUI / "Figure_semillas_cadena_v8.png")

    # --- Guardar resultados ---
    with open(AQUI / "resultados_v8.json", "w", encoding="utf-8") as f:
        json.dump({
            "framework": "pytorch",
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

    MODEL_PATH = AQUI / "modelopt_v8_cadena.pt"
    torch.save({
        "cabeza_state_dict": modelos_b[0].state_dict(),
        "best_params": PARAMS,
        "semilla": SEMILLAS[0],
        "class_names": class_names,
        "img_size": SIZE_MNET,
        "emb_dim": EMB_DIM,
        "arch": "mobilenet_v3_small",
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
        "destilacion": {"temperatura": TEMPERATURA, "alpha": ALPHA},
    }, MODEL_PATH)
    print(f"  guardado {MODEL_PATH.name}  (brazo B, semilla {SEMILLAS[0]})")
