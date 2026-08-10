"""
Métricas compartidas por TensorFlow y PyTorch — todas calculadas A MANO con numpy.

Por qué un módulo compartido (y por qué numpy puro):
  Hasta la v6 cada script se copiaba su propio reporte_por_clase() y su propio bloque
  de ROC. Eso produjo dos problemas reales en el repo:
    - Las curvas ROC quedaron SOLO del lado PyTorch (6 figuras) y nunca se hicieron
      del lado TensorFlow: la paridad "espejo" del proyecto estaba rota.
    - Cualquier métrica nueva había que escribirla dos veces, con riesgo de que las
      dos implementaciones no coincidan y las comparaciones TF vs PyTorch mientan.
  Este módulo es numpy puro (sin sklearn, sin torch, sin tf) justamente para poder
  importarse desde LOS DOS venvs. A partir de la v7, ambos frameworks miden con
  exactamente el mismo código: si los números difieren, es por el modelo, no por la
  regla de medición.

Convención de matriz (la misma de todo el proyecto):
  cm[i, j] = casos cuya clase REAL es i y fueron PREDICHOS como j.
  Filas = real, columnas = predicho. Clases en orden 0_sin_ia, 1_rastro_ia, 2_saturada_ia.

Qué métricas hay y por qué cada una:
  - accuracy / precision / recall / f1  : las de siempre.
  - errores_extremos (0<->2)            : el error CARO del proyecto. Una diapo saturada
                                          clasificada como "sin rastro" deja pasar justo
                                          lo que se quiere detectar.
  - QWK (kappa cuadrático ponderado)    : las 3 clases son ORDINALES y hasta ahora se
                                          trataban como nominales (accuracy castiga igual
                                          confundir 0<->1 que 0<->2). QWK pondera el error
                                          por la DISTANCIA al cuadrado, así que confundir
                                          extremos pesa 4 veces más que confundir vecinos.
                                          Es el criterio de negocio convertido en métrica.
  - MAE ordinal                         : el mismo espíritu, en la escala de la clase.
  - ROC-AUC One-vs-Rest                 : mide la calidad del SCORE continuo, sin depender
                                          del umbral 0.5 implícito en el argmax.
  - ECE (Expected Calibration Error)    : ¿cuando el modelo dice 90%, acierta el 90%?
                                          Importa para la v8 (destilación): destilar suele
                                          mejorar la CALIBRACIÓN más que la accuracy, así
                                          que sin ECE ese efecto sería invisible.
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORES = ["#1F3864", "#B45309", "#2E7D32", "#7B1FA2"]


# --- 1. Matriz de confusión ---
def matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_clases: int) -> np.ndarray:
    """cm[i, j] = real i, predicho j. A mano, sin tf.math.confusion_matrix ni sklearn."""
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        cm[t, p] += 1
    return cm


# --- 2. Reporte por clase ---
def reporte_por_clase(cm: np.ndarray, nombres: list[str], imprimir: bool = True) -> dict:
    """Precision / recall / f1 / soporte por clase + macro avg + accuracy.

    Idéntico al que traían v4/v5/v6 inline, movido acá para que lo usen los dos
    frameworks. Devuelve además un dict para poder agregarlo entre semillas.
    """
    total = cm.sum()
    precs, recs, f1s = [], [], []
    filas = []
    for i in range(len(nombres)):
        soporte = cm[i, :].sum()
        predichos = cm[:, i].sum()
        recall = cm[i, i] / soporte if soporte else 0.0
        precision = cm[i, i] / predichos if predichos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
        filas.append({"clase": nombres[i], "precision": precision, "recall": recall,
                      "f1": f1, "soporte": int(soporte)})

    acc = np.trace(cm) / total if total else 0.0
    resumen = {
        "por_clase": filas,
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
        "accuracy": float(acc),
    }

    if imprimir:
        print(f"{'clase':<16}{'precision':>10}{'recall':>9}{'f1':>7}{'soporte':>9}")
        print("-" * 51)
        for f in filas:
            print(f"{f['clase']:<16}{f['precision']:>10.3f}{f['recall']:>9.3f}"
                  f"{f['f1']:>7.3f}{f['soporte']:>9}")
        print("-" * 51)
        print(f"{'macro avg':<16}{resumen['macro_precision']:>10.3f}"
              f"{resumen['macro_recall']:>9.3f}{resumen['macro_f1']:>7.3f}{int(total):>9}")
        print(f"accuracy: {acc:.3f}  ({int(np.trace(cm))}/{int(total)})")
    return resumen


# --- 3. El error caro del proyecto: la esquina 0<->2 ---
def errores_extremos(cm: np.ndarray) -> dict:
    """Cuenta la confusión entre los EXTREMOS del eje ordinal.

    En un eje 0 -> 1 -> 2, confundir vecinos es barato (la frontera 1<->2 es
    genuinamente ambigua) pero confundir extremos es cualitativamente distinto:
      cm[2, 0] = una diapo SATURADA clasificada como SIN RASTRO -> falso negativo
                 del producto: deja pasar exactamente lo que se quería detectar.
      cm[0, 2] = una diapo HUMANA acusada de saturada -> falso positivo.
    Este contador es el instrumento que el proyecto usa desde la v2 para decidir si
    un modelo sirve, por encima de la accuracy.
    """
    n = cm.shape[0]
    falsos_negativos = int(cm[n - 1, 0])   # 2 -> 0
    falsos_positivos = int(cm[0, n - 1])   # 0 -> 2
    return {
        "extremos_total": falsos_negativos + falsos_positivos,
        "saturada_como_sin_ia": falsos_negativos,
        "sin_ia_como_saturada": falsos_positivos,
    }


# --- 4. Métricas ORDINALES (nuevas en la v7) ---
def qwk(cm: np.ndarray) -> float:
    """Kappa cuadrático ponderado (Quadratic Weighted Kappa).

    Accuracy trata las 3 clases como nominales: confundir 0<->2 le cuesta lo mismo
    que confundir 0<->1. Pero nuestro eje es ORDINAL (densidad de artefactos), así
    que ese empate es conceptualmente falso — y el propio avances_proyecto.md §4 lo
    declara como un tradeoff asumido. QWK levanta ese tradeoff:

        w[i,j] = (i - j)^2 / (k - 1)^2        peso del error según la DISTANCIA
        kappa  = 1 - sum(w * O) / sum(w * E)  O = observado, E = esperado por azar

    Con k=3: confundir vecinos pesa 0.25, confundir extremos pesa 1.00 -> el error
    caro cuesta 4x. Además E descuenta el acuerdo que se obtendría por azar dadas
    las marginales, así que un modelo que siempre predice la clase mayoritaria da
    ~0 aunque su accuracy parezca decente.

    Escala: 1.0 = acuerdo perfecto · 0.0 = como el azar · <0 = peor que el azar.
    """
    n = cm.shape[0]
    total = cm.sum()
    if total == 0 or n < 2:
        return 0.0
    i, j = np.indices((n, n))
    w = ((i - j) ** 2) / ((n - 1) ** 2)
    esperado = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / total
    den = (w * esperado).sum()
    return float(1.0 - (w * cm).sum() / den) if den else 0.0


def mae_ordinal(cm: np.ndarray) -> float:
    """Error absoluto medio en la escala de clases: media de |real - predicho|.

    Complemento directo de QWK, en unidades interpretables: 0.15 significa que, en
    promedio, el modelo se corre 0.15 niveles de saturación. Un modelo con accuracy
    0.90 y MAE 0.10 falla solo entre vecinos; con accuracy 0.90 y MAE 0.20 sus
    errores incluyen saltos de dos niveles.
    """
    n = cm.shape[0]
    total = cm.sum()
    if total == 0:
        return 0.0
    i, j = np.indices((n, n))
    return float((np.abs(i - j) * cm).sum() / total)


# --- 5. ROC-AUC One-vs-Rest ---
def roc_binaria(y_bin: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Curva ROC de un problema binario, a mano.

    Ordena por score descendente y acumula verdaderos/falsos positivos umbral a
    umbral; el AUC sale por regla trapezoidal. Se escribe a mano (en vez de usar
    sklearn) por dos razones: es la convención del repo desde la v4, y evita meter
    sklearn como dependencia del venv de TensorFlow.

    np.trapz se renombró a np.trapezoid en numpy 2.x, así que el área se suma
    explícitamente para que el módulo corra en numpy 1.x y 2.x sin tocar nada.
    """
    y_bin = np.asarray(y_bin)
    scores = np.asarray(scores)
    orden = np.argsort(-scores, kind="mergesort")
    y = y_bin[orden].astype(float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    tpr = np.concatenate([[0.0], np.cumsum(y) / n_pos]) if n_pos else np.zeros(len(y) + 1)
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / n_neg]) if n_neg else np.zeros(len(y) + 1)
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def curvas_roc_ovr(y_true: np.ndarray, y_prob: np.ndarray, n_clases: int):
    """ROC One-vs-Rest: {clase -> (fpr, tpr, auc)}, curva micro-promedio y AUC macro.

    ROC es binaria por naturaleza. Con 3 clases se arma, para cada clase i, el
    problema "i contra el resto" barriendo el umbral sobre su probabilidad softmax.
      - macro: promedio simple de los 3 AUC (cada clase pesa igual).
      - micro: se aplanan el one-hot real y las probabilidades y se corre UNA sola
        ROC sobre todas las decisiones juntas (pondera por soporte).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    curvas = {i: roc_binaria((y_true == i).astype(int), y_prob[:, i]) for i in range(n_clases)}
    macro = float(np.mean([curvas[i][2] for i in range(n_clases)]))
    onehot = np.eye(n_clases)[y_true].ravel()
    micro = roc_binaria(onehot.astype(int), y_prob.ravel())
    return curvas, micro, macro


# --- 6. Calibración ---
def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: ¿la confianza del modelo dice la verdad?

    Se agrupan las predicciones en n_bins tramos según su confianza (la probabilidad
    máxima del softmax) y en cada tramo se compara la confianza MEDIA contra la
    accuracy REAL. ECE es el promedio de esa brecha, ponderado por cuántos casos
    cayeron en cada tramo:

        ECE = sum_b  (|b| / N) * | accuracy(b) - confianza(b) |

    0.0 = perfectamente calibrado. Un modelo puede tener buena accuracy y ECE malo
    (acierta, pero siempre dice 99% aunque se equivoque) — es el patrón típico de un
    modelo sobreajustado, y exactamente lo que la destilación con temperatura suele
    corregir. Por eso esta métrica entra ahora: es el instrumento con el que se va a
    medir la v8.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acierto = (pred == y_true).astype(float)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    total = 0.0
    for b in range(n_bins):
        lo, hi = bordes[b], bordes[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            total += (m.sum() / n) * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


# --- 7. Resumen completo ---
def resumen_completo(y_true: np.ndarray, y_prob: np.ndarray, nombres: list[str]) -> dict:
    """Todas las métricas del proyecto en un dict, listo para promediar entre semillas."""
    n_clases = len(nombres)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    cm = matriz_confusion(y_true, y_pred, n_clases)
    rep = reporte_por_clase(cm, nombres, imprimir=False)
    _, micro, macro = curvas_roc_ovr(y_true, y_prob, n_clases)
    return {
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro_f1"],
        "recall_saturada": rep["por_clase"][n_clases - 1]["recall"],
        **errores_extremos(cm),
        "qwk": qwk(cm),
        "mae_ordinal": mae_ordinal(cm),
        "auc_macro": macro,
        "auc_micro": micro[2],
        "ece": ece(y_true, y_prob),
        "cm": cm.tolist(),
    }


# --- 8. Agregación entre semillas (media ± desvío) ---
# Claves que se promedian; 'cm' se suma aparte porque es una matriz, no un escalar.
CLAVES_ESCALARES = [
    "accuracy", "macro_f1", "recall_saturada", "extremos_total",
    "saturada_como_sin_ia", "sin_ia_como_saturada", "qwk", "mae_ordinal",
    "auc_macro", "auc_micro", "ece",
]

# Cómo se lee cada métrica: True = más alto es mejor.
MAS_ES_MEJOR = {
    "accuracy": True, "macro_f1": True, "recall_saturada": True,
    "extremos_total": False, "saturada_como_sin_ia": False, "sin_ia_como_saturada": False,
    "qwk": True, "mae_ordinal": False, "auc_macro": True, "auc_micro": True, "ece": False,
}

ETIQUETAS = {
    "accuracy": "accuracy",
    "macro_f1": "F1 macro",
    "recall_saturada": "recall clase 2",
    "extremos_total": "errores 0<->2",
    "saturada_como_sin_ia": "  de los cuales 2->0",
    "sin_ia_como_saturada": "  de los cuales 0->2",
    "qwk": "QWK (ordinal)",
    "mae_ordinal": "MAE ordinal",
    "auc_macro": "AUC macro (OvR)",
    "auc_micro": "AUC micro (OvR)",
    "ece": "ECE (calibración)",
}


def agregar_semillas(resumenes: list[dict]) -> dict:
    """Media, desvío, mínimo y máximo de cada métrica a lo largo de N semillas.

    Esta función es la respuesta al límite más grande del proyecto: con 60 imágenes
    de test, UNA imagen vale 1.67 puntos de accuracy. Reportar '93.3%' de una sola
    corrida sugiere una precisión que el dataset no soporta. Con N semillas se
    reporta 'media ± std' y se puede decir si una diferencia entre dos modelos está
    dentro del ruido o no.

    Devuelve además 'cm_sumada': la matriz de confusión acumulada sobre todas las
    semillas. Sumarlas (en vez de promediarlas) da una matriz de enteros con N veces
    más casos, mucho más estable para leer el patrón de errores.
    """
    agregado = {}
    for clave in CLAVES_ESCALARES:
        vals = np.array([r[clave] for r in resumenes], dtype=float)
        agregado[clave] = {
            "media": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
        }
    cms = np.array([r["cm"] for r in resumenes], dtype=int)
    agregado["cm_sumada"] = cms.sum(axis=0).tolist()
    agregado["n_semillas"] = len(resumenes)
    return agregado


def imprimir_agregado(agregado: dict, titulo: str = "RESULTADOS SOBRE TEST") -> None:
    n = agregado["n_semillas"]
    print("=" * 66)
    print(f"  {titulo}  ·  {n} semillas  ·  media ± desvío")
    print("=" * 66)
    print(f"  {'métrica':<24}{'media':>10}{'±std':>9}{'mín':>9}{'máx':>9}")
    print("  " + "-" * 61)
    for clave in CLAVES_ESCALARES:
        a = agregado[clave]
        flecha = "↑" if MAS_ES_MEJOR[clave] else "↓"
        print(f"  {ETIQUETAS[clave]:<22}{flecha:<2}{a['media']:>10.4f}{a['std']:>9.4f}"
              f"{a['min']:>9.4f}{a['max']:>9.4f}")
    print("  " + "-" * 61)
    print("  ↑ = más alto es mejor   ·   ↓ = más bajo es mejor")


# --- 9. Figuras ---
def plot_matriz(cm: np.ndarray, nombres: list[str], titulo: str, salida) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(nombres))); ax.set_yticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=45, ha="right"); ax.set_yticklabels(nombres)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(titulo, fontweight="bold", color="#1F3864")
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {getattr(salida, 'name', salida)}")


def plot_roc(curvas, micro, macro, nombres: list[str], titulo: str, salida) -> None:
    fpr_m, tpr_m, micro_auc = micro
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (fpr, tpr, auc) in curvas.items():
        ax.plot(fpr, tpr, lw=2, color=COLORES[i % len(COLORES)],
                label=f"{nombres[i]} (AUC={auc:.3f})")
    ax.plot(fpr_m, tpr_m, lw=2, ls=":", color="gray",
            label=f"micro-promedio (AUC={micro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="azar (AUC=0.500)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title(f"{titulo}\nAUC macro = {macro:.3f}", fontweight="bold", color="#1F3864")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {getattr(salida, 'name', salida)}")


def plot_calibracion(y_true, y_prob, titulo: str, salida, n_bins: int = 10) -> None:
    """Diagrama de fiabilidad: confianza declarada (x) vs accuracy real (y).

    La diagonal es la calibración perfecta. Barras POR DEBAJO de la diagonal =
    el modelo es demasiado confiado (dice 95% y acierta 80%): el patrón clásico de
    una red sobreajustada, y lo que la destilación con temperatura tiende a corregir.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1)
    acierto = (y_prob.argmax(axis=1) == y_true).astype(float)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    centros, accs, confs, pesos = [], [], [], []
    for b in range(n_bins):
        lo, hi = bordes[b], bordes[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            centros.append((lo + hi) / 2); accs.append(acierto[m].mean())
            confs.append(conf[m].mean()); pesos.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="calibración perfecta")
    ax.bar(centros, accs, width=1.0 / n_bins * 0.9, color="#1F3864", alpha=0.8,
           edgecolor="white", label="accuracy real")
    ax.plot(confs, accs, "o-", color="#B45309", lw=2, ms=6, label="confianza media del tramo")
    for x, y, p in zip(centros, accs, pesos):
        ax.text(x, y + 0.02, f"n={p}", ha="center", fontsize=8, color="#444")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Confianza declarada (prob. máx. del softmax)")
    ax.set_ylabel("Accuracy real del tramo")
    ax.set_title(f"{titulo}\nECE = {ece(y_true, y_prob, n_bins):.4f}",
                 fontweight="bold", color="#1F3864")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {getattr(salida, 'name', salida)}")


def plot_semillas(resumenes: list[dict], claves: list[str], titulo: str, salida) -> None:
    """Dispersión de cada métrica a lo largo de las semillas.

    Es la figura que hace VISIBLE el ruido del dataset: si los puntos de dos modelos
    se solapan, la diferencia entre sus medias no significa nada.
    """
    fig, axes = plt.subplots(1, len(claves), figsize=(3.4 * len(claves), 4.2))
    if len(claves) == 1:
        axes = [axes]
    for ax, clave in zip(axes, claves):
        vals = [r[clave] for r in resumenes]
        ax.scatter(range(len(vals)), vals, s=70, color="#1F3864", zorder=3)
        media = float(np.mean(vals)); std = float(np.std(vals))
        ax.axhline(media, color="#B45309", ls="--", lw=2, label=f"media {media:.3f}")
        ax.axhspan(media - std, media + std, color="#B45309", alpha=0.15,
                   label=f"±1 std ({std:.3f})")
        ax.set_title(ETIQUETAS.get(clave, clave), fontweight="bold")
        ax.set_xlabel("semilla"); ax.set_xticks(range(len(vals)))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(titulo, fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  guardado {getattr(salida, 'name', salida)}")
