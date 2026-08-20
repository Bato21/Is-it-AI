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

EL EJE ORDINAL DEJA DE SER TODO EL PROBLEMA (v10)
-------------------------------------------------
La v10 agregó una CUARTA clase, `3_no_diapositiva`, que NO pertenece al eje ordinal: no es
"más saturada que 2", es otra cosa — la compuerta de rechazo del producto. Eso obliga a
separar las métricas en dos grupos, porque mezclarlas produciría números sin sentido:

  MÉTRICAS ORDINALES (QWK, MAE ordinal, errores 0<->2). Solo tienen sentido sobre el eje
  0-1-2. Si se calcularan sobre las 4 clases, QWK trataría "no es una diapositiva" como si
  estuviera un escalón más allá de "saturada" y penalizaría confundir 0 con 3 cuatro veces
  más que confundir 0 con 1 — cuando conceptualmente ni siquiera están en la misma escala.
  Por eso, con `n_ordinales=3`, estas métricas se calculan sobre el SUB-BLOQUE cm[:3, :3]:
  se leen como "de las imágenes que el modelo aceptó como diapositivas, qué tan bien las
  ordenó". Es una métrica CONDICIONAL, y así hay que reportarla.

  MÉTRICAS DE COMPUERTA (recall/precisión de rechazo, fuga, rechazo indebido). Miden lo otro:
  si el modelo deja pasar como diapositiva algo que no lo es, y a qué costo. Son las que
  responden la pregunta que motivó la v10.

  Los dos errores de la compuerta NO son simétricos y por eso se cuentan aparte:
    fuga_no_diapositiva  una foto que no es diapositiva analizada igual -> el modelo opina
                         sobre algo que no entiende. Es EL fallo que la v10 vino a cerrar.
    rechazo_indebido     una diapositiva real descartada como "no es una diapositiva" ->
                         el usuario no obtiene su respuesta. Molesto, pero honesto.

  Con `n_ordinales = n_clases` (o sin pasarlo) todo se comporta EXACTAMENTE como en la v9:
  los scripts v1-v9 no cambian de resultado.

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
def errores_extremos(cm: np.ndarray, n_ordinales: int | None = None) -> dict:
    """Cuenta la confusión entre los EXTREMOS del eje ordinal.

    En un eje 0 -> 1 -> 2, confundir vecinos es barato (la frontera 1<->2 es
    genuinamente ambigua) pero confundir extremos es cualitativamente distinto:
      cm[2, 0] = una diapo SATURADA clasificada como SIN RASTRO -> falso negativo
                 del producto: deja pasar exactamente lo que se quería detectar.
      cm[0, 2] = una diapo HUMANA acusada de saturada -> falso positivo.
    Este contador es el instrumento que el proyecto usa desde la v2 para decidir si
    un modelo sirve, por encima de la accuracy.

    `n_ordinales` acota el eje cuando hay clases fuera de él (la de rechazo de la v10).
    Sin él se usa la matriz entera, que es el comportamiento de v1-v9.
    """
    k = cm.shape[0] if n_ordinales is None else n_ordinales
    falsos_negativos = int(cm[k - 1, 0])   # 2 -> 0
    falsos_positivos = int(cm[0, k - 1])   # 0 -> 2
    return {
        "extremos_total": falsos_negativos + falsos_positivos,
        "saturada_como_sin_ia": falsos_negativos,
        "sin_ia_como_saturada": falsos_positivos,
    }


# --- 3b. La compuerta de rechazo (nueva en la v10) ---
def metricas_rechazo(cm: np.ndarray, n_ordinales: int | None = None) -> dict:
    """Calidad de la compuerta "esto no es una diapositiva".

    Con `n_ordinales = k` y k+1 clases, la fila/columna k es la de rechazo:

      recall_rechazo       cm[k,k] / fila k    de todo lo que NO era diapositiva,
                                               cuánto atajó el modelo.
      precision_rechazo    cm[k,k] / col. k    de todo lo que rechazó, cuánto
                                               efectivamente no era diapositiva.
      fuga_no_diapositiva  fila k - cm[k,k]    casos que NO eran diapositiva y que el
                                               modelo analizó igual, inventando un nivel
                                               de IA. Es el fallo que la v10 vino a cerrar.
      rechazo_indebido     col. k - cm[k,k]    diapositivas reales descartadas. El costo.

    Se reportan los CONTEOS además de las tasas porque el test es chico (~270 imágenes): una
    tasa de 0.017 esconde que se trata de una sola imagen, y decidir sobre tasas con esos
    tamaños de muestra es exactamente lo que el proyecto viene evitando desde la v7.

    Sin clase de rechazo devuelve ceros, así los scripts de v1-v9 pueden llamarla sin
    cambiar de resultado y las claves existen siempre (agregar entre semillas no rompe).
    """
    n = cm.shape[0]
    k = n if n_ordinales is None else n_ordinales
    if k >= n:
        return {"recall_rechazo": 0.0, "precision_rechazo": 0.0,
                "fuga_no_diapositiva": 0, "rechazo_indebido": 0}

    aciertos = int(cm[k, k])
    reales = int(cm[k, :].sum())        # imágenes que realmente no eran diapositivas
    predichos = int(cm[:, k].sum())     # imágenes que el modelo mandó a rechazo
    return {
        "recall_rechazo": aciertos / reales if reales else 0.0,
        "precision_rechazo": aciertos / predichos if predichos else 0.0,
        "fuga_no_diapositiva": reales - aciertos,
        "rechazo_indebido": predichos - aciertos,
    }


# --- 4. Métricas ORDINALES (nuevas en la v7) ---
def qwk(cm: np.ndarray, n_ordinales: int | None = None) -> float:
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

    `n_ordinales` acota el cálculo al sub-bloque del eje: con la clase de rechazo de la v10
    adentro, QWK trataría "no es una diapositiva" como un cuarto escalón de saturación, que
    es falso. Acotado, el número se lee como CONDICIONAL: qué tan bien ordena el modelo las
    imágenes que aceptó como diapositivas. Los casos que salieron por la compuerta se miden
    con metricas_rechazo(), no acá.
    """
    n = cm.shape[0] if n_ordinales is None else min(n_ordinales, cm.shape[0])
    cm = np.asarray(cm)[:n, :n]
    total = cm.sum()
    if total == 0 or n < 2:
        return 0.0
    i, j = np.indices((n, n))
    w = ((i - j) ** 2) / ((n - 1) ** 2)
    esperado = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / total
    den = (w * esperado).sum()
    return float(1.0 - (w * cm).sum() / den) if den else 0.0


def mae_ordinal(cm: np.ndarray, n_ordinales: int | None = None) -> float:
    """Error absoluto medio en la escala de clases: media de |real - predicho|.

    Complemento directo de QWK, en unidades interpretables: 0.15 significa que, en
    promedio, el modelo se corre 0.15 niveles de saturación. Un modelo con accuracy
    0.90 y MAE 0.10 falla solo entre vecinos; con accuracy 0.90 y MAE 0.20 sus
    errores incluyen saltos de dos niveles.

    Se acota al eje ordinal por la misma razón que QWK (ver arriba).
    """
    n = cm.shape[0] if n_ordinales is None else min(n_ordinales, cm.shape[0])
    cm = np.asarray(cm)[:n, :n]
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
def resumen_completo(y_true: np.ndarray, y_prob: np.ndarray, nombres: list[str],
                     n_ordinales: int | None = None) -> dict:
    """Todas las métricas del proyecto en un dict, listo para promediar entre semillas.

    `n_ordinales` = cuántas de las clases forman el eje ordinal. Con 3 clases (v1-v9) son
    las tres y el parámetro sobra. Con las 4 de la v10 hay que pasar 3, y entonces las
    métricas ordinales se acotan al sub-bloque y se agregan las de la compuerta de rechazo.
    """
    n_clases = len(nombres)
    k = n_clases if n_ordinales is None else n_ordinales
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    cm = matriz_confusion(y_true, y_pred, n_clases)
    rep = reporte_por_clase(cm, nombres, imprimir=False)
    _, micro, macro = curvas_roc_ovr(y_true, y_prob, n_clases)
    return {
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro_f1"],
        # Recall de 2_saturada_ia, SIEMPRE el índice k-1 del eje ordinal y no "la última
        # clase": con la clase de rechazo al final, "la última" pasó a ser otra cosa. Se
        # calcula sobre la fila completa, así que una diapositiva saturada que el modelo
        # mandó a rechazo cuenta como fallo — que es lo correcto: el usuario no obtuvo su
        # alerta, y el motivo de por qué no la obtuvo es irrelevante para él.
        "recall_saturada": rep["por_clase"][k - 1]["recall"],
        **errores_extremos(cm, k),
        **metricas_rechazo(cm, k),
        "qwk": qwk(cm, k),
        "mae_ordinal": mae_ordinal(cm, k),
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

# Métricas de la compuerta de rechazo (v10). Van en una lista APARTE y no dentro de
# CLAVES_ESCALARES a propósito: así los scripts de v1-v9 siguen imprimiendo y guardando
# exactamente las mismas 11 métricas de siempre, y sus resultados no dejan de ser
# comparables por un cambio en este módulo. Los scripts de la v10 usan la suma de las dos.
CLAVES_RECHAZO = [
    "recall_rechazo", "precision_rechazo", "fuga_no_diapositiva", "rechazo_indebido",
]

CLAVES_V10 = CLAVES_ESCALARES + CLAVES_RECHAZO

# Cómo se lee cada métrica: True = más alto es mejor.
MAS_ES_MEJOR = {
    "accuracy": True, "macro_f1": True, "recall_saturada": True,
    "extremos_total": False, "saturada_como_sin_ia": False, "sin_ia_como_saturada": False,
    "qwk": True, "mae_ordinal": False, "auc_macro": True, "auc_micro": True, "ece": False,
    "recall_rechazo": True, "precision_rechazo": True,
    "fuga_no_diapositiva": False, "rechazo_indebido": False,
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
    "recall_rechazo": "recall de rechazo",
    "precision_rechazo": "precisión de rechazo",
    "fuga_no_diapositiva": "fugas (no-diapo analizada)",
    "rechazo_indebido": "rechazos indebidos",
}


def agregar_semillas(resumenes: list[dict], claves: list[str] | None = None) -> dict:
    """Media, desvío, mínimo y máximo de cada métrica a lo largo de N semillas.

    Esta función es la respuesta al límite más grande del proyecto: con 60 imágenes
    de test, UNA imagen vale 1.67 puntos de accuracy. Reportar '93.3%' de una sola
    corrida sugiere una precisión que el dataset no soporta. Con N semillas se
    reporta 'media ± std' y se puede decir si una diferencia entre dos modelos está
    dentro del ruido o no.

    Devuelve además 'cm_sumada': la matriz de confusión acumulada sobre todas las
    semillas. Sumarlas (en vez de promediarlas) da una matriz de enteros con N veces
    más casos, mucho más estable para leer el patrón de errores.

    `claves` permite pedir otro juego de métricas (la v10 pasa CLAVES_V10, que suma las de
    la compuerta de rechazo). Por defecto son las 11 de siempre.
    """
    agregado = {}
    for clave in claves or CLAVES_ESCALARES:
        vals = np.array([r[clave] for r in resumenes], dtype=float)
        agregado[clave] = {
            "media": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
        }
    cms = np.array([r["cm"] for r in resumenes], dtype=int)
    agregado["cm_sumada"] = cms.sum(axis=0).tolist()
    agregado["n_semillas"] = len(resumenes)
    return agregado


def imprimir_agregado(agregado: dict, titulo: str = "RESULTADOS SOBRE TEST",
                      claves: list[str] | None = None) -> None:
    n = agregado["n_semillas"]
    print("=" * 66)
    print(f"  {titulo}  ·  {n} semillas  ·  media ± desvío")
    print("=" * 66)
    print(f"  {'métrica':<24}{'media':>10}{'±std':>9}{'mín':>9}{'máx':>9}")
    print("  " + "-" * 61)
    for clave in claves or CLAVES_ESCALARES:
        a = agregado.get(clave)
        if a is None:
            continue
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
