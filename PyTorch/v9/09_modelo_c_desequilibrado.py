"""
MODELO C — EfficientNet-B0 entrenado bajo DESEQUILIBRIO DE CLASES  ·  PyTorch  ·  v9

POR QUÉ EXISTE ESTE TERCER MODELO (las dos razones, que son distintas):

  RAZÓN 1 — LA CONSIGNA. El enunciado de la segunda entrega pide "un flujo que considere
  tres modelos como mínimo" y agrega: "también consideren uno de los modelos que sea
  desequilibrado". Los modelos A (TensorFlow) y B (PyTorch) de la v9 son MobileNetV3-Small
  entrenados sobre un dataset casi balanceado (347/348/374). Este es el tercero, y es el
  que corre el escenario de desequilibrio.

  RAZÓN 2 — EL PRODUCTO. La app tiene un selector de modelos, y un selector solo tiene
  sentido si los modelos son DISTINTOS entre sí de una forma que el usuario pueda notar.
  Tres MobileNetV3 iguales entrenados igual serían tres botones que hacen lo mismo. Este
  modelo cambia las dos cosas que importan:

      arquitectura : EfficientNet-B0 (5.3 M parámetros, escalado compuesto) en vez de
                     MobileNetV3-Small (2.5 M, búsqueda de arquitectura + squeeze-excite).
                     Familias de diseño distintas -> errores distintos -> comparar dos
                     predicciones desacordes es información real, no ruido.
      entrenamiento: dataset deliberadamente desequilibrado + Focal Loss + muestreo
                     ponderado, en vez de datos balanceados + cross-entropy.

POR QUÉ EFFICIENTNET-B0 Y NO OTRA:
  - Es la arquitectura que la asignatura contrapone a MobileNet: escalado compuesto
    (profundidad/ancho/resolución a la vez) contra búsqueda de arquitectura. Poder explicar
    esa diferencia es parte de lo que se evalúa.
  - Sigue siendo desplegable: 5.3 M parámetros son ~21 MB en ONNX. Pesa 3x más que
    MobileNetV3-Small y corre más lento en el navegador — y ESO es justamente lo que hace
    que el selector tenga sentido para el usuario: elegir entre rápido y pesado.
  - VGG16 o ResNet50 quedaban descartadas por lo que el propio cierre de la asignatura
    señala: "VGG16 con 21 ejemplos predice al azar". Con 1000 imágenes, 138 M parámetros
    son un modelo sobredimensionado para el dataset, y además impensables en un móvil.

EL EXPERIMENTO DE DESEQUILIBRIO (esto es lo que se analiza, no el modelo en sí):

  Se construye un subconjunto desequilibrado 10 : 3 : 1 quedándose con el 100% de
  0_sin_ia, el 30% de 1_rastro_ia y el 10% de 2_saturada_ia. La clase que se vuelve rara es
  a propósito 2_saturada_ia: es la que el producto necesita detectar. Es el escenario
  comercial realista — las presentaciones obviamente generadas por IA son minoría en el
  mundo real — y es el que produce el fracaso más engañoso.

  Se entrenan TRES variantes sobre EXACTAMENTE los mismos datos desequilibrados:

    C0  cross-entropy plana              (sin corrección: la línea base que falla)
    C1  cross-entropy + class weights    (corrección clásica: pesar la pérdida por 1/frecuencia)
    C2  Focal Loss + WeightedRandomSampler (Lin et al. 2017: γ=2 baja el peso de los
                                            ejemplos fáciles, α compensa la frecuencia, y
                                            el sampler además rebalancea el batch)

  Y se evalúan las tres sobre el MISMO test de fotos balanceado de la v9 (65/68/74). Es la
  parte importante del diseño: el test NO se desequilibra. Si se desequilibrara también el
  test, un modelo que predice siempre la clase mayoritaria sacaría 77% de accuracy y el
  experimento no mostraría nada.

LO QUE SE ESPERA VER (declarado antes de correr):
  C0 debería tener una accuracy que parece decente y un recall de la clase 2 cercano a cero:
  el modelo no aprendió a detectar diapositivas saturadas, aprendió que casi nunca las hay.
  Es exactamente el diagnóstico que describe el cierre de la asignatura ("cuando accuracy es
  alta pero Recall es cercano a cero, el modelo no aprendió el problema: memorizó la clase
  mayoritaria"). Por eso el reporte de este script encabeza con MACRO-F1 y RECALL de la
  clase 2, y deja la accuracy abajo: con desequilibrio, la accuracy es la métrica que miente.

  La variante que gane por macro-F1 es la que se exporta a ONNX y se sube a la app como
  "Modelo C". La elección queda registrada en resultados_modelo_c_v9.json.
"""

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402

# La arquitectura vive en modelos_v9.py, compartida con el modelo B y con el exportador
# a ONNX. Ver el encabezado de ese módulo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelos_v9 import ModeloC, bn_a_eval, descongelar_desde  # noqa: E402
from imagenes import cargar_para_particion, generar_vistas  # noqa: E402
from particion_v9 import cargar_particion, conteos, dominios_de, rutas_y_etiquetas  # noqa: E402

# --- 1. Parámetros ---
SIZE = 224
BATCH = 24            # EfficientNet-B0 usa más memoria por imagen que MobileNetV3-Small
N_VISTAS_AUG = 2
SEED_VISTAS = 999     # LA MISMA de A y B: las vistas son idénticas antes de desequilibrar
SEMILLAS = [123, 7, 42]

# Proporción que se conserva de cada clase. 1.0 / 0.30 / 0.10 -> desequilibrio 10:3:1.
PROPORCION_CLASE = {0: 1.00, 1: 0.30, 2: 0.10}

EPOCAS_CABEZA = 8     # fase 1: solo el clasificador, backbone congelado
EPOCAS_FT = 4         # fase 2: fine-tuning del último bloque (mismo esquema que A y B)
LR_CABEZA = 1e-3
LR_FINETUNE = 1e-5
DROPOUT = 0.3

FOCAL_GAMMA = 2.0     # el valor del paper (Lin et al. 2017)

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AQUI = Path(__file__).resolve().parent

# --- 3. Las tres estrategias frente al desequilibrio ---
def focal_loss(logits, objetivo, alpha: torch.Tensor, gamma: float = FOCAL_GAMMA):
    """Focal Loss (Lin et al. 2017): CE reponderada para que los ejemplos FÁCILES pesen poco.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    La intuición, en el caso de esta clase minoritaria: con cross-entropy plana, las 1000
    diapositivas fáciles de la clase mayoritaria —que el modelo ya clasifica con p=0.99—
    siguen aportando gradiente, y en suma tapan el de las 30 imágenes difíciles de la clase
    rara. El factor (1 - p_t)^gamma vale casi 0 cuando p_t≈1, así que esos ejemplos dejan de
    aportar y el gradiente queda dominado por lo que el modelo todavía NO resuelve.

    alpha compensa aparte la frecuencia de cada clase; γ y α atacan cosas distintas y por eso
    se usan juntos: γ pondera por DIFICULTAD, α por RAREZA.
    """
    log_p = F.log_softmax(logits, dim=1)
    log_pt = log_p.gather(1, objetivo.view(-1, 1)).squeeze(1)
    pt = log_pt.exp()
    at = alpha.to(logits.device).gather(0, objetivo)
    return (-at * (1.0 - pt) ** gamma * log_pt).mean()


ESTRATEGIAS = {
    "C0_sin_correccion": "cross-entropy plana (línea base: no hace nada con el desequilibrio)",
    "C1_class_weights": "cross-entropy ponderada por 1/frecuencia de clase",
    "C2_focal_sampler": "Focal Loss (γ=2, α=1/frec) + WeightedRandomSampler",
}


# --- 4. Datos ---
def submuestrear_desequilibrado(Y: np.ndarray, proporcion: dict, semilla: int) -> np.ndarray:
    """Índices que quedan tras conservar `proporcion[clase]` de cada clase.

    El submuestreo se hace por IMAGEN y no por vista: las N vistas de una misma imagen son
    consecutivas y se conservan o se descartan juntas. Descartar vistas sueltas dejaría el
    mismo contenido presente en el dataset con menos augmentation, que no es lo mismo que
    tener menos datos — y el experimento mide justamente el efecto de tener menos datos.
    """
    rng = np.random.default_rng(semilla)
    por_img = N_VISTAS_AUG + 1
    n_imgs = len(Y) // por_img
    etiquetas_img = Y[::por_img]

    conservados = []
    for clase, prop in proporcion.items():
        idx_clase = np.where(etiquetas_img == clase)[0]
        n = max(1, int(round(len(idx_clase) * prop)))
        conservados.append(rng.choice(idx_clase, size=n, replace=False))
    imgs_ok = np.sort(np.concatenate(conservados))

    return np.concatenate([np.arange(i * por_img, (i + 1) * por_img) for i in imgs_ok])


def lotes(n, batch, rng=None, pesos=None):
    """Índices por lote. Con `pesos`, muestrea CON reemplazo proporcional a ellos.

    Ese es el WeightedRandomSampler de la estrategia C2, escrito a mano para no depender del
    DataLoader (que en Windows arrastra el problema de spawn descrito en 09_scripts.py).
    Muestrear con reemplazo y pesos 1/frecuencia hace que, en promedio, cada batch tenga las
    tres clases en partes iguales aunque el dataset esté 10:3:1.
    """
    if pesos is not None:
        p = pesos / pesos.sum()
        orden = rng.choice(n, size=n, replace=True, p=p)
    else:
        orden = np.arange(n)
        if rng is not None:
            rng.shuffle(orden)
    for i in range(0, n, batch):
        yield orden[i:i + batch]


def a_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x_uint8)).permute(0, 3, 1, 2).float().to(
        DISPOSITIVO)


@torch.no_grad()
def logits_de(modelo, X, batch=32):
    modelo.eval()
    return np.concatenate([modelo(a_tensor(X[idx])).cpu().numpy()
                           for idx in lotes(len(X), batch)])


def evaluar(logits, Y, clases):
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    r = metricas.resumen_completo(Y, prob, clases)
    r["y_prob"] = prob
    return r


# --- 5. Entrenamiento de una variante ---
def hacer_perdida(estrategia: str, peso_clase: torch.Tensor):
    """Devuelve la función de pérdida de la estrategia. Es LA única diferencia entre las tres.

    Que las tres variantes compartan absolutamente todo lo demás —datos, arquitectura,
    semillas, épocas, optimizador, learning rate— es lo que permite atribuir la diferencia de
    resultados a la estrategia y no a otra cosa. Si C2 además entrenara más épocas, no se
    sabría si ganó por la Focal Loss o por el presupuesto extra.
    """
    if estrategia == "C0_sin_correccion":
        return lambda logits, y: F.cross_entropy(logits, y)
    if estrategia == "C1_class_weights":
        return lambda logits, y: F.cross_entropy(logits, y, weight=peso_clase.to(logits.device))
    return lambda logits, y: focal_loss(logits, y, peso_clase)


def pesos_de_clase(Y: np.ndarray, num_clases: int) -> tuple[torch.Tensor, np.ndarray]:
    """Pesos 1/frecuencia normalizados a media 1, y el peso por muestra para el sampler.

    La normalización a media 1 no es cosmética: sin ella, ponderar la pérdida escala su
    magnitud total y por lo tanto el tamaño efectivo del paso de Adam. C1 estaría entrenando
    con otro learning rate que C0 y la comparación mezclaría dos efectos.
    """
    conteo = np.bincount(Y, minlength=num_clases).astype(np.float64)
    inv = 1.0 / np.maximum(conteo, 1)
    return torch.tensor(inv / inv.mean(), dtype=torch.float32), inv[Y]


def entrenar(estrategia: str, E, X, Y, num_clases: int, semilla: int) -> ModeloC:
    """Entrena una variante completa: fase 1 sobre embeddings + fase 2 con fine-tuning.

    `E` son los embeddings de 1280-d ya pre-computados con el backbone congelado. Igual que
    en los modelos A y B, la fase 1 no necesita volver a pasar las imágenes por la red: el
    backbone está congelado, sus salidas no cambian entre épocas ni entre estrategias, y
    reusarlas convierte 8 épocas de EfficientNet-B0 (lo más caro del script) en 8 épocas de
    dos capas lineales. El resultado numérico es idéntico; el tiempo baja de horas a minutos.
    """
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    rng = np.random.default_rng(semilla)

    peso_clase, pesos_muestra_todos = pesos_de_clase(Y, num_clases)
    pesos_muestra = pesos_muestra_todos if estrategia == "C2_focal_sampler" else None
    perdida = hacer_perdida(estrategia, peso_clase)

    Y_t = torch.from_numpy(Y).to(DISPOSITIVO)
    E_t = torch.from_numpy(E).float().to(DISPOSITIVO)

    # --- Fase 1: solo la cabeza, sobre embeddings ---
    modelo = ModeloC(num_clases, dropout=DROPOUT).to(DISPOSITIVO)
    opt = torch.optim.Adam(modelo.cabeza.parameters(), lr=LR_CABEZA)
    modelo.cabeza.train()
    for _ in range(EPOCAS_CABEZA):
        for idx in lotes(len(E), BATCH, rng, pesos_muestra):
            opt.zero_grad()
            perdida(modelo.cabeza(E_t[idx]), Y_t[idx]).backward()
            opt.step()

    # --- Fase 2: fine-tuning del último bloque, con las imágenes ---
    descongelar_desde(modelo, len(modelo.features) - 2)
    opt = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad],
                           lr=LR_FINETUNE)
    for _ in range(EPOCAS_FT):
        modelo.train(); bn_a_eval(modelo)
        for idx in lotes(len(X), BATCH, rng, pesos_muestra):
            opt.zero_grad()
            perdida(modelo(a_tensor(X[idx])), Y_t[idx]).backward()
            opt.step()

    return modelo


# --- 6. Main ---
if __name__ == "__main__":
    t_inicio = time.time()

    particion = cargar_particion()
    clases = particion["clases"]
    num_clases = len(clases)
    print("=" * 78)
    print("  MODELO C — EfficientNet-B0 bajo DESEQUILIBRIO DE CLASES")
    print("=" * 78)
    print(f"Clases: {clases}   ·   dispositivo: {DISPOSITIVO}")

    imgs, indice = cargar_para_particion(particion)

    rutas_dev, y_dev = rutas_y_etiquetas(particion, "desarrollo")
    doms_dev = dominios_de(particion, rutas_dev)
    print(f"\nGenerando vistas (semilla {SEED_VISTAS}, las MISMAS de los modelos A y B)...")
    X_dev, Y_dev, _ = generar_vistas(imgs, indice, rutas_dev, y_dev, doms_dev,
                                     N_VISTAS_AUG, SEED_VISTAS, verbose=False)

    rutas_test, y_test = rutas_y_etiquetas(particion, "test")
    X_test = np.stack([imgs[indice[r]] for r in rutas_test])
    Y_test = np.array(y_test, dtype=np.int64)

    # --- 6.1 El desequilibrio ---
    idx_des = submuestrear_desequilibrado(Y_dev, PROPORCION_CLASE, semilla=123)
    X_des, Y_des = X_dev[idx_des], Y_dev[idx_des]

    print("\n" + "=" * 78)
    print("  EL DATASET DESEQUILIBRADO")
    print("=" * 78)
    print(f"  {'clase':<16}{'balanceado':>12}{'desequilibrado':>16}{'proporción':>12}")
    print("  " + "-" * 58)
    conteo_bal = np.bincount(Y_dev, minlength=num_clases)
    conteo_des = np.bincount(Y_des, minlength=num_clases)
    for i, c in enumerate(clases):
        print(f"  {c:<16}{conteo_bal[i]:>12}{conteo_des[i]:>16}"
              f"{PROPORCION_CLASE[i]:>11.0%}")
    print("  " + "-" * 58)
    ratio = conteo_des.max() / max(1, conteo_des.min())
    print(f"  vistas totales: {len(Y_dev)} -> {len(Y_des)}   ·   "
          f"ratio mayoritaria:minoritaria = {ratio:.1f} : 1")
    print(f"  TEST (sin tocar): {len(Y_test)} fotos  {conteos(particion, rutas_test)}")
    print("\n  Un modelo que predijera SIEMPRE la clase mayoritaria sacaría "
          f"{conteo_des.max() / conteo_des.sum():.1%} de accuracy sobre datos con esta")
    print("  distribución. Ese es el número contra el que hay que leer los resultados de C0.")

    # --- 6.2 Embeddings pre-computados (una sola vez para las tres estrategias) ---
    print("\nPre-computando embeddings (1280-d) con EfficientNet-B0 congelada...")
    t0 = time.time()
    extractor = ModeloC(num_clases, dropout=DROPOUT).to(DISPOSITIVO)
    extractor.eval()

    @torch.no_grad()
    def extraer(X):
        salidas = []
        for idx in lotes(len(X), BATCH):
            x = extractor.norm(a_tensor(X[idx]))
            salidas.append(torch.flatten(extractor.pool(extractor.features(x)), 1).cpu().numpy())
        return np.concatenate(salidas)

    E_des = extraer(X_des)
    print(f"  {E_des.shape}  ({time.time() - t0:.0f}s)")

    # --- 6.3 Las tres estrategias ---
    resultados = {e: [] for e in ESTRATEGIAS}
    modelos = {}
    for estrategia, descripcion in ESTRATEGIAS.items():
        print("\n" + "=" * 78)
        print(f"  {estrategia}  —  {descripcion}")
        print("=" * 78)
        for semilla in SEMILLAS:
            t0 = time.time()
            modelo = entrenar(estrategia, E_des, X_des, Y_des, num_clases, semilla)
            r = evaluar(logits_de(modelo, X_test), Y_test, clases)
            r["semilla"] = semilla
            resultados[estrategia].append(r)
            modelos[(estrategia, semilla)] = modelo
            print(f"  semilla {semilla:<5} macro_F1={r['macro_f1']:.4f}  "
                  f"recall_clase2={r['recall_saturada']:.4f}  "
                  f"accuracy={r['accuracy']:.4f}  0<->2={r['extremos_total']}  "
                  f"({time.time() - t0:.0f}s)")

    # --- 6.3 La comparación que importa ---
    agr = {e: metricas.agregar_semillas(v) for e, v in resultados.items()}

    print("\n" + "=" * 78)
    print("  COMPARACIÓN — por qué la ACCURACY es la métrica equivocada acá")
    print("=" * 78)
    print(f"  {'estrategia':<20}{'macro F1':>10}{'recall c2':>11}{'accuracy':>10}"
          f"{'AUC macro':>11}{'0<->2':>8}")
    print("  " + "-" * 72)
    for e in ESTRATEGIAS:
        a = agr[e]
        print(f"  {e:<20}{a['macro_f1']['media']:>10.4f}{a['recall_saturada']['media']:>11.4f}"
              f"{a['accuracy']['media']:>10.4f}{a['auc_macro']['media']:>11.4f}"
              f"{a['extremos_total']['media']:>8.1f}")
    print("  " + "-" * 72)

    base_f1 = agr["C0_sin_correccion"]["macro_f1"]["media"]
    base_rec = agr["C0_sin_correccion"]["recall_saturada"]["media"]
    base_acc = agr["C0_sin_correccion"]["accuracy"]["media"]
    print(f"\n  Línea base C0 (sin corrección):")
    print(f"    accuracy      = {base_acc:.4f}   <- el número que uno reportaría sin pensar")
    print(f"    recall clase 2= {base_rec:.4f}   <- lo que el producto realmente necesita")
    print(f"    macro F1      = {base_f1:.4f}")
    # El diagnóstico se decide con TRES ramas y no con dos, porque hay tres desenlaces
    # posibles y confundirlos haría que el informe contradijera a sus propios números:
    #   (a) recall hundido MIENTRAS la accuracy se sostiene -> patología de desequilibrio.
    #   (b) recall hundido Y accuracy también baja -> el modelo es malo en general; el
    #       desequilibrio no es (todavía) el problema dominante.
    #   (c) recall sano -> el desequilibrio no alcanzó a romper nada.
    if base_rec < 0.5 and base_acc - base_rec > 0.2:
        print("\n  DIAGNÓSTICO (a): la accuracy se sostiene mientras el recall de la clase rara")
        print("  se hunde. El modelo no aprendió a detectar diapositivas saturadas: aprendió")
        print("  que casi nunca las hay. Es el patrón exacto que describe el cierre de la")
        print("  asignatura, y la solución no está en los hiperparámetros sino en cómo se pesa")
        print("  la pérdida — que es precisamente lo que hacen C1 y C2.")
    elif base_rec < 0.5:
        print("\n  DIAGNÓSTICO (b): el recall de la clase rara está hundido, pero la accuracy")
        print("  TAMPOCO se sostiene. Ojo con la lectura: esto todavía no es la patología")
        print("  clásica del desequilibrio, es un modelo mal entrenado en general. Con tan")
        print("  pocos datos de la clase 2, el modelo perdió también las otras fronteras.")
        print("  El desequilibrio agrava un problema de datos; no lo reemplaza.")
    else:
        print("\n  DIAGNÓSTICO (c): el desequilibrio 10:3:1 NO alcanzó a hundir el recall de la")
        print("  clase 2. Lectura honesta: EfficientNet-B0 preentrenada extrae rasgos lo")
        print("  bastante separables como para que las pocas vistas de la clase rara alcancen.")
        print("  Para reproducir el fracaso clásico haría falta un desequilibrio más extremo")
        print("  (ej. 2% de la clase 2) o un backbone sin preentrenar.")

    for e in ("C1_class_weights", "C2_focal_sampler"):
        d_f1 = agr[e]["macro_f1"]["media"] - base_f1
        d_rec = agr[e]["recall_saturada"]["media"] - base_rec
        d_acc = agr[e]["accuracy"]["media"] - base_acc
        print(f"\n  {e} vs C0:  macro_F1 {d_f1:+.4f}   recall_c2 {d_rec:+.4f}   "
              f"accuracy {d_acc:+.4f}")
        if d_rec > 0 > d_acc:
            print("    (recall arriba y accuracy abajo = el intercambio esperado: la corrección")
            print("     compra sensibilidad en la clase rara pagando con falsos positivos.)")

    # --- 6.4 Elección del modelo que va a la app ---
    ganadora = max(ESTRATEGIAS, key=lambda e: agr[e]["macro_f1"]["media"])
    mejor_semilla = max(SEMILLAS, key=lambda s: next(
        r["macro_f1"] for r in resultados[ganadora] if r["semilla"] == s))
    print("\n" + "=" * 78)
    print(f"  ESTRATEGIA GANADORA POR MACRO-F1: {ganadora}  (semilla {mejor_semilla})")
    print("=" * 78)
    print("  Se elige por macro F1 y no por accuracy porque macro F1 pesa igual a las tres")
    print("  clases: es la métrica que NO se deja engañar por el desequilibrio.")
    print("  Este es el modelo que se exporta a ONNX y se sube a la app como 'Modelo C'.")

    cm = np.array(agr[ganadora]["cm_sumada"])
    print(f"\nMatriz ACUMULADA — {ganadora} sobre el test de fotos:")
    print(cm)
    metricas.reporte_por_clase(cm, clases)

    # --- 6.5 Figuras ---
    print("\nGenerando figuras...")
    metricas.plot_matriz(cm, clases,
                         f"Matriz Modelo C — {ganadora} (test de fotos)",
                         AQUI / "Figure_matriz_modelo_c_v9.png")
    ganadores = resultados[ganadora]
    metricas.plot_calibracion(Y_test, ganadores[0]["y_prob"],
                              f"Calibración Modelo C — {ganadora}",
                              AQUI / "Figure_calibracion_modelo_c_v9.png")
    curvas, micro, macro = metricas.curvas_roc_ovr(Y_test, ganadores[0]["y_prob"], num_clases)
    metricas.plot_roc(curvas, micro, macro, clases,
                      "ROC One-vs-Rest — Modelo C (EfficientNet-B0 desequilibrado)",
                      AQUI / "Figure_roc_modelo_c_v9.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    claves = ["accuracy", "macro_f1", "recall_saturada"]
    etiquetas = ["accuracy\n(engañosa)", "macro F1\n(honesta)", "recall clase 2\n(la del negocio)"]
    x = np.arange(len(claves))
    ancho = 0.26
    colores = ["#B00020", "#B45309", "#1F3864"]
    for i, e in enumerate(ESTRATEGIAS):
        vals = [agr[e][k]["media"] for k in claves]
        errs = [agr[e][k]["std"] for k in claves]
        ax.bar(x + (i - 1) * ancho, vals, ancho, yerr=errs, capsize=3,
               label=e, color=colores[i])
        for j, v in enumerate(vals):
            ax.text(j + (i - 1) * ancho, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 1.15); ax.set_ylabel("valor (media de 3 semillas)")
    ax.set_title("Modelo C — tres estrategias frente al desequilibrio 10:3:1\n"
                 "misma data, distinta forma de pesar la pérdida",
                 fontweight="bold", color="#1F3864")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(AQUI / "Figure_estrategias_desequilibrio_v9.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  guardado Figure_estrategias_desequilibrio_v9.png")

    # --- 6.6 Guardar ---
    def limpiar(a):
        return {k: a[k] for k in metricas.CLAVES_ESCALARES + ["cm_sumada", "n_semillas"]}

    with open(AQUI / "resultados_modelo_c_v9.json", "w", encoding="utf-8") as f:
        json.dump({
            "modelo": "C", "framework": "pytorch", "arquitectura": "EfficientNet-B0",
            "particion": particion["meta"],
            "desequilibrio": {"proporcion_conservada": PROPORCION_CLASE,
                              "conteo_balanceado": conteo_bal.tolist(),
                              "conteo_desequilibrado": conteo_des.tolist(),
                              "ratio": float(ratio)},
            "semillas": SEMILLAS,
            "estrategias": ESTRATEGIAS,
            "resultados": {e: limpiar(a) for e, a in agr.items()},
            "ganadora": ganadora,
            "semilla_exportada": mejor_semilla,
        }, f, indent=2, ensure_ascii=False)
    print("  guardado resultados_modelo_c_v9.json")

    mejor = modelos[(ganadora, mejor_semilla)].cpu().eval()
    torch.save({"state_dict": mejor.state_dict(), "clases": clases,
                "estrategia": ganadora, "semilla": mejor_semilla},
               AQUI / "modelopt_v9_modelo_c.pt")
    print("  guardado modelopt_v9_modelo_c.pt")

    with open(AQUI / "modelo_c_v9_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v9", "modelo": "C", "framework": "pytorch",
            "arquitectura": "EfficientNet-B0", "clases": clases,
            "entrenamiento": f"dataset desequilibrado 10:3:1 · {ganadora}",
            "entrada": {"alto": SIZE, "ancho": SIZE, "canales": 3, "rango": "0-255",
                        "orden": "NCHW"},
            "preprocesamiento": "ninguno en el cliente (Normalizador va dentro del modelo)",
            "salida": "logits de 3 clases; aplicar softmax en el cliente",
            "macro_f1_test_fotos": float(agr[ganadora]["macro_f1"]["media"]),
            "recall_clase2_test_fotos": float(agr[ganadora]["recall_saturada"]["media"]),
            "accuracy_test_fotos": float(agr[ganadora]["accuracy"]["media"]),
        }, f, indent=2, ensure_ascii=False)
    print("  guardado modelo_c_v9_meta.json")

    print(f"\nTiempo total: {(time.time() - t_inicio) / 60:.1f} min")
