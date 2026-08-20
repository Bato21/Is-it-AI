"""
Clasificador de huella de IA en diapositivas — PyTorch
Versión 10  (COMPUERTA DE RECHAZO: el modelo puede decir "esto no es una diapositiva")

ESPEJO EXACTO de TensorFlow/v10/10_scripts.py. Misma partición (particion_v10.json), mismo
caché de imágenes, MISMAS vistas aumentadas (semilla fija 999 desde documentacion/imagenes.py,
así que los dos frameworks reciben el mismo array de píxeles byte a byte) y las mismas
métricas (documentacion/metricas.py).

Leer el encabezado del script de TensorFlow para el planteo: por qué el softmax de 3 clases no
puede abstenerse, por qué la compuerta es una cuarta clase y no un segundo modelo, y por qué
450 imágenes. Acá se documenta solo lo que CAMBIA por ser PyTorch.

LAS TRES DIFERENCIAS REALES ENTRE FRAMEWORKS (las mismas de la v9, y siguen valiendo)
-------------------------------------------------------------------------------------
  1. LA NORMALIZACIÓN NO VIENE INCLUIDA. Keras trae include_preprocessing=True y come 0-255
     crudos; torchvision espera la entrada ya normalizada y deja esa tarea al DataLoader. Como
     el modelo se exporta a ONNX y corre en un navegador, el DataLoader no viaja: por eso la
     normalización se mete DENTRO del nn.Module (capa `Normalizador`) y el .onnx también come
     0-255. Se replica por diseño lo que TensorFlow trae de fábrica.

  2. CONGELAR LAS BATCHNORM REQUIERE DOS COSAS, NO UNA. requires_grad_(False) congela los
     pesos (gamma, beta) pero NO las estadísticas: running_mean y running_var se siguen
     actualizando en cada forward mientras el módulo esté en train(). Hay que llamar .eval()
     sobre cada BN, y hay que volver a llamarlo DESPUÉS de cada model.train(). En Keras,
     trainable=False hace las dos cosas de una.

  3. NCHW vs NHWC. torchvision espera (N, 3, H, W); el caché compartido guarda NHWC (el
     formato natural de PIL/numpy), así que acá hay un permute explícito. Esa diferencia se
     propaga al .onnx y la app tiene que saberla: por eso el archivo de metadatos declara el
     orden de ejes.

POR QUÉ LA ABLACIÓN DE §5.7 NO SE REPITE ACÁ
--------------------------------------------
El script de TensorFlow corre una ablación sobre el tamaño de la clase de rechazo (25% / 50% /
100%) para justificar con una medición el número de 450 imágenes. Este script NO la repite, y
la omisión es deliberada:

  Esa ablación responde una pregunta de DISEÑO DE DATOS ("¿cuántas imágenes de rechazo hacen
  falta?"), no una pregunta de framework. La respuesta no puede depender de si el bucle de
  entrenamiento está escrito en Keras o en PyTorch — y si dependiera, el problema sería que el
  espejo está roto, no que haga falta medirlo dos veces.

  Correrla de los dos lados costaría el doble de cómputo para producir dos veces el mismo
  número. Lo que este script SÍ hace es reportar exactamente las mismas métricas de compuerta
  sobre el mismo test: si el recall de rechazo de PyTorch cae cerca del de TensorFlow, esa
  coincidencia es la validación cruzada del resultado de la ablación, y sale gratis.

Hiperparámetros de la cabeza: los que Optuna eligió en PyTorch/v7 (n_capas=2, units=256,
dropout=0.3, adam, lr=3.1e-3). NO son los mismos que los de TensorFlow y eso es correcto: cada
framework buscó los suyos en su propia v7.
"""

import json
import sys
import time
from pathlib import Path

# La consola de Windows (cp1252) no sabe codificar las flechas ↑/↓ de metricas.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "documentacion"))
import metricas  # noqa: E402

# Las arquitecturas son LAS MISMAS de la v9 (ver modelos_v10.py: reexporta a propósito).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelos_v10 import (  # noqa: E402
    EMB_MOBILENET, CabezaV9, ModeloV9, bn_a_eval, construir_backbone_mobilenet,
    descongelar_desde,
)
from imagenes import cargar_para_particion, generar_vistas  # noqa: E402
from particion_v10 import (  # noqa: E402
    CLASE_RECHAZO, N_ORDINALES, cargar_particion, conteos, dominios_de, rutas_y_etiquetas,
)

# --- 1. Parámetros (espejo del script de TensorFlow, idénticos a los de la v9) ---
SIZE = 224
BATCH_SIZE = 32
BATCH_FT = 16

N_VISTAS_AUG = 2
SEED_VISTAS = 999      # LA MISMA que TensorFlow: garantiza vistas idénticas

SEMILLAS = [123, 7, 42, 2024, 31]
SEMILLAS_FT = SEMILLAS

# TensorFlow descongela sus "últimas 12 capas", que caen dentro del último bloque invertido
# más la conv final. En torchvision la granularidad es distinta (bloques, no capas sueltas),
# así que se descongela el bloque EQUIVALENTE: features[11] (último InvertedResidual) y
# features[12] (Conv final). El script imprime los parámetros entrenables para que la
# equivalencia se pueda auditar en vez de creerla.
BLOQUES_DESCONGELADOS = 11
LR_FINETUNE = 1e-5
EPOCAS_FT = 4

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AQUI = Path(__file__).resolve().parent


# --- 2. Utilidades de entrenamiento ---
def lotes(n: int, batch: int, rng=None):
    """Índices por lote. Con rng, barajados. Reemplaza al DataLoader.

    Se evita torch.utils.data.DataLoader a propósito: en Windows los workers usan spawn, que
    reimporta el módulo entero en cada proceso y acá eso reconstruiría el caché de imágenes.
    Con los datos ya en RAM como arrays numpy, batchear a mano es más simple, más rápido y
    100% reproducible.
    """
    orden = np.arange(n)
    if rng is not None:
        rng.shuffle(orden)
    for i in range(0, n, batch):
        yield orden[i:i + batch]


def a_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    """NHWC uint8 -> NCHW float32 0-255 (ver diferencia 3)."""
    t = torch.from_numpy(np.ascontiguousarray(x_uint8)).permute(0, 3, 1, 2).float()
    return t.to(DISPOSITIVO)


@torch.no_grad()
def logits_de(modelo, X: np.ndarray, batch: int = BATCH_SIZE, sobre_embeddings=False):
    modelo.eval()
    salidas = []
    for idx in lotes(len(X), batch):
        entrada = (torch.from_numpy(X[idx]).float().to(DISPOSITIVO) if sobre_embeddings
                   else a_tensor(X[idx]))
        salidas.append(modelo(entrada).cpu().numpy())
    return np.concatenate(salidas)


def evaluar(logits: np.ndarray, Y, clases) -> dict:
    """Logits -> resumen completo. n_ordinales acota las métricas del eje al sub-bloque 0-1-2."""
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    r = metricas.resumen_completo(Y, prob, clases, n_ordinales=N_ORDINALES)
    r["y_prob"] = prob
    return r


def agregar(resumenes):
    return metricas.agregar_semillas(resumenes, claves=metricas.CLAVES_V10)


def optimizador(parametros, params, lr=None):
    lr = params["lr"] if lr is None else lr
    return {"adam": lambda: torch.optim.Adam(parametros, lr=lr),
            "rmsprop": lambda: torch.optim.RMSprop(parametros, lr=lr),
            "sgd": lambda: torch.optim.SGD(parametros, lr=lr, momentum=0.9)}[
        params["optimizer"]]()


# --- 3. Main ---
if __name__ == "__main__":
    t_inicio = time.time()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    with open(RAIZ / "PyTorch/v7/mejores_hiperparametros.json", encoding="utf-8") as f:
        cfg_v7 = json.load(f)
    PARAMS = cfg_v7["params"]
    EPOCHS_CABEZA = cfg_v7["epochs_refit"]

    # --- 3.1 Datos (idénticos a los de TensorFlow) ---
    particion = cargar_particion()
    clases = particion["clases"]
    num_clases = len(clases)
    IDX_RECHAZO = clases.index(CLASE_RECHAZO)
    print("Orden de clases:", clases)
    print(f"  eje ordinal: {clases[:N_ORDINALES]}   ·   compuerta: {clases[N_ORDINALES:]}")
    print(f"\nPartición v10 (huella {particion['meta']['huella_dataset']}) — "
          f"política: {particion['meta']['politica']}")
    print(f"Dispositivo: {DISPOSITIVO}")

    imgs, indice = cargar_para_particion(particion)

    bloques = {}
    for nombre in ("desarrollo", "test", "test_render"):
        rutas, y = rutas_y_etiquetas(particion, nombre)
        doms = dominios_de(particion, rutas)
        n_fotos = sum(1 for d in doms if d == "foto")
        print(f"  {nombre:<12}: {len(rutas):>4} imgs  ({n_fotos} fotos / "
              f"{len(rutas) - n_fotos} renders)  {conteos(particion, rutas)}")
        bloques[nombre] = (rutas, y, doms)

    print(f"\nGenerando vistas de desarrollo (1 limpia + {N_VISTAS_AUG} aumentadas, "
          f"semilla {SEED_VISTAS} — LAS MISMAS que TensorFlow)...")
    t0 = time.time()
    rutas_dev, y_dev, doms_dev = bloques["desarrollo"]
    X_dev, Y_dev, _ = generar_vistas(imgs, indice, rutas_dev, y_dev, doms_dev,
                                     N_VISTAS_AUG, SEED_VISTAS)
    print(f"  {X_dev.shape}  ({time.time() - t0:.0f}s)")

    tests = {n: (np.stack([imgs[indice[r]] for r in bloques[n][0]]),
                 np.array(bloques[n][1], dtype=np.int64))
             for n in ("test", "test_render")}

    print("\nHiperparámetros CONGELADOS de PyTorch/v7 (elegidos por CV, sin ver el test):")
    print(f"  {PARAMS}   ·   épocas de la cabeza: {EPOCHS_CABEZA}")
    print(f"Fine-tuning: features[{BLOQUES_DESCONGELADOS}:] · lr={LR_FINETUNE} · "
          f"{EPOCAS_FT} épocas · BatchNorm en eval()")

    # --- 3.2 Embeddings pre-computados (fase 1) ---
    backbone_base = construir_backbone_mobilenet()
    modelo_extractor = ModeloV9(backbone_base, nn.Identity()).to(DISPOSITIVO)
    modelo_extractor.eval()
    EMB_DIM = EMB_MOBILENET

    print(f"\nPre-computando embeddings ({EMB_DIM}-d) con el backbone congelado...")
    t0 = time.time()

    @torch.no_grad()
    def extraer(X):
        salidas = []
        for idx in lotes(len(X), BATCH_SIZE):
            salidas.append(modelo_extractor.embeddings(a_tensor(X[idx])).cpu().numpy())
        return np.concatenate(salidas)

    E_dev = extraer(X_dev)
    E_tests = {k: extraer(v[0]) for k, v in tests.items()}
    print(f"  desarrollo {E_dev.shape}  ({time.time() - t0:.0f}s)")

    # --- 3.3 FASE 1: cabeza sobre backbone congelado ---
    print("\n" + "=" * 78)
    print(f"  FASE 1 — BACKBONE CONGELADO  ·  {len(SEMILLAS)} semillas")
    print("=" * 78)

    res_f1 = {"test": [], "test_render": []}
    cabezas = {}
    Y_dev_t = torch.from_numpy(Y_dev).to(DISPOSITIVO)
    E_dev_t = torch.from_numpy(E_dev).float().to(DISPOSITIVO)
    perdida = nn.CrossEntropyLoss()

    for semilla in SEMILLAS:
        torch.manual_seed(semilla)
        np.random.seed(semilla)
        cabeza = CabezaV9(EMB_DIM, PARAMS, num_clases).to(DISPOSITIVO)
        opt = optimizador(cabeza.parameters(), PARAMS)
        rng = np.random.default_rng(semilla)

        cabeza.train()
        for _ in range(EPOCHS_CABEZA):
            for idx in lotes(len(E_dev), BATCH_SIZE, rng):
                opt.zero_grad()
                perdida(cabeza(E_dev_t[idx]), Y_dev_t[idx]).backward()
                opt.step()
        cabezas[semilla] = cabeza

        linea = f"  semilla {semilla:<5}"
        for nombre in ("test", "test_render"):
            r = evaluar(logits_de(cabeza, E_tests[nombre], sobre_embeddings=True),
                        tests[nombre][1], clases)
            r["semilla"] = semilla
            res_f1[nombre].append(r)
            linea += (f"   {nombre}: acc={r['accuracy']:.4f} "
                      f"rech={r['recall_rechazo']:.3f} fuga={r['fuga_no_diapositiva']}")
        print(linea)

    # --- 3.4 FASE 2: fine-tuning progresivo ---
    print("\n" + "=" * 78)
    print(f"  FASE 2 — FINE-TUNING PROGRESIVO  ·  {len(SEMILLAS_FT)} semillas")
    print("=" * 78)

    res_f2 = {"test": [], "test_render": []}
    modelos_ft = {}
    for semilla in SEMILLAS_FT:
        t0 = time.time()
        torch.manual_seed(semilla)
        np.random.seed(semilla)

        # Se parte de la cabeza YA entrenada en la fase 1 con esta MISMA semilla, igual que en
        # TensorFlow: la fase 2 continúa, no reinicia.
        modelo = ModeloV9(construir_backbone_mobilenet(), cabezas[semilla]).to(DISPOSITIVO)
        entrenables, total = descongelar_desde(modelo, BLOQUES_DESCONGELADOS)
        opt = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad],
                               lr=LR_FINETUNE)
        rng = np.random.default_rng(semilla)

        for _ in range(EPOCAS_FT):
            modelo.train()
            bn_a_eval(modelo)   # OBLIGATORIO después de cada train(): ver diferencia 2
            for idx in lotes(len(X_dev), BATCH_FT, rng):
                opt.zero_grad()
                perdida(modelo(a_tensor(X_dev[idx])), Y_dev_t[idx]).backward()
                opt.step()
        modelos_ft[semilla] = modelo

        linea = f"  semilla {semilla:<5}"
        for nombre in ("test", "test_render"):
            r = evaluar(logits_de(modelo, tests[nombre][0]), tests[nombre][1], clases)
            r["semilla"] = semilla
            res_f2[nombre].append(r)
            linea += (f"   {nombre}: acc={r['accuracy']:.4f} "
                      f"rech={r['recall_rechazo']:.3f} fuga={r['fuga_no_diapositiva']}")
        print(linea + f"   ({time.time() - t0:.0f}s, {entrenables}/{total} params "
                      f"entrenables del backbone)")

    # --- 3.5 Agregados y comparación A/B ---
    agr = {"fase1": {k: agregar(v) for k, v in res_f1.items()},
           "fase2": {k: agregar(v) for k, v in res_f2.items()}}
    res_f1_pareado = {k: [r for r in v if r["semilla"] in SEMILLAS_FT]
                      for k, v in res_f1.items()}
    agr_pareado = {k: agregar(v) for k, v in res_f1_pareado.items()}

    print()
    metricas.imprimir_agregado(agr["fase1"]["test"],
                               "FASE 1 — congelado · TEST DE FOTOS (accuracy comercial)",
                               claves=metricas.CLAVES_V10)
    print()
    metricas.imprimir_agregado(agr["fase2"]["test"],
                               "FASE 2 — fine-tuning · TEST DE FOTOS (accuracy comercial)",
                               claves=metricas.CLAVES_V10)

    print("\n" + "=" * 78)
    print(f"  COMPARACIÓN A/B sobre el TEST DE FOTOS  ·  semillas pareadas {SEMILLAS_FT}")
    print("=" * 78)
    print(f"  {'métrica':<28}{'F1 congelado':>14}{'F2 finetune':>13}{'F2-F1':>10}  veredicto")
    print("  " + "-" * 80)
    veredictos = {}
    for clave in metricas.CLAVES_V10:
        m1, s1 = agr_pareado["test"][clave]["media"], agr_pareado["test"][clave]["std"]
        m2, s2 = agr["fase2"]["test"][clave]["media"], agr["fase2"]["test"][clave]["std"]
        delta = m2 - m1
        if abs(delta) <= s1 + s2:
            v = "ruido"
        else:
            v = ("GANA FINETUNE" if (delta > 0) == metricas.MAS_ES_MEJOR[clave]
                 else "gana congelado")
        veredictos[clave] = v
        print(f"  {metricas.ETIQUETAS[clave]:<28}{m1:>14.4f}{m2:>13.4f}{delta:>+10.4f}  {v}")
    print("  " + "-" * 80)

    # --- 3.6 Brecha de dominio ---
    print("\n" + "=" * 78)
    print("  BRECHA DE DOMINIO — el mismo modelo sobre fotos y sobre renders")
    print("=" * 78)
    print(f"  {'':<18}{'TEST fotos':>13}{'TEST renders':>14}{'brecha':>10}")
    print("  " + "-" * 56)
    brechas = {}
    for fase, etiqueta in (("fase1", "FASE 1 congelado"), ("fase2", "FASE 2 finetune")):
        a_foto = agr[fase]["test"]["accuracy"]["media"]
        a_rend = agr[fase]["test_render"]["accuracy"]["media"]
        brechas[fase] = {"fotos": a_foto, "renders": a_rend, "brecha": a_rend - a_foto}
        print(f"  {etiqueta:<18}{a_foto:>13.4f}{a_rend:>14.4f}{a_rend - a_foto:>+10.4f}")
    print("  " + "-" * 56)
    b1, b2 = brechas["fase1"]["brecha"], brechas["fase2"]["brecha"]
    if abs(b2) < abs(b1):
        print(f"\n  El fine-tuning ACERCÓ los dominios: |brecha| {abs(b1):.4f} -> {abs(b2):.4f}.")
    else:
        print(f"\n  El fine-tuning SEPARÓ los dominios: |brecha| {abs(b1):.4f} -> {abs(b2):.4f}.")

    # --- 3.7 La compuerta de rechazo ---
    print("\n" + "=" * 78)
    print("  LA COMPUERTA — ¿deja de analizar lo que no es una diapositiva?")
    print("=" * 78)
    a2 = agr["fase2"]["test"]
    cm2 = np.array(a2["cm_sumada"])
    n_rechazo_test = int(cm2[IDX_RECHAZO, :].sum())
    n_diapos_test = int(cm2[:IDX_RECHAZO, :].sum())
    print(f"  Test acumulado sobre {len(SEMILLAS_FT)} semillas: {n_diapos_test} diapositivas "
          f"+ {n_rechazo_test} no-diapositivas")
    print(f"\n  {'métrica':<30}{'media':>10}{'±std':>9}   lectura")
    print("  " + "-" * 78)
    for clave, lectura in (
            ("recall_rechazo", "de lo que NO era diapositiva, cuánto atajó"),
            ("precision_rechazo", "de lo que rechazó, cuánto no era diapositiva"),
            ("fuga_no_diapositiva", "EL FALLO QUE CIERRA LA v10"),
            ("rechazo_indebido", "el costo: diapositivas reales descartadas")):
        d = a2[clave]
        print(f"  {metricas.ETIQUETAS[clave]:<30}{d['media']:>10.4f}{d['std']:>9.4f}   {lectura}")
    print("  " + "-" * 78)
    print(f"\n  Sin la compuerta (v9) la fuga habría sido del 100%: las "
          f"{n_rechazo_test // len(SEMILLAS_FT)} imágenes no-diapositiva")
    print("  del test habrían recibido igual un nivel de saturación de IA.")
    print("\n  Comparar este recall con el de TensorFlow es la validación cruzada del")
    print("  resultado: si dos frameworks con hiperparámetros distintos coinciden, la")
    print("  compuerta es una propiedad de los DATOS y no del bucle de entrenamiento.")

    # --- 3.8 Riesgo heredado ---
    print("\n" + "=" * 78)
    print("  RIESGO HEREDADO — ¿el moiré empuja las diapos humanas hacia 'saturada'?")
    print("=" * 78)
    for fase in ("fase1", "fase2"):
        a = agr[fase]["test"]
        print(f"  {fase}: 0->2 = {a['sin_ia_como_saturada']['media']:.1f} casos   ·   "
              f"2->0 = {a['saturada_como_sin_ia']['media']:.1f} casos   ·   "
              f"total extremos = {a['extremos_total']['media']:.1f}")

    for fase in ("fase1", "fase2"):
        cm = np.array(agr[fase]["test"]["cm_sumada"])
        print(f"\nMatriz ACUMULADA — {fase} sobre TEST DE FOTOS:")
        print(cm)
        metricas.reporte_por_clase(cm, clases)

    # --- 3.9 Figuras ---
    print("\nGenerando figuras...")
    metricas.plot_matriz(np.array(agr["fase1"]["test"]["cm_sumada"]), clases,
                         "Matriz v10 — Fase 1 congelado (test de fotos, PyTorch)",
                         AQUI / "Figure_matriz_fase1_v10.png")
    metricas.plot_matriz(np.array(agr["fase2"]["test"]["cm_sumada"]), clases,
                         "Matriz v10 — Fase 2 fine-tuning (test de fotos, PyTorch)",
                         AQUI / "Figure_matriz_fase2_v10.png")
    metricas.plot_calibracion(tests["test"][1], res_f2["test"][0]["y_prob"],
                              f"Calibración v10 PyTorch — fase 2 (semilla {SEMILLAS_FT[0]})",
                              AQUI / "Figure_calibracion_v10.png")
    curvas, micro, macro = metricas.curvas_roc_ovr(
        tests["test"][1], res_f2["test"][0]["y_prob"], num_clases)
    metricas.plot_roc(curvas, micro, macro, clases,
                      "ROC One-vs-Rest v10 — fase 2 (test de fotos, PyTorch)",
                      AQUI / "Figure_roc_v10.png")
    metricas.plot_semillas(res_f2["test"], ["accuracy", "macro_f1", "recall_rechazo", "qwk"],
                           "Dispersión entre semillas — v10 fase 2 (PyTorch)",
                           AQUI / "Figure_semillas_v10.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    ancho = 0.35
    ax.bar(x - ancho / 2, [brechas["fase1"]["fotos"], brechas["fase1"]["renders"]],
           ancho, label="Fase 1 (congelado)", color="#1F3864")
    ax.bar(x + ancho / 2, [brechas["fase2"]["fotos"], brechas["fase2"]["renders"]],
           ancho, label="Fase 2 (fine-tuning)", color="#B45309")
    ax.set_xticks(x)
    ax.set_xticklabels(["TEST fotos\n(accuracy comercial)", "TEST renders\n(laboratorio)"])
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Brecha de dominio — PyTorch v10", fontweight="bold", color="#1F3864")
    for i, fase in enumerate(("fase1", "fase2")):
        for j, k in enumerate(("fotos", "renders")):
            ax.text(j + (i - 0.5) * ancho, brechas[fase][k] + 0.015,
                    f"{brechas[fase][k]:.3f}", ha="center", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(AQUI / "Figure_brecha_dominio_v10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  guardado Figure_brecha_dominio_v10.png")

    # --- 3.10 Resultados y modelo de producción ---
    def limpiar(a):
        return {k: a[k] for k in metricas.CLAVES_V10 + ["cm_sumada", "n_semillas"]}

    with open(AQUI / "resultados_v10.json", "w", encoding="utf-8") as f:
        json.dump({
            "framework": "pytorch",
            "version": "v10",
            "clases": clases,
            "n_ordinales": N_ORDINALES,
            "clase_rechazo": CLASE_RECHAZO,
            "particion": particion["meta"],
            "semillas_fase1": SEMILLAS,
            "semillas_fase2": SEMILLAS_FT,
            "params_congelados_v7": PARAMS,
            "finetuning": {"bloques_descongelados": f"features[{BLOQUES_DESCONGELADOS}:]",
                           "lr": LR_FINETUNE, "epocas": EPOCAS_FT,
                           "batchnorm": "eval()", "batch_size": BATCH_FT},
            "vistas": {"n_aug": N_VISTAS_AUG, "seed_vistas": SEED_VISTAS},
            "fase1": {k: limpiar(v) for k, v in agr["fase1"].items()},
            "fase2": {k: limpiar(v) for k, v in agr["fase2"].items()},
            "comparacion_pareada": {k: limpiar(v) for k, v in agr_pareado.items()},
            "veredictos": veredictos,
            "brecha_dominio": brechas,
        }, f, indent=2, ensure_ascii=False)
    print("  guardado resultados_v10.json")

    # Se elige por MACRO F1 y no por accuracy (cambio respecto de la v9): con 4 clases y una
    # de ellas fácil, la accuracy premia acertar la compuerta y esconde el desempeño en el eje
    # ordinal, que es lo que el producto vende.
    mejor_semilla = max(SEMILLAS_FT,
                        key=lambda s: next(r["macro_f1"] for r in res_f2["test"]
                                           if r["semilla"] == s))
    mejor = modelos_ft[mejor_semilla].cpu().eval()
    torch.save({"state_dict": mejor.state_dict(), "params": PARAMS, "clases": clases,
                "semilla": mejor_semilla}, AQUI / "modelopt_v10_finetune.pt")
    print(f"  guardado modelopt_v10_finetune.pt  (fase 2, semilla {mejor_semilla})")

    r_mejor = next(r for r in res_f2["test"] if r["semilla"] == mejor_semilla)
    with open(AQUI / "modelo_v10_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v10", "framework": "pytorch", "arquitectura": "MobileNetV3-Small",
            "clases": clases, "n_ordinales": N_ORDINALES, "clase_rechazo": CLASE_RECHAZO,
            "entrada": {"alto": SIZE, "ancho": SIZE, "canales": 3, "rango": "0-255",
                        "orden": "NCHW"},
            "preprocesamiento": "ninguno en el cliente: la capa Normalizador viaja dentro "
                                "del modelo (y por lo tanto dentro del .onnx)",
            "salida": f"logits de {num_clases} clases; aplicar softmax en el cliente",
            "semilla": mejor_semilla,
            "criterio_seleccion": "macro F1 sobre el test de fotos",
            "accuracy_test_fotos": float(r_mejor["accuracy"]),
            "macro_f1_test_fotos": float(r_mejor["macro_f1"]),
            "recall_rechazo_test_fotos": float(r_mejor["recall_rechazo"]),
        }, f, indent=2, ensure_ascii=False)
    print("  guardado modelo_v10_meta.json")

    print(f"\nTiempo total: {(time.time() - t_inicio) / 60:.1f} min")
