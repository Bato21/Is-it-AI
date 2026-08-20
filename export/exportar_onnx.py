"""
Exportación a ONNX de los modelos PyTorch de la versión de producción — modelo B y modelo C.

VERSIÓN QUE SE EXPORTA: v10 (4 clases — eje ordinal 0-1-2 + compuerta de rechazo).
  Está en la constante VERSION de abajo. Es el único lugar donde figura: las rutas de
  entrada, las de salida y la partición con la que se verifica se derivan de ella. Volver a
  exportar la v9 es cambiar esa línea.

QUÉ ES ONNX Y POR QUÉ ES EL PASO QUE CONVIERTE UN EXPERIMENTO EN UN PRODUCTO:
  Un .pt de PyTorch es un diccionario de tensores más la clase Python que sabe cómo usarlos.
  Para cargarlo hace falta PyTorch instalado y el código fuente del modelo. En un navegador
  no hay ninguna de las dos cosas.

  ONNX (Open Neural Network Exchange) guarda el GRAFO COMPLETO: las operaciones, su orden,
  sus atributos y los pesos, en un formato que no depende del framework que lo creó. Un
  .onnx lo puede ejecutar ONNX Runtime en C++, en Python, en C# o —como acá— en WebAssembly
  dentro de un navegador, sin PyTorch por ningún lado. Eso es exactamente lo que el cierre de
  la asignatura llama transferencia tecnológica: "convertir un modelo a ONNX, verificarlo y
  desplegarlo en una app Ionic es el pipeline exacto que separa un experimento académico de
  un producto de ingeniería".

CÓMO EXPORTA torch.onnx.export (y por qué importa saberlo):
  Por TRACING: corre el modelo una vez con un tensor de ejemplo y anota las operaciones que
  se ejecutaron. Consecuencia práctica: lo que no se ejecutó en esa pasada, no queda en el
  grafo. Un `if` que dependa del contenido del tensor se congela en la rama que tocó ese día.
  Estos modelos no tienen ramas condicionales, así que el tracing los captura enteros
  — pero es la limitación que hay que poder nombrar si preguntan.

  Por eso el modelo se pone en .eval() ANTES de exportar. Si quedara en train(), el tracing
  capturaría el Dropout ACTIVO (que apagaría neuronas al azar en producción) y las BatchNorm
  usando estadísticas del batch en vez de las acumuladas. Sería un modelo que da resultados
  distintos en cada corrida y nadie entendería por qué.

dynamic_axes:
  Se declara el eje 0 (batch) como dinámico. Sin eso, el .onnx queda fijo al batch del tensor
  de ejemplo — 1 — y pedirle 4 imágenes de una falla. La app hoy manda de a una, pero
  procesar un lote es la optimización obvia si mañana se agrega "analizar una galería
  entera", y dejarlo dinámico no cuesta nada.

VERIFICACIÓN (el paso que no se puede saltear):
  Exportar es silencioso: produce un archivo aunque el grafo esté mal. Este script compara,
  sobre imágenes REALES del dataset, las salidas de PyTorch contra las de ONNX Runtime y
  aborta si difieren más que la tolerancia. Sin esta verificación, un error de exportación
  aparecería recién en el teléfono, como "el modelo predice cualquier cosa", y sería
  carísimo de rastrear.

Uso:
    PyTorch/.venv/Scripts/python export/exportar_onnx.py
    PyTorch/.venv/Scripts/python export/exportar_onnx.py --solo b
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

VERSION = "v10"       # ver el encabezado: la única mención de la versión en este archivo

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN = RAIZ / "PyTorch" / VERSION
sys.path.insert(0, str(RAIZ / "documentacion"))
sys.path.insert(0, str(ORIGEN))

from imagenes import cargar_para_particion  # noqa: E402
from modelos_v10 import cargar_modelo_b, cargar_modelo_c  # noqa: E402
from particion_v10 import cargar_particion, rutas_y_etiquetas  # noqa: E402

SALIDA = RAIZ / "app" / "src" / "assets" / "modelos"
PESOS_B = ORIGEN / f"modelopt_{VERSION}_finetune.pt"
PESOS_C = ORIGEN / f"modelopt_{VERSION}_modelo_c.pt"
OPSET = 17            # opset 17 lo soportan onnxruntime-web 1.20 y todos los runtimes actuales
TOLERANCIA = 1e-4     # diferencia máxima admitida entre PyTorch y ONNX Runtime
N_VERIFICACION = 12   # imágenes reales con las que se verifica la paridad


def construir_modelo_b():
    """Modelo B (MobileNetV3-Small) con sus pesos entrenados, en eval().

    cargar_modelo_b lee la lista de clases del propio checkpoint y arma la cabeza con
    len(clases), así que reconstruye tanto un modelo de 3 clases (v9) como uno de 4 (v10) sin
    saber de antemano cuál le toca.
    """
    return cargar_modelo_b(PESOS_B)


def construir_modelo_c():
    """Modelo C (EfficientNet-B0) con sus pesos entrenados, en eval()."""
    return cargar_modelo_c(PESOS_C)


def imagenes_de_verificacion(n: int) -> np.ndarray:
    """n imágenes REALES del test, en NCHW float32 0-255.

    Se verifica con imágenes del dataset y no con ruido aleatorio a propósito: el ruido
    activa la red de forma atípica y puede ocultar discrepancias que solo aparecen en el
    régimen donde el modelo realmente opera (por ejemplo en las ramas de hard-swish).
    """
    particion = cargar_particion()
    imgs, indice = cargar_para_particion(particion, verbose=False)
    rutas, _ = rutas_y_etiquetas(particion, "test")
    lote = np.stack([imgs[indice[r]] for r in rutas[:n]])
    return np.ascontiguousarray(lote.transpose(0, 3, 1, 2)).astype(np.float32)


def exportar(modelo: nn.Module, destino: Path, ejemplo: np.ndarray) -> None:
    """Exporta a ONNX con batch dinámico y verifica el grafo resultante."""
    import onnx

    modelo.eval()   # CRÍTICO: ver el encabezado (dropout y batchnorm)
    destino.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        modelo,
        torch.from_numpy(ejemplo[:1]),
        str(destino),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,      # precalcula las constantes: grafo más chico y rápido
        input_names=["entrada"],
        output_names=["logits"],
        dynamic_axes={"entrada": {0: "batch"}, "logits": {0: "batch"}},
    )

    # checker: valida que el grafo esté bien formado (tipos, shapes, ops del opset).
    onnx.checker.check_model(onnx.load(str(destino)))
    print(f"    grafo válido · {destino.stat().st_size / 1e6:.1f} MB")


def verificar_paridad(modelo: nn.Module, destino: Path, lote: np.ndarray) -> float:
    """Compara PyTorch vs ONNX Runtime sobre imágenes reales. Devuelve la diferencia máxima."""
    import onnxruntime as ort

    modelo.eval()
    with torch.no_grad():
        referencia = modelo(torch.from_numpy(lote)).numpy()

    sesion = ort.InferenceSession(str(destino), providers=["CPUExecutionProvider"])
    obtenido = sesion.run(None, {sesion.get_inputs()[0].name: lote})[0]

    diferencia = float(np.abs(referencia - obtenido).max())

    # Además de los logits, se comparan las CLASES predichas: una diferencia numérica chica
    # que igual cambiara un argmax sería un problema real, y al revés, una diferencia grande
    # en un logit irrelevante no lo sería. Se miran las dos cosas.
    iguales = int((referencia.argmax(1) == obtenido.argmax(1)).sum())
    print(f"    paridad PyTorch vs ONNX: max|dif| = {diferencia:.2e}   ·   "
          f"clases coincidentes {iguales}/{len(lote)}")
    return diferencia


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Exporta los modelos PyTorch de la {VERSION} a ONNX.")
    ap.add_argument("--solo", choices=["b", "c"], help="exportar solo uno de los dos")
    args = ap.parse_args()

    lote = imagenes_de_verificacion(N_VERIFICACION)
    print(f"Verificación con {len(lote)} imágenes reales del test  {lote.shape}\n")

    trabajos = []
    if args.solo in (None, "b"):
        trabajos.append(("Modelo B — MobileNetV3-Small", construir_modelo_b,
                         SALIDA / f"modelo_b_{VERSION}.onnx"))
    if args.solo in (None, "c"):
        trabajos.append(("Modelo C — EfficientNet-B0", construir_modelo_c,
                         SALIDA / f"modelo_c_{VERSION}.onnx"))

    resumen = {}
    for nombre, constructor, destino in trabajos:
        origen = PESOS_B if "MobileNet" in nombre else PESOS_C
        if not origen.is_file():
            print(f"  [SALTEADO] {nombre}: falta {origen.relative_to(RAIZ)}")
            print(f"             Entrenalo primero.")
            continue

        print(f"  {nombre}")
        modelo, clases = constructor()
        exportar(modelo, destino, lote)
        diferencia = verificar_paridad(modelo, destino, lote)

        if diferencia > TOLERANCIA:
            raise SystemExit(
                f"\nERROR: la exportación de {nombre} NO es fiel al modelo original.\n"
                f"  max|diferencia| = {diferencia:.2e}  >  tolerancia {TOLERANCIA:.0e}\n"
                "El .onnx quedó escrito pero NO hay que usarlo: revisá que el modelo esté en\n"
                "eval() y que la reconstrucción del grafo coincida con la del entrenamiento."
            )

        resumen[destino.name] = {
            "clases": clases,
            "mb": round(destino.stat().st_size / 1e6, 2),
            "max_diferencia": diferencia,
            "opset": OPSET,
        }
        print(f"    guardado {destino.relative_to(RAIZ)}\n")

    if resumen:
        with open(SALIDA / "verificacion_onnx.json", "w", encoding="utf-8") as f:
            json.dump({"tolerancia": TOLERANCIA, "modelos": resumen}, f, indent=2,
                      ensure_ascii=False)
        print(f"  guardado {(SALIDA / 'verificacion_onnx.json').relative_to(RAIZ)}")
        print("\nListo. Inspeccionar los grafos en https://netron.app arrastrando el .onnx:")
        print("  verificar nombre de entrada ('entrada'), shape [batch,3,224,224] y salida "
              "('logits').")


if __name__ == "__main__":
    main()
