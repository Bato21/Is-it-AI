"""
generar_catalogo.py — Construye app/src/assets/modelos/catalogo.json.

QUÉ HACE Y POR QUÉ IMPORTA:
  La app muestra, al lado de cada modelo del selector, sus métricas reales. Esas métricas
  podrían estar escritas a mano en el TypeScript — y sería la decisión equivocada, por dos
  razones:

    1. Se desactualizan en silencio. Se reentrena un modelo, mejora, y la app sigue mostrando
       los números viejos. Nadie se entera hasta que alguien los cruza con el informe.
    2. Invitan a redondear. Un número escrito a mano es un número que alguien eligió; uno
       leído de un archivo es un número que salió de un experimento.

  Este script lee los JSON que producen los scripts de entrenamiento y arma el catálogo. Las
  métricas de la app y las del informe salen de la misma fuente, siempre.

VERSIÓN QUE SE PUBLICA: v10 (constante VERSION, único lugar donde figura).

DE DÓNDE SALE CADA NÚMERO:
    Modelo A -> TensorFlow/v10/resultados_v10.json        · bloque fase2.test
    Modelo B -> PyTorch/v10/resultados_v10.json           · bloque fase2.test
    Modelo C -> PyTorch/v10/resultados_modelo_c_v10.json  · bloque de la estrategia ganadora

  Siempre el bloque "test", que es el test de FOTOS: ningún render. O sea que lo que la app
  muestra es la accuracy COMERCIAL, no la de laboratorio. Mostrar la de renders al lado de un
  producto que solo consume fotos sería publicidad engañosa contra el propio usuario.

  El peso en MB no se declara: se mide sobre los archivos ya exportados. Si un modelo todavía
  no se exportó, queda marcado como no disponible y la app lo muestra deshabilitado en vez de
  fallar al intentar cargarlo.

LO QUE AGREGA LA v10 AL CATÁLOGO
--------------------------------
  · `clases` pasa a tener 4 entradas y aparece `nOrdinales`: la app necesita saber CUÁL de
    ellas es la compuerta para dibujar un veredicto distinto (no un nivel de saturación) en
    vez de tratarla como un cuarto escalón del eje.
  · Cada modelo suma `recallRechazo` y `fugas` a sus métricas. Son los dos números que
    describen la compuerta, y el selector los muestra porque son parte de lo que distingue a
    un modelo de otro tanto como la accuracy.

Uso (no necesita ningún venv: es stdlib pura):
    python export/generar_catalogo.py
"""

import json
from pathlib import Path

VERSION = "v10"

RAIZ = Path(__file__).resolve().parents[1]
ASSETS = RAIZ / "app" / "src" / "assets" / "modelos"
SALIDA = ASSETS / "catalogo.json"

# Fallback: si los JSON de entrenamiento no están, el catálogo igual se genera con estas
# clases para que la app abra. El orden es el alfabético de las carpetas del dataset, que es
# el que usan Keras e ImageFolder — cambiarlo rompería silenciosamente las predicciones.
CLASES_POR_DEFECTO = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia", "3_no_diapositiva"]
N_ORDINALES_POR_DEFECTO = 3


def leer_json(ruta: Path):
    if not ruta.is_file():
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def metricas_de(bloque: dict | None) -> dict | None:
    """Extrae las métricas que muestra la app de un bloque agregado de metricas.py.

    Cada métrica viene como {'media', 'std', 'min', 'max'} porque los entrenamientos corren
    con varias semillas. Se usa la MEDIA: reportar el máximo sería elegir la mejor corrida,
    que es exactamente el sesgo de selección que el proyecto viene corrigiendo desde la v7.

    Las dos últimas son de la v10 y se leen con .get() porque un JSON de la v9 no las tiene:
    así este script sigue funcionando contra resultados viejos en vez de explotar.
    """
    if not bloque:
        return None
    def media(clave, defecto=None):
        d = bloque.get(clave)
        return round(d["media"], 4) if d else defecto

    return {
        "accuracy": media("accuracy", 0),
        "macroF1": media("macro_f1", 0),
        "recallClase2": media("recall_saturada", 0),
        "errores02": round((bloque.get("extremos_total") or {"media": 0})["media"], 1),
        "recallRechazo": media("recall_rechazo"),
        "fugas": (round(bloque["fuga_no_diapositiva"]["media"], 1)
                  if "fuga_no_diapositiva" in bloque else None),
    }


def peso_mb(*rutas: Path) -> float | None:
    """Suma el tamaño real de los archivos exportados. None si falta alguno."""
    total = 0
    for r in rutas:
        if r.is_dir():
            archivos = list(r.glob("*"))
            if not archivos:
                return None
            total += sum(f.stat().st_size for f in archivos if f.is_file())
        elif r.is_file():
            total += r.stat().st_size
        else:
            return None
    return round(total / 1e6, 1)


def main() -> None:
    res_tf = leer_json(RAIZ / f"TensorFlow/{VERSION}/resultados_{VERSION}.json")
    res_pt = leer_json(RAIZ / f"PyTorch/{VERSION}/resultados_{VERSION}.json")
    res_c = leer_json(RAIZ / f"PyTorch/{VERSION}/resultados_modelo_c_{VERSION}.json")

    tfjs_dir = ASSETS / f"tfjs_{VERSION}"
    onnx_b = ASSETS / f"modelo_b_{VERSION}.onnx"
    onnx_c = ASSETS / f"modelo_c_{VERSION}.onnx"

    # Las clases salen de los resultados del entrenamiento y no de una constante: si alguien
    # agrega una clase al dataset y reentrena, la app se entera sola.
    fuente_clases = res_tf or res_pt or res_c or {}
    clases = fuente_clases.get("clases") or CLASES_POR_DEFECTO
    n_ordinales = fuente_clases.get("n_ordinales", N_ORDINALES_POR_DEFECTO)

    metricas_c = None
    entrenamiento_c = "EfficientNet-B0 sobre dataset desequilibrado"
    if res_c:
        ganadora = res_c.get("ganadora")
        metricas_c = metricas_de(res_c.get("resultados", {}).get(ganadora))
        entrenamiento_c = (f"dataset desequilibrado "
                           f"{res_c.get('desequilibrio', {}).get('ratio', 10):.0f}:1 · "
                           f"{ganadora}")

    modelos = [
        {
            "id": "a-tfjs",
            "nombre": "Modelo A — MobileNetV3 (TensorFlow.js)",
            "etiqueta": "A · TF.js",
            "framework": "TensorFlow / Keras",
            "arquitectura": "MobileNetV3-Small",
            "backend": "tfjs",
            "ruta": f"assets/modelos/tfjs_{VERSION}/model.json",
            "ordenEjes": "NHWC",
            "salida": "softmax",
            "tamano": 224,
            "pesoMB": peso_mb(tfjs_dir) or 0,
            "disponible": tfjs_dir.is_dir() and (tfjs_dir / "model.json").is_file(),
            "descripcion": "Fine-tuning en dos fases sobre fotos de pantalla, con compuerta "
                           "de rechazo. El más rápido de los tres y el único sobre WebGL.",
            "porQue": "Default del producto: menor latencia por foto y backend WebGL.",
            "metricas": metricas_de((res_tf or {}).get("fase2", {}).get("test")),
        },
        {
            "id": "b-onnx",
            "nombre": "Modelo B — MobileNetV3 (PyTorch → ONNX)",
            "etiqueta": "B · ONNX",
            "framework": "PyTorch",
            "arquitectura": "MobileNetV3-Small",
            "backend": "onnx",
            "ruta": f"assets/modelos/modelo_b_{VERSION}.onnx",
            "ordenEjes": "NCHW",
            "salida": "logits",
            "tamano": 224,
            "pesoMB": peso_mb(onnx_b) or 0,
            "disponible": onnx_b.is_file(),
            "descripcion": "El espejo de A entrenado en PyTorch: misma partición, mismas "
                           "vistas aumentadas, misma arquitectura.",
            "porQue": "Verifica en producción la paridad entre frameworks y sirve de "
                      "respaldo donde WebGL falla (ONNX corre sobre WASM).",
            "metricas": metricas_de((res_pt or {}).get("fase2", {}).get("test")),
        },
        {
            "id": "c-onnx",
            "nombre": "Modelo C — EfficientNet-B0 desequilibrado (ONNX)",
            "etiqueta": "C · Efficient",
            "framework": "PyTorch",
            "arquitectura": "EfficientNet-B0",
            "backend": "onnx",
            "ruta": f"assets/modelos/modelo_c_{VERSION}.onnx",
            "ordenEjes": "NCHW",
            "salida": "logits",
            "tamano": 224,
            "pesoMB": peso_mb(onnx_c) or 0,
            "disponible": onnx_c.is_file(),
            "descripcion": f"Otra familia de arquitectura, entrenada con {entrenamiento_c}.",
            "porQue": "Tercer voto de una arquitectura no emparentada: rompe los empates "
                      "cuando A y B discrepan.",
            "metricas": metricas_c,
        },
    ]

    particion = (fuente_clases or {}).get("particion") or {}
    n_test = None
    if res_tf:
        cm = (res_tf.get("fase2", {}).get("test") or {}).get("cm_sumada")
        n_semillas = (res_tf.get("fase2", {}).get("test") or {}).get("n_semillas") or 1
        if cm:
            n_test = sum(sum(fila) for fila in cm) // n_semillas

    catalogo = {
        "version": VERSION,
        "generado": "export/generar_catalogo.py",
        "clases": clases,
        # Cuántas de las clases forman el eje ordinal de saturación. Las que sobran son
        # compuertas: la app tiene que mostrarlas como "esto no es una diapositiva" y no como
        # un nivel más. Ver app/src/app/servicios/catalogo-modelos.ts.
        "nOrdinales": n_ordinales,
        "particion": particion,
        "nota": (f"Las métricas son la media entre semillas sobre el TEST DE FOTOS de la "
                 f"{VERSION}"
                 + (f" ({n_test} fotos, ningún render)" if n_test else "")
                 + ": la accuracy comercial."),
        "modelos": modelos,
    }

    ASSETS.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(catalogo, f, indent=2, ensure_ascii=False)

    print(f"guardado {SALIDA.relative_to(RAIZ)}   ·   {len(clases)} clases "
          f"({n_ordinales} ordinales + {len(clases) - n_ordinales} compuerta)\n")
    print(f"  {'modelo':<14}{'disp.':>7}{'peso':>9}{'accuracy':>10}{'macroF1':>9}"
          f"{'recall c2':>11}{'rec.rech':>10}{'fugas':>7}")
    print("  " + "-" * 78)
    for m in modelos:
        k = m["metricas"]
        disp = "sí" if m["disponible"] else "NO"
        if k:
            rech = f"{k['recallRechazo']:.4f}" if k["recallRechazo"] is not None else "—"
            fug = f"{k['fugas']:.1f}" if k["fugas"] is not None else "—"
            print(f"  {m['etiqueta']:<14}{disp:>7}{m['pesoMB']:>7} MB"
                  f"{k['accuracy']:>10.4f}{k['macroF1']:>9.4f}{k['recallClase2']:>11.4f}"
                  f"{rech:>10}{fug:>7}")
        else:
            print(f"  {m['etiqueta']:<14}{disp:>7}{m['pesoMB']:>7} MB"
                  f"{'—':>10}{'—':>9}{'—':>11}{'—':>10}{'—':>7}"
                  "   (sin resultados de entrenamiento)")

    faltan = [m["etiqueta"] for m in modelos if not m["disponible"]]
    if faltan:
        print(f"\n  Faltan exportar: {', '.join(faltan)}")
        print("    Modelo A:  TensorFlow/export_tfjs/.venv/Scripts/python "
              "TensorFlow/export_tfjs/exportar_tfjs.py")
        print("    Modelos B/C:  PyTorch/.venv/Scripts/python export/exportar_onnx.py")
    else:
        print("\n  Los tres modelos están exportados y disponibles para la app.")


if __name__ == "__main__":
    main()
