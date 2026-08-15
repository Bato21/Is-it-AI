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

DE DÓNDE SALE CADA NÚMERO:
    Modelo A -> TensorFlow/v9/resultados_v9.json        · bloque fase2.test
    Modelo B -> PyTorch/v9/resultados_v9.json           · bloque fase2.test
    Modelo C -> PyTorch/v9/resultados_modelo_c_v9.json  · bloque de la estrategia ganadora

  Siempre el bloque "test", que en la v9 es el test de FOTOS: 207 fotos de pantalla, ningún
  render. O sea que lo que la app muestra es la accuracy COMERCIAL, no la de laboratorio.
  Mostrar la de renders al lado de un producto que solo consume fotos sería publicidad
  engañosa contra el propio usuario.

  El peso en MB no se declara: se mide sobre los archivos ya exportados. Si un modelo todavía
  no se exportó, queda marcado como no disponible y la app lo muestra deshabilitado en vez de
  fallar al intentar cargarlo.

Uso (no necesita ningún venv: es stdlib pura):
    python export/generar_catalogo.py
"""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ASSETS = RAIZ / "app" / "src" / "assets" / "modelos"
SALIDA = ASSETS / "catalogo.json"


def leer_json(ruta: Path):
    if not ruta.is_file():
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def metricas_de(bloque: dict | None) -> dict | None:
    """Extrae las 4 métricas que muestra la app de un bloque agregado de metricas.py.

    Cada métrica viene como {'media', 'std', 'min', 'max'} porque los entrenamientos corren
    con varias semillas. Se usa la MEDIA: reportar el máximo sería elegir la mejor corrida,
    que es exactamente el sesgo de selección que el proyecto viene corrigiendo desde la v7.
    """
    if not bloque:
        return None
    return {
        "accuracy": round(bloque["accuracy"]["media"], 4),
        "macroF1": round(bloque["macro_f1"]["media"], 4),
        "recallClase2": round(bloque["recall_saturada"]["media"], 4),
        "errores02": round(bloque["extremos_total"]["media"], 1),
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
    res_tf = leer_json(RAIZ / "TensorFlow/v9/resultados_v9.json")
    res_pt = leer_json(RAIZ / "PyTorch/v9/resultados_v9.json")
    res_c = leer_json(RAIZ / "PyTorch/v9/resultados_modelo_c_v9.json")

    tfjs_dir = ASSETS / "tfjs_v9"
    onnx_b = ASSETS / "modelo_b_v9.onnx"
    onnx_c = ASSETS / "modelo_c_v9.onnx"

    metricas_c = None
    entrenamiento_c = "EfficientNet-B0 sobre dataset desequilibrado 10:3:1"
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
            "ruta": "assets/modelos/tfjs_v9/model.json",
            "ordenEjes": "NHWC",
            "salida": "softmax",
            "tamano": 224,
            "pesoMB": peso_mb(tfjs_dir) or 0,
            "disponible": tfjs_dir.is_dir() and (tfjs_dir / "model.json").is_file(),
            "descripcion": "Fine-tuning en dos fases sobre fotos de pantalla. El más rápido "
                           "de los tres y el único que corre sobre la GPU del teléfono.",
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
            "ruta": "assets/modelos/modelo_b_v9.onnx",
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
            "ruta": "assets/modelos/modelo_c_v9.onnx",
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

    catalogo = {
        "version": "v9",
        "generado": "export/generar_catalogo.py",
        "clases": ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"],
        "particion": (res_tf or res_pt or {}).get("particion"),
        "nota": "Las métricas son la media entre semillas sobre el TEST DE FOTOS de la v9 "
                "(207 fotos de pantalla, ningún render): la accuracy comercial.",
        "modelos": modelos,
    }

    ASSETS.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(catalogo, f, indent=2, ensure_ascii=False)

    print(f"guardado {SALIDA.relative_to(RAIZ)}\n")
    print(f"  {'modelo':<14}{'disp.':>7}{'peso':>9}{'accuracy':>10}{'macroF1':>9}"
          f"{'recall c2':>11}")
    print("  " + "-" * 60)
    for m in modelos:
        k = m["metricas"]
        disp = "sí" if m["disponible"] else "NO"
        if k:
            print(f"  {m['etiqueta']:<14}{disp:>7}{m['pesoMB']:>7} MB"
                  f"{k['accuracy']:>10.4f}{k['macroF1']:>9.4f}{k['recallClase2']:>11.4f}")
        else:
            print(f"  {m['etiqueta']:<14}{disp:>7}{m['pesoMB']:>7} MB"
                  f"{'—':>10}{'—':>9}{'—':>11}   (sin resultados de entrenamiento)")

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
