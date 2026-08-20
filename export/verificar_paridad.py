"""
verificar_paridad.py — ¿Los modelos desplegados dicen lo MISMO que los originales?

Es el último control antes de dar la versión por entregada, y cierra el bucle que abrieron
los scripts de exportación. La versión verificada está en la constante VERSION de abajo
(hoy: v10, con la compuerta de rechazo como cuarta clase).

    modelo A:  .keras  ->  TensorFlow.js   (se verifica ACÁ, con Node)
    modelo B:  .pt     ->  ONNX            (se verifica en export/exportar_onnx.py)
    modelo C:  .pt     ->  ONNX            (se verifica en export/exportar_onnx.py)

La rama de ONNX se puede verificar dentro de Python porque onnxruntime tiene binding de
Python. La de TensorFlow.js no: el runtime es JavaScript. Así que este script levanta un
servidor HTTP sobre la carpeta del modelo convertido, corre `app/scripts/paridad-tfjs.mjs`
con Node, y compara su salida contra la del .keras original.

El script de Node vive dentro de `app/` y no de `export/` por una razón concreta de Node: la
resolución de módulos ESM parte del directorio del ARCHIVO, no del cwd del proceso. Un .mjs
en `export/` no encontraría `@tensorflow/tfjs`, que está instalado en `app/node_modules`.

LA ENTRADA ES UN PATRÓN SINTÉTICO, NO UNA IMAGEN
------------------------------------------------
    x[i, j, c] = (i*7 + j*13 + c*29) % 256

Generado con la misma fórmula entera en Python y en JavaScript. Si se usara un JPEG, las
diferencias entre el decodificador de Pillow y el del navegador se sumarían a las del modelo
y el número medido no sería atribuible a nada. Con un patrón calculado, la entrada es
idéntica bit a bit y lo único que puede diferir es el modelo.

Uso:
    TensorFlow/.venv/Scripts/python export/verificar_paridad.py

Requiere Node y que la app tenga sus dependencias instaladas (`cd app && npm install`),
porque el script de Node importa @tensorflow/tfjs desde app/node_modules.
"""

import functools
import http.server
import json
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

VERSION = "v10"

RAIZ = Path(__file__).resolve().parents[1]
MODELO_TFJS = RAIZ / "app" / "src" / "assets" / "modelos" / f"tfjs_{VERSION}"
MODELO_KERAS = RAIZ / "TensorFlow" / VERSION / f"modelotf_{VERSION}_finetune.keras"
SCRIPT_NODE = RAIZ / "app" / "scripts" / "paridad-tfjs.mjs"
APP = RAIZ / "app"
SALIDA = RAIZ / "export" / f"paridad_{VERSION}.txt"

LADO = 224
PUERTO = 8123
# Criterio de aceptación. 0.02 sobre una probabilidad es holgado a propósito: lo que importa
# es que no haya un error ESTRUCTURAL (ejes cambiados, preprocesamiento distinto, una op mal
# convertida), que produce diferencias de décimas. Las diferencias de implementación float32
# entre TensorFlow y tf.js viven en el orden de 1e-5.
TOLERANCIA = 0.02


def patron_sintetico() -> np.ndarray:
    """El MISMO patrón que genera paridad_tfjs.mjs. Enteros: reproducible en cualquier lenguaje."""
    i = np.arange(LADO).reshape(LADO, 1, 1)
    j = np.arange(LADO).reshape(1, LADO, 1)
    c = np.arange(3).reshape(1, 1, 3)
    return ((i * 7 + j * 13 + c * 29) % 256).astype(np.float32)


class ServidorSilencioso(http.server.SimpleHTTPRequestHandler):
    def log_message(self, formato, *args):
        pass   # sin ruido: el log del servidor no aporta nada al reporte

    def end_headers(self):
        # El fetch de tf.js desde Node no es un navegador, pero CORS igual no molesta y
        # deja este mismo servidor servible desde una pestaña para depurar a mano.
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main() -> None:
    if not MODELO_TFJS.is_dir() or not (MODELO_TFJS / "model.json").is_file():
        raise SystemExit(
            f"Falta el modelo TensorFlow.js en {MODELO_TFJS.relative_to(RAIZ)}.\n"
            "Exportalo:  TensorFlow/export_tfjs/.venv/Scripts/python "
            "TensorFlow/export_tfjs/exportar_tfjs.py")
    if not MODELO_KERAS.is_file():
        raise SystemExit(f"Falta {MODELO_KERAS.relative_to(RAIZ)}.")
    if not (APP / "node_modules" / "@tensorflow" / "tfjs").is_dir():
        raise SystemExit("Falta @tensorflow/tfjs. Corré:  cd app && npm install")

    x = patron_sintetico()

    # --- Lado Python: el .keras original ---
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    print(f"Cargando {MODELO_KERAS.name} ...")
    modelo = tf.keras.models.load_model(MODELO_KERAS)
    probs_py = np.asarray(modelo.predict(x[None, ...], verbose=0))[0]
    print(f"  Python  : {np.round(probs_py, 6).tolist()}")

    # --- Lado JavaScript: el GraphModel convertido, servido por HTTP ---
    manejador = functools.partial(ServidorSilencioso, directory=str(MODELO_TFJS))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PUERTO), manejador) as servidor:
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        try:
            proceso = subprocess.run(
                ["node", str(SCRIPT_NODE), f"http://127.0.0.1:{PUERTO}/model.json"],
                cwd=str(APP), capture_output=True, text=True, timeout=180)
        finally:
            servidor.shutdown()

    if proceso.returncode != 0:
        print(proceso.stdout)
        print(proceso.stderr, file=sys.stderr)
        raise SystemExit("El script de Node falló.")

    salida_json = json.loads(proceso.stdout.strip().splitlines()[-1])
    probs_js = np.array(salida_json["probabilidades"], dtype=np.float64)
    print(f"  tf.js   : {np.round(probs_js, 6).tolist()}   (backend {salida_json['backend']})")

    # --- Veredicto ---
    diferencia = float(np.abs(probs_py - probs_js).max())
    clase_py, clase_js = int(probs_py.argmax()), int(probs_js.argmax())
    ok = diferencia <= TOLERANCIA and clase_py == clase_js

    print(f"\n  max|diferencia| = {diferencia:.2e}   (tolerancia {TOLERANCIA})")
    print(f"  clase Python = {clase_py}   ·   clase tf.js = {clase_js}")
    print(f"  {'OK: el modelo desplegado es fiel al entrenado.' if ok else 'FALLA'}")

    reporte = (
        f"Paridad Python <-> navegador (TensorFlow.js) — modelo A {VERSION}\n"
        "=" * 62 + "\n"
        "Entrada: patron sintetico x[i,j,c] = (i*7 + j*13 + c*29) % 256, 224x224x3.\n"
        "Se usa un patron generado y no una imagen para que la diferencia medida sea\n"
        "atribuible al MODELO y no al decodificador de JPEG de cada lado.\n\n"
        "Preprocesamiento en ambos lados: 0-255 crudo, SIN normalizar (la normalizacion\n"
        "ImageNet viaja dentro del grafo). Backend tf.js: CPU.\n\n"
        f"  Python (.keras, TF {tf.__version__})  : {np.round(probs_py, 4).tolist()}\n"
        f"  Navegador (tf.js, GraphModel)   : {np.round(probs_js, 4).tolist()}\n"
        f"  max |diferencia|                : {diferencia:.2e}   "
        f"(criterio: < {TOLERANCIA} -> {'OK' if ok else 'FALLA'})\n"
        f"  clase predicha                  : {clase_py} en ambos\n\n"
        "Nota: la exportacion desactiva el pase 'remap' de Grappler para evitar la op\n"
        "_FusedHardSwish, que ningun backend de tf.js implementa. Esta verificacion es la\n"
        "que confirma que ese cambio de grafo no altero el resultado.\n"
    )
    SALIDA.write_text(reporte, encoding="utf-8")
    print(f"  guardado {SALIDA.relative_to(RAIZ)}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main()
