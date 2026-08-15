"""
consumir_v9.py — CONSUMO independiente de los modelos de la v9.

Este script NO entrena, NO toca el dataset y NO depende de los scripts de entrenamiento.
Carga un modelo ya entrenado y clasifica imágenes. Es el equivalente de escritorio de lo que
hace la app Ionic, y existe por dos razones:

  1. Es el "script independiente de consumo del modelo" que pide el punto 3 del enunciado de
     la segunda entrega.
  2. Es la referencia contra la que se verifica la app. Si el navegador y este script dan
     probabilidades distintas para la MISMA imagen, el problema está en el preprocesamiento
     del cliente, no en el modelo. Sin una referencia así, diagnosticar eso es adivinar.

LOS TRES MODELOS, UNA INTERFAZ
------------------------------
    a  ->  TensorFlow/v9/modelotf_v9_finetune.keras     (Keras, NHWC, ya devuelve softmax)
    b  ->  app/src/assets/modelos/modelo_b_v9.onnx      (ONNX,  NCHW, devuelve logits)
    c  ->  app/src/assets/modelos/modelo_c_v9.onnx      (ONNX,  NCHW, devuelve logits)

Los tres esperan píxeles 0-255 SIN normalizar: la normalización ImageNet viaja dentro del
grafo. Por eso este script no normaliza nada — igual que la app.

Los modelos ONNX se consumen con onnxruntime, no con PyTorch. Es a propósito: demuestra que
el .onnx es autónomo (no necesita ni PyTorch ni el código de la clase para ejecutarse), que
es justamente el argumento por el que se exporta a ONNX.

QUÉ VENV USAR
-------------
    modelo a  ->  TensorFlow/.venv   (necesita tensorflow)
    modelos b y c  ->  PyTorch/.venv (necesita onnxruntime; NO necesita torch)

Uso:
    # una carpeta de imágenes con el modelo A
    TensorFlow/.venv/Scripts/python modelos/v9/consumir_v9.py --modelo a --ruta imagenes_a_probar

    # una sola imagen con el modelo B
    PyTorch/.venv/Scripts/python modelos/v9/consumir_v9.py --modelo b --ruta foto.jpg

    # los tres a la vez sobre la misma carpeta, comparando (necesita los dos venvs; se corre
    # con el de PyTorch y saltea el modelo A si no encuentra tensorflow)
    PyTorch/.venv/Scripts/python modelos/v9/consumir_v9.py --modelo todos --ruta imagenes_a_probar

    # cámara en vivo (requiere opencv-python instalado)
    PyTorch/.venv/Scripts/python modelos/v9/consumir_v9.py --modelo b --camara
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]

CLASES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
TITULOS = {
    "0_sin_ia": "Sin rastro de IA",
    "1_rastro_ia": "Rastro de IA",
    "2_saturada_ia": "Saturada de IA",
}
LADO = 224
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

MODELOS = {
    "a": {
        "nombre": "Modelo A — MobileNetV3-Small (TensorFlow.js / Keras)",
        "ruta": RAIZ / "TensorFlow/v9/modelotf_v9_finetune.keras",
        "tipo": "keras", "ejes": "NHWC", "salida": "softmax",
    },
    "b": {
        "nombre": "Modelo B — MobileNetV3-Small (PyTorch → ONNX)",
        "ruta": RAIZ / "app/src/assets/modelos/modelo_b_v9.onnx",
        "tipo": "onnx", "ejes": "NCHW", "salida": "logits",
    },
    "c": {
        "nombre": "Modelo C — EfficientNet-B0 desequilibrado (ONNX)",
        "ruta": RAIZ / "app/src/assets/modelos/modelo_c_v9.onnx",
        "tipo": "onnx", "ejes": "NCHW", "salida": "logits",
    },
}


# --- 1. Preprocesamiento (EL MISMO que el entrenamiento y que la app) ---
def preparar(img: Image.Image, ejes: str) -> np.ndarray:
    """PIL -> array float32 0-255 listo para el modelo.

    Tres detalles que tienen que coincidir EXACTAMENTE con el entrenamiento, o el modelo ve
    otra cosa de la que aprendió:

      convert("RGB")  : normaliza PNG con alfa y escala de grises a 3 canales.
      resize cuadrado : se APLASTA el aspecto (no se recorta). Es la convención del proyecto
                        desde la v1 — recortar una diapositiva 16:9 a cuadrado tira los
                        bordes laterales, que es donde suelen estar las marcas de generación.
      BILINEAR        : Pillow aplica el filtro escalado al factor de reducción, o sea hace
                        antialiasing real al achicar. Es lo que usó documentacion/imagenes.py.
    """
    arr = np.asarray(img.convert("RGB").resize((LADO, LADO), Image.BILINEAR),
                     dtype=np.float32)
    if ejes == "NHWC":
        return arr[None, ...]                       # (1, 224, 224, 3)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))[None, ...]   # (1, 3, 224, 224)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax estable: se resta el máximo antes de exponenciar para no desbordar."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# --- 2. Carga de cada tipo de modelo ---
def cargar(clave: str):
    """Devuelve una función predecir(array) -> probabilidades (3,)."""
    cfg = MODELOS[clave]
    if not cfg["ruta"].is_file():
        raise SystemExit(
            f"Falta el modelo {cfg['ruta'].relative_to(RAIZ)}.\n"
            "Entrenalo y exportalo primero (ver README de la raíz)."
        )

    if cfg["tipo"] == "keras":
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        modelo = tf.keras.models.load_model(cfg["ruta"])

        def predecir(x):
            return np.asarray(modelo.predict(x, verbose=0))[0]

        return predecir, cfg

    import onnxruntime as ort
    sesion = ort.InferenceSession(str(cfg["ruta"]), providers=["CPUExecutionProvider"])
    nombre_entrada = sesion.get_inputs()[0].name

    def predecir(x):
        crudo = sesion.run(None, {nombre_entrada: x})[0][0]
        return softmax(crudo)

    return predecir, cfg


# --- 3. Reporte ---
def imprimir(nombre_archivo: str, probs: np.ndarray, ms: float, ancho: int = 24) -> int:
    """Imprime el vector COMPLETO de probabilidades, no solo la ganadora.

    Es una regla del proyecto desde la v2: la distribución dice cuánto duda el modelo.
    "0.51 / 0.49 / 0.00" y "0.99 / 0.01 / 0.00" tienen la misma clase ganadora y son dos
    respuestas completamente distintas.
    """
    ganador = int(np.argmax(probs))
    print(f"\n  {nombre_archivo}")
    for i, clase in enumerate(CLASES):
        # Se usa "#" y no un bloque Unicode: la consola de Windows (cp1252) no puede
        # codificar los caracteres de bloque y los imprime como "?".
        barra = "#" * int(round(probs[i] * 30))
        marca = " <-" if i == ganador else "   "
        print(f"    {TITULOS[clase]:<18} {probs[i]:6.2%}  {barra:<30}{marca}")
    print(f"    -> {TITULOS[CLASES[ganador]]}   (confianza {probs[ganador]:.1%} · {ms:.0f} ms)")
    if probs[ganador] < 0.60:
        print("       AVISO: confianza baja. Sobre 3 clases, el azar es 33%: el modelo está dudando.")
    return ganador


# --- 4. Modos de uso ---
def modo_carpeta(claves: list[str], ruta: Path) -> None:
    archivos = ([ruta] if ruta.is_file()
                else sorted(p for p in ruta.iterdir()
                            if p.is_file() and p.suffix.lower() in EXTENSIONES))
    if not archivos:
        raise SystemExit(f"No hay imágenes en {ruta}")

    print(f"{len(archivos)} imagen(es) desde {ruta}")

    resultados = {}
    for clave in claves:
        try:
            predecir, cfg = cargar(clave)
        except SystemExit as e:
            print(f"\n[SALTEADO] modelo {clave}: {e}")
            continue
        except ImportError as e:
            print(f"\n[SALTEADO] modelo {clave}: falta una dependencia ({e}). "
                  f"Revisá qué venv usar en el encabezado.")
            continue

        print("\n" + "=" * 70)
        print(f"  {cfg['nombre']}")
        print("=" * 70)

        ganadores = []
        for archivo in archivos:
            with Image.open(archivo) as img:
                x = preparar(img, cfg["ejes"])
            t0 = time.perf_counter()
            probs = predecir(x)
            ms = (time.perf_counter() - t0) * 1000
            ganadores.append(imprimir(archivo.name, probs, ms))
        resultados[clave] = ganadores

    # Si se corrió más de un modelo, se compara: en qué imágenes coinciden y en cuáles no.
    if len(resultados) > 1:
        print("\n" + "=" * 70)
        print("  ACUERDO ENTRE MODELOS")
        print("=" * 70)
        claves_ok = list(resultados)
        desacuerdos = 0
        for i, archivo in enumerate(archivos):
            votos = [resultados[k][i] for k in claves_ok]
            if len(set(votos)) > 1:
                desacuerdos += 1
                detalle = "  ".join(f"{k}={TITULOS[CLASES[v]]}" for k, v in zip(claves_ok, votos))
                print(f"  DISCREPAN · {archivo.name}:  {detalle}")
        total = len(archivos)
        print(f"\n  {total - desacuerdos}/{total} imágenes con acuerdo unánime "
              f"({(total - desacuerdos) / total:.0%}).")
        if desacuerdos:
            print("  Las que discrepan son las genuinamente ambiguas: son las que conviene")
            print("  revisar a mano, y las que valen para ampliar el dataset.")


def modo_camara(clave: str) -> None:
    """Inferencia en vivo por webcam con OpenCV, apuntando a una diapositiva en pantalla."""
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "El modo cámara necesita opencv-python, que no está instalado en este venv:\n"
            "    pip install opencv-python\n"
            "(La app Ionic hace lo mismo con OpenCV.js y no necesita nada de esto.)"
        )

    predecir, cfg = cargar(clave)
    print(f"{cfg['nombre']}\nCámara abierta. 'q' para salir, ESPACIO para analizar el frame.")

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    ultimo = "sin analizar"
    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break

            # Varianza del Laplaciano: la MISMA medida de nitidez que usa la app antes de
            # aceptar una foto. Acá se muestra en pantalla para poder encuadrar bien.
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nitidez = cv2.Laplacian(gris, cv2.CV_64F).var()

            cv2.putText(frame, f"nitidez: {nitidez:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if nitidez >= 60 else (0, 0, 255), 2)
            cv2.putText(frame, ultimo, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 200, 0), 2)
            cv2.imshow("Is it AI? — consumo v9", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord(" "):
                if nitidez < 60:
                    ultimo = "foto movida: no se analiza"
                    print("\n  Frame rechazado por nitidez baja "
                          f"({nitidez:.0f} < 60). Estabilizá la cámara.")
                    continue
                # cv2 entrega BGR; PIL y el entrenamiento trabajan en RGB.
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                probs = predecir(preparar(img, cfg["ejes"]))
                g = imprimir("frame en vivo", probs, 0)
                ultimo = f"{TITULOS[CLASES[g]]} ({probs[g]:.0%})"
    finally:
        captura.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consumo independiente de los modelos v9 (no entrena nada).")
    ap.add_argument("--modelo", default="a", choices=["a", "b", "c", "todos"],
                    help="qué modelo usar (default: a)")
    ap.add_argument("--ruta", default=str(RAIZ / "imagenes_a_probar"),
                    help="imagen o carpeta de imágenes a clasificar")
    ap.add_argument("--camara", action="store_true",
                    help="inferencia en vivo por webcam (requiere opencv-python)")
    ap.add_argument("--listar", action="store_true",
                    help="solo mostrar qué modelos están disponibles y salir")
    args = ap.parse_args()

    if args.listar:
        print(f"  {'clave':<8}{'disponible':<13}{'formato':<10}{'ejes':<7}nombre")
        print("  " + "-" * 76)
        for clave, cfg in MODELOS.items():
            disp = "sí" if cfg["ruta"].is_file() else "NO"
            print(f"  {clave:<8}{disp:<13}{cfg['tipo']:<10}{cfg['ejes']:<7}{cfg['nombre']}")
        return

    claves = list(MODELOS) if args.modelo == "todos" else [args.modelo]

    if args.camara:
        if args.modelo == "todos":
            raise SystemExit("El modo cámara admite un solo modelo a la vez.")
        modo_camara(args.modelo)
        return

    ruta = Path(args.ruta)
    if not ruta.exists():
        raise SystemExit(f"No existe: {ruta}")
    modo_carpeta(claves, ruta)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main()
