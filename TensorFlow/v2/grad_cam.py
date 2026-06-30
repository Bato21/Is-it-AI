"""
Grad-CAM para el modelo de diapositivas — TensorFlow v2

Técnica para ver gráficamente DÓNDE mira la red dentro de la imagen al decidir
la clase. Es la misma idea del ejemplo de clase (grad-cam.py con Xception),
adaptada a nuestro modelo con MobileNetV3Large como base.

  python grad_cam.py una_diapositiva.png
  -> genera grad_cam_output.png (mapa de calor superpuesto)

Útil para explicar el modelo: ¿se fija en el layout/plantilla, en el texto, en
las imágenes generadas? Eso respalda el criterio de las 3 clases.
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # consola UTF-8 en Windows
except Exception:
    pass

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
MODELO = AQUI / "modelo_diapositivas.keras"
IMG_SIZE = (224, 224)
CLASES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]


def encontrar_base(modelo):
    """Devuelve la capa sub-modelo (MobileNetV3Large) dentro del modelo."""
    for capa in modelo.layers:
        if isinstance(capa, tf.keras.Model):
            return capa
    raise RuntimeError("No se encontró el sub-modelo base en el modelo cargado.")


def construir_grad_model(modelo, base):
    """
    Reconstruye un modelo que va de la entrada del base (0-255) a
    [mapa de características del base, salida de clases].

    No usamos modelo.inputs porque la capa 'augmentation' es un sub-modelo con su
    propio Input y rompe el grafo (Graph disconnected). En inferencia el
    augmentation es identidad, así que partimos directo desde base.input y
    reaplicamos la cabeza (GAP -> Dense -> Dropout -> Softmax).
    """
    idx = modelo.layers.index(base)
    cabeza = modelo.layers[idx + 1:]
    inp = tf.keras.Input(shape=base.input_shape[1:])   # entrada nueva y limpia (0-255)
    feat = base(inp)                                    # llamar al base como capa
    x = feat
    for capa in cabeza:
        x = capa(x)
    return tf.keras.models.Model(inp, [feat, x])


def heatmap_gradcam(img_array, modelo, base, pred_index=None):
    grad_model = construir_grad_model(modelo, base)
    with tf.GradientTape() as tape:
        feat, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        canal = preds[:, pred_index]
    grads = tape.gradient(canal, feat)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    feat = feat[0]
    heat = tf.squeeze(feat @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + tf.keras.backend.epsilon())
    return heat.numpy(), int(pred_index)


def guardar(img_path, heat, pred, alpha=0.4, out="grad_cam_output.png"):
    img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(img_path))
    heat = np.uint8(255 * heat)
    jet = plt.colormaps["jet"](np.arange(256))[:, :3]
    jet_heat = jet[heat]
    jet_heat = tf.keras.utils.array_to_img(jet_heat).resize((img.shape[1], img.shape[0]))
    jet_heat = tf.keras.utils.img_to_array(jet_heat)
    superp = tf.keras.utils.array_to_img(jet_heat * alpha + img)
    plt.imshow(superp); plt.axis("off")
    plt.title(f"Grad-CAM — predicción: {CLASES[pred]}")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Grad-CAM guardado en '{out}'")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python grad_cam.py <imagen.png>")
    if not MODELO.exists():
        sys.exit(f"No existe {MODELO}. Entrena primero: python entrenar.py")
    img_path = sys.argv[1]

    modelo = tf.keras.models.load_model(MODELO)
    base = encontrar_base(modelo)

    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), 0)  # 0-255
    try:
        heat, pred = heatmap_gradcam(arr, modelo, base)
        guardar(img_path, heat, pred)
    except Exception as exc:  # noqa: BLE001
        # Grad-CAM sobre un modelo guardado con base anidada + preprocesamiento
        # interno es delicado en TF 2.15. Extensión opcional: para una versión
        # 100% estable, reconstruir el base con include_preprocessing=False y
        # cargar los pesos, o usar el ejemplo de clase (grad-cam.py con Xception).
        print(f"[Grad-CAM opcional] no se pudo generar el mapa: {exc}")
        print("La predicción del modelo sí funciona (ver consumidor.py).")


if __name__ == "__main__":
    main()
