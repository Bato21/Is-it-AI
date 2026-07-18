"""
exportar_tfjs.py - Exporta el modelo TF v4 (.keras) a TensorFlow.js para la app web.

Pipeline:
    modelotf_v4_mobilenetv3.keras
        -> model.export(saved_model_v4)        (API Keras 3 -> SavedModel)
        -> tensorflowjs (convert_tf_saved_model)
        -> web/model/  (model.json + shards de pesos)

IMPORTANTE: tensorflowjs fija sus propias versiones de TensorFlow. Correr este script en
un venv APARTE (ver requirements_tfjs.txt), NO en el entorno de entrenamiento.

    python -m venv TensorFlow/export_tfjs/.venv
    source TensorFlow/export_tfjs/.venv/bin/activate   # Windows: .venv\\Scripts\\activate
    pip install -r TensorFlow/export_tfjs/requirements_tfjs.txt
    python TensorFlow/export_tfjs/exportar_tfjs.py

La normalización ImageNet viaja DENTRO del modelo (include_preprocessing=True), así que el
model.json exportado espera píxeles 0-255 crudos: el navegador NO tiene que normalizar.
"""

import sys
import types
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
KERAS = ROOT / "TensorFlow" / "v4" / "modelotf_v4_mobilenetv3.keras"
SAVED = AQUI / "saved_model_v4"
WEB_MODEL = ROOT / "web" / "model"


def main() -> None:
    if not KERAS.exists():
        raise SystemExit(
            f"Falta el modelo {KERAS}.\n"
            "Entrená TF v4 primero:  python TensorFlow/v4/04_scripts.py"
        )

    import tensorflow as tf

    print(f"TensorFlow {tf.__version__} — cargando {KERAS.name} ...")
    model = tf.keras.models.load_model(KERAS)

    # Modelo de INFERENCIA sin las capas de data augmentation. En inferencia esas capas
    # (RandomRotation/Zoom/Brightness/Contrast) son no-op, pero guardan estado RNG
    # (seed_generator) que rompe el guardado del SavedModel. Reconstruimos la ruta de
    # inferencia reusando los MISMOS pesos entrenados (base preentrenada + cabeza), sin
    # augmentation. El preprocesamiento ImageNet sigue DENTRO de la base: entrada 0-255.
    # Recorremos las capas en orden y salteamos el Input y el Sequential de augmentation
    # (OJO: data_augmentation es un Sequential, que también es tf.keras.Model). Reusamos
    # las capas entrenadas (base preentrenada -> GAP -> Dropout -> Dense) con sus pesos.
    inf_in = tf.keras.Input(shape=(224, 224, 3))
    x = inf_in
    for capa in model.layers:
        if isinstance(capa, tf.keras.layers.InputLayer) or isinstance(capa, tf.keras.Sequential):
            continue  # entrada y data_augmentation: fuera del modelo de inferencia
        if isinstance(capa, (tf.keras.Model, tf.keras.layers.Dropout)):
            x = capa(x, training=False)  # base con BN en inferencia; dropout inactivo
        else:
            x = capa(x)
    inf_model = tf.keras.Model(inf_in, x)

    # SavedModel con firma de serving EXPLÍCITA (salida nombrada). model.export() de Keras 3
    # genera una firma cuyo nodo de salida el converter de tfjs no encuentra ("Identity is
    # not in graph"); con nuestra tf.function el grafo tiene un nodo de salida claro.
    print(f"Exportando SavedModel -> {SAVED} ...")

    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name="entrada")])
    def serving(x):
        return {"probabilidades": inf_model(x, training=False)}

    tf.saved_model.save(inf_model, str(SAVED), signatures={"serving_default": serving})

    # SavedModel -> TensorFlow.js (GraphModel), en proceso.
    # tensorflowjs importa tensorflow_decision_forests al cargar, y esa dependencia choca
    # por versiones de protobuf (gencode 6.31 vs runtime 5.29 que fija TF). Un modelo CNN
    # NO usa las ops de tfdf, así que lo sustituimos por un stub vacío ANTES de importar
    # tensorflowjs: la conversión de un SavedModel normal no lo necesita. Forzamos el stub
    # (no setdefault) para que funcione también en un venv fresco donde tfdf esté instalado.
    sys.modules["tensorflow_decision_forests"] = types.ModuleType("tensorflow_decision_forests")
    from tensorflowjs.converters import tf_saved_model_conversion_v2 as tfjs_conv

    # MobileNetV3 usa hard-swish. El pase 'remap' de Grappler fusiona conv+hardswish en
    # la op _FusedHardSwish, que NINGÚN backend de tf.js implementa (WebGL ni CPU). Lo
    # sacamos del pipeline de Grappler: el grafo queda con ops primitivas (Conv2D, Relu6,
    # Mul) que tf.js sí soporta. Pierde algo de fusión, gana que el modelo CORRE en el
    # navegador. Parcheamos _run_grappler para que funcione también en un venv fresco.
    _run_grappler_orig = tfjs_conv._run_grappler

    def _run_grappler_sin_remap(config, graph_def, graph, signature_def):
        opts = config.graph_options.rewrite_options.optimizers
        if "remap" in opts:
            opts.remove("remap")
        return _run_grappler_orig(config, graph_def, graph, signature_def)

    tfjs_conv._run_grappler = _run_grappler_sin_remap

    WEB_MODEL.mkdir(parents=True, exist_ok=True)
    print(f"Convirtiendo SavedModel -> TensorFlow.js en {WEB_MODEL} ...")
    tfjs_conv.convert_tf_saved_model(str(SAVED), str(WEB_MODEL))

    print(f"\nListo. Modelo TF.js en {WEB_MODEL} (model.json + shards).")
    print("Smoke test:  cd web && python -m http.server  ->  http://localhost:8000/")


if __name__ == "__main__":
    main()
