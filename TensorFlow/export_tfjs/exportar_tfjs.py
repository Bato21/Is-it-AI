"""
exportar_tfjs.py — Exporta el modelo A de la v9 (.keras) a TensorFlow.js para la app Ionic.

Pipeline:
    TensorFlow/v9/modelotf_v9_finetune.keras
        -> tf.saved_model.save con firma de serving explícita
        -> tensorflowjs (convert_tf_saved_model)
        -> app/src/assets/modelos/tfjs_v9/   (model.json + shards de pesos)

IMPORTANTE: tensorflowjs fija sus PROPIAS versiones de TensorFlow. Correr este script en un
venv APARTE, NUNCA en TensorFlow/.venv — instalarlo ahí cambiaría la versión de TF del
entorno de entrenamiento y los resultados de la v9 dejarían de ser reproducibles.

    python -m venv TensorFlow/export_tfjs/.venv
    TensorFlow/export_tfjs/.venv/Scripts/activate      # Linux/Mac: source .../bin/activate
    pip install -r TensorFlow/export_tfjs/requirements_tfjs.txt
    python TensorFlow/export_tfjs/exportar_tfjs.py

EL MODELO EXPORTADO COME PÍXELES 0-255 CRUDOS.
    MobileNetV3 de Keras trae include_preprocessing=True: la normalización ImageNet vive
    DENTRO del grafo. Consecuencia directa en el cliente: el InferenceService de Angular
    pasa el canvas tal cual, sin dividir por 255 ni restar medias. Es la misma propiedad que
    los modelos ONNX consiguen con su capa `Normalizador` — los tres modelos de la app tienen
    el mismo contrato de entrada, y por eso el preprocesamiento del cliente es uno solo.

DOS PARCHES QUE ESTE SCRIPT NECESITA (y por qué):

  1. STUB DE LAS DEPENDENCIAS QUE NO SE USAN (jax, flax, tfdf, tensorflow_hub).
     `import tensorflowjs` arrastra el paquete entero, incluidos los conversores de JAX y de
     TensorFlow Decision Forests. Instalarlos de verdad es inviable acá: jax/flax hacen que
     el resolutor de pip entre en backtracking durante horas probando decenas de versiones
     de flax y optax, y tfdf además choca por versiones de protobuf (gencode 6.31 contra el
     runtime 5.29 que fija TF).

     Un CNN convertido desde SavedModel no ejecuta NADA de esos módulos. Así que se instala
     tensorflowjs con --no-deps y se registra un buscador en sys.meta_path que fabrica
     módulos vacíos para esos nombres. Es un stub honesto: si algún día la conversión SÍ
     necesitara jax, fallaría con un AttributeError sobre el stub en vez de dar un resultado
     incorrecto en silencio.

     Detalle no obvio: los stubs tienen que ser PAQUETES (con __path__), no módulos sueltos,
     porque tensorflowjs hace `import jax.experimental...`. Un módulo simple falla con
     "'jax' is not a package".

  2. DESACTIVAR EL PASE 'remap' DE GRAPPLER.
     MobileNetV3 usa hard-swish. El optimizador de grafos de TensorFlow fusiona
     conv + hardswish en la op `_FusedHardSwish`, que NINGÚN backend de TensorFlow.js
     implementa — ni WebGL ni CPU. El modelo convertido carga y después falla en la primera
     predicción con un error sobre una op desconocida. Sacando 'remap' del pipeline, el grafo
     queda con ops primitivas (Conv2D, Relu6, Mul) que tf.js sí soporta. Se pierde algo de
     fusión y se gana que el modelo CORRA en el navegador.

  Los dos parches son de la v4 y siguen siendo necesarios: la arquitectura no cambió.
"""

import importlib.abc
import importlib.machinery
import shutil
import sys
import types
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

# Módulos que tensorflowjs importa pero que la conversión de un SavedModel NO usa.
MODULOS_STUB = ("jax", "jaxlib", "flax", "tensorflow_decision_forests", "tensorflow_hub",
                "orbax", "optax", "chex")

KERAS = RAIZ / "TensorFlow" / "v9" / "modelotf_v9_finetune.keras"
SAVED = AQUI / "saved_model_v9"
DESTINO = RAIZ / "app" / "src" / "assets" / "modelos" / "tfjs_v9"
LADO = 224


class ModuloVacio(types.ModuleType):
    """Módulo que finge tener cualquier atributo, y que además es un PAQUETE.

    __path__ = [] es lo que hace que Python lo acepte como paquete y permita
    `import jax.experimental.algo`. Sin eso el import falla con "'jax' is not a package".
    __getattr__ fabrica submódulos a demanda, así que cualquier ruta de atributos funciona.
    """

    def __init__(self, nombre: str):
        super().__init__(nombre)
        self.__path__: list[str] = []

    def __getattr__(self, nombre: str):
        if nombre.startswith("__"):
            raise AttributeError(nombre)
        hijo = ModuloVacio(f"{self.__name__}.{nombre}")
        sys.modules[hijo.__name__] = hijo
        setattr(self, nombre, hijo)
        return hijo


class BuscadorDeStubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Intercepta los imports de MODULOS_STUB y devuelve módulos vacíos.

    Se registra en sys.meta_path ANTES de importar tensorflowjs. Solo actúa sobre los
    nombres declarados: cualquier otro import sigue el camino normal, así que un error real
    de dependencias se sigue viendo como error.
    """

    def find_spec(self, nombre, ruta=None, destino=None):
        if nombre.split(".")[0] in MODULOS_STUB:
            return importlib.machinery.ModuleSpec(nombre, self, is_package=True)
        return None

    def create_module(self, spec):
        return ModuloVacio(spec.name)

    def exec_module(self, modulo):
        pass


def main() -> None:
    if not KERAS.exists():
        raise SystemExit(
            f"Falta el modelo {KERAS.relative_to(RAIZ)}.\n"
            "Entrenalo primero:  TensorFlow/.venv/Scripts/python TensorFlow/v9/09_scripts.py"
        )

    import tensorflow as tf

    print(f"TensorFlow {tf.__version__} — cargando {KERAS.name} ...")
    modelo = tf.keras.models.load_model(KERAS)

    # A diferencia de la v4, el modelo de la v9 NO tiene capas de data augmentation adentro
    # (la augmentation de la v9 es numpy puro y vive en documentacion/aug_pantalla.py, fuera
    # del grafo). Así que no hay que reconstruir nada: el modelo guardado ya es exactamente
    # la ruta de inferencia, entrada 0-255 -> softmax. Es un beneficio no obvio de haber
    # sacado la augmentation del grafo: la exportación se simplifica y deja de ser un paso
    # frágil que había que revisar en cada versión.
    print(f"  entrada: {modelo.input_shape}   salida: {modelo.output_shape}")

    if SAVED.exists():
        shutil.rmtree(SAVED)

    # Firma de serving EXPLÍCITA. model.export() de Keras 3 genera una firma cuyo nodo de
    # salida el conversor de tfjs no encuentra ("Identity is not in graph"); definiendo la
    # tf.function a mano el grafo queda con un nodo de salida nombrado y localizable.
    print(f"Exportando SavedModel -> {SAVED.name} ...")

    @tf.function(input_signature=[
        tf.TensorSpec([None, LADO, LADO, 3], tf.float32, name="entrada")])
    def serving(x):
        return {"probabilidades": modelo(x, training=False)}

    tf.saved_model.save(modelo, str(SAVED), signatures={"serving_default": serving})

    # --- Parche 1: stubs de jax/flax/tfdf (ver encabezado) ---
    sys.meta_path.insert(0, BuscadorDeStubs())
    from tensorflowjs.converters import tf_saved_model_conversion_v2 as tfjs_conv

    # --- Parche 2: sacar 'remap' de Grappler (ver encabezado) ---
    _run_grappler_orig = tfjs_conv._run_grappler

    def _run_grappler_sin_remap(config, graph_def, graph, signature_def):
        opts = config.graph_options.rewrite_options.optimizers
        if "remap" in opts:
            opts.remove("remap")
        return _run_grappler_orig(config, graph_def, graph, signature_def)

    tfjs_conv._run_grappler = _run_grappler_sin_remap

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True, exist_ok=True)

    print(f"Convirtiendo SavedModel -> TensorFlow.js en {DESTINO.relative_to(RAIZ)} ...")
    tfjs_conv.convert_tf_saved_model(str(SAVED), str(DESTINO))

    total = sum(f.stat().st_size for f in DESTINO.glob("*"))
    archivos = sorted(f.name for f in DESTINO.glob("*"))
    print(f"\nListo. {len(archivos)} archivos · {total / 1e6:.1f} MB")
    for a in archivos:
        print(f"  {a}")
    print("\nSiguiente paso: generar el catálogo con las métricas reales")
    print("  python export/generar_catalogo.py")


if __name__ == "__main__":
    main()
