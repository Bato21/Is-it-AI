"""
Arquitecturas PyTorch de la v10 — REEXPORTA las de la v9, a propósito.

Por qué este módulo no define nada nuevo:
  La v10 cambia los DATOS (una cuarta clase, la compuerta de rechazo), no la arquitectura.
  MobileNetV3-Small con su cabeza y EfficientNet-B0 con la suya son exactamente las mismas
  redes que en la v9; lo único que cambia es el `num_clases` del último Linear, y eso ya era
  un parámetro del constructor desde el primer día.

  Copiar las clases acá para "que la v10 tenga las suyas" sería el error que el encabezado de
  modelos_v9.py describe: dos definiciones que hay que mantener sincronizadas a mano, y un
  exportador que puede cargar pesos en una arquitectura silenciosamente distinta de la que se
  entrenó. Reexportar deja UNA sola definición viva en el repo.

  Y hay una razón experimental además de la de mantenimiento: el resultado que la v10 quiere
  medir es "qué le pasa al modelo cuando se le agrega la compuerta". Ese resultado solo es
  atribuible a la compuerta si la arquitectura es literalmente la misma — no "equivalente",
  la misma. Este import lo garantiza a nivel de código.

  `cargar_modelo_b` y `cargar_modelo_c` leen la lista de clases del propio checkpoint y
  construyen la cabeza con `len(clases)`, así que cargan los modelos de 4 clases de la v10 sin
  ningún cambio.

Qué SÍ es propio de la v10: los pesos entrenados (modelopt_v10_*.pt) y la cantidad de clases
que se le pasa al constructor. Nada más.
"""

import sys
from pathlib import Path

# Las clases viven en PyTorch/v9/modelos_v9.py. Se agrega esa carpeta al path en vez de mover
# el módulo, para no tocar la v9: sus scripts la siguen importando desde donde siempre.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v9"))

from modelos_v9 import (  # noqa: E402,F401
    EMB_EFFICIENTNET, EMB_MOBILENET, MEDIA_IMAGENET, STD_IMAGENET, CabezaV9, ModeloC, ModeloV9,
    Normalizador, bn_a_eval, cargar_modelo_b, cargar_modelo_c, construir_backbone_mobilenet,
    descongelar_desde,
)

__all__ = [
    "EMB_EFFICIENTNET", "EMB_MOBILENET", "MEDIA_IMAGENET", "STD_IMAGENET", "CabezaV9",
    "ModeloC", "ModeloV9", "Normalizador", "bn_a_eval", "cargar_modelo_b", "cargar_modelo_c",
    "construir_backbone_mobilenet", "descongelar_desde",
]
