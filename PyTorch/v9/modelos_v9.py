"""
Arquitecturas PyTorch de la v9 — FUENTE ÚNICA de las definiciones.

Por qué este módulo existe:
  Las clases de los modelos las necesitan TRES archivos distintos: los dos scripts de
  entrenamiento (09_scripts.py y 09_modelo_c_desequilibrado.py) y el exportador a ONNX
  (export/exportar_onnx.py). Definirlas en cada uno sería el mismo código escrito tres
  veces — y la copia del exportador es la peligrosa: si alguien cambia el dropout de la
  cabeza en el entrenamiento y no en el exportador, load_state_dict falla con un error de
  claves que se entiende, pero si cambia algo que NO altera los nombres de los parámetros
  (el orden de las operaciones en el forward, por ejemplo), carga sin quejarse y exporta un
  modelo silenciosamente distinto del que se entrenó.

  Con las clases acá, esa clase de bug no puede existir: los tres importan lo mismo.

Nota sobre los nombres de archivo: los scripts de entrenamiento se llaman `09_scripts.py`,
que no es un identificador válido de Python y por lo tanto NO se puede importar. Ese es el
motivo concreto por el que las clases viven en un módulo aparte con nombre importable en vez
de en el propio script de entrenamiento.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights, MobileNet_V3_Small_Weights, efficientnet_b0, mobilenet_v3_small,
)

# Normalización ImageNet: los mismos números que aplica TensorFlow adentro de su grafo
# (include_preprocessing=True) y que torchvision espera vía transforms.Normalize.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)

EMB_MOBILENET = 576      # salida del avgpool de MobileNetV3-Small
EMB_EFFICIENTNET = 1280  # salida del avgpool de EfficientNet-B0


class Normalizador(nn.Module):
    """0-255 (NCHW float) -> normalizado ImageNet. Va DENTRO del modelo.

    Esta capa es una decisión de despliegue, no de entrenamiento. Los pesos de torchvision
    esperan la entrada normalizada, y lo normal es hacerlo en el DataLoader con
    transforms.Normalize. Pero el DataLoader no viaja al navegador: si la normalización
    viviera ahí, habría que reescribirla en TypeScript y cualquier discrepancia de un decimal
    daría un modelo que "funciona pero predice raro", que es el bug más caro de diagnosticar
    del despliegue. Metiéndola en el módulo, viaja dentro del .onnx y el cliente entrega
    píxeles crudos.

    register_buffer y no nn.Parameter: son constantes que no se entrenan, pero sí tienen que
    moverse con .to(device), guardarse en el state_dict y exportarse al grafo ONNX.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("media", torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD_IMAGENET).view(1, 3, 1, 1) * 255.0)

    def forward(self, x):
        return (x - self.media) / self.std


class CabezaV9(nn.Module):
    """Cabeza del modelo B, con los hiperparámetros que Optuna eligió en PyTorch/v7.

    Devuelve LOGITS (sin softmax). Es la convención del proyecto desde la v8: el softmax se
    aplica aparte, porque una vez aplicado ya no se puede dividir por una temperatura, y
    porque las pérdidas de PyTorch son numéricamente más estables recibiendo logits.
    """

    def __init__(self, emb_dim: int, params: dict, num_clases: int):
        super().__init__()
        capas, dentro = [], emb_dim
        for _ in range(params["n_capas"]):
            capas += [nn.Linear(dentro, params["units"]), nn.ReLU(),
                      nn.Dropout(params["dropout"])]
            dentro = params["units"]
        capas.append(nn.Linear(dentro, num_clases))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


class ModeloV9(nn.Module):
    """MODELO B — MobileNetV3-Small de punta a punta: 0-255 NCHW -> logits."""

    def __init__(self, backbone: nn.Module, cabeza: nn.Module):
        super().__init__()
        self.norm = Normalizador()
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cabeza = cabeza

    def embeddings(self, x):
        return torch.flatten(self.pool(self.features(self.norm(x))), 1)

    def forward(self, x):
        return self.cabeza(self.embeddings(x))


class ModeloC(nn.Module):
    """MODELO C — EfficientNet-B0 de punta a punta: 0-255 NCHW -> logits.

    El clasificador original de torchvision es Dropout + Linear(1280, 1000). Se reemplaza la
    Linear por una de 3 salidas y se conserva el Dropout: patrón estándar de transfer
    learning, deja intactos todos los pesos convolucionales de ImageNet.
    """

    def __init__(self, num_clases: int, dropout: float = 0.3, preentrenado: bool = True):
        super().__init__()
        pesos = EfficientNet_B0_Weights.IMAGENET1K_V1 if preentrenado else None
        base = efficientnet_b0(weights=pesos)
        self.norm = Normalizador()
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cabeza = nn.Sequential(nn.Dropout(dropout), nn.Linear(EMB_EFFICIENTNET, num_clases))

    def embeddings(self, x):
        return torch.flatten(self.pool(self.features(self.norm(x))), 1)

    def forward(self, x):
        return self.cabeza(self.embeddings(x))


# --- Utilidades compartidas de congelado ---
def bn_a_eval(modulo: nn.Module) -> None:
    """Pone TODAS las BatchNorm en modo inferencia.

    EL GOTCHA DE PYTORCH, y el contraste más limpio con Keras en todo el proyecto:
    requires_grad_(False) congela los PESOS de una BN (gamma, beta) pero NO sus
    ESTADÍSTICAS: running_mean y running_var se siguen actualizando en cada forward mientras
    el módulo esté en train(). En Keras, capa.trainable = False hace las dos cosas de una.

    Y hay que llamarla DESPUÉS de cada model.train(), porque train() se propaga a los hijos y
    vuelve a activar las BN. Ese es el error silencioso: el script "congela" las BN una vez al
    principio, después llama train() dentro del bucle de épocas y las estadísticas se mueven
    igual, sin que nada avise. Con lr=1e-5 eso desestabiliza justo la fase que pide cambios
    mínimos.
    """
    for m in modulo.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def descongelar_desde(modelo: nn.Module, desde_bloque: int) -> tuple[int, int]:
    """Libera features[desde_bloque:] dejando TODAS las BatchNorm congeladas.

    Devuelve (parámetros entrenables, parámetros totales) del backbone, para poder auditar
    que la cantidad descongelada sea comparable con la del espejo en TensorFlow.
    """
    for p in modelo.features.parameters():
        p.requires_grad_(False)
    for bloque in modelo.features[desde_bloque:]:
        for p in bloque.parameters():
            p.requires_grad_(True)
    for m in modelo.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad_(False)

    entrenables = sum(p.numel() for p in modelo.features.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.features.parameters())
    return entrenables, total


def construir_backbone_mobilenet() -> nn.Module:
    return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)


# --- Carga de los modelos ya entrenados (lo que usa el exportador) ---
def cargar_modelo_b(ruta) -> tuple[ModeloV9, list[str]]:
    """Reconstruye el modelo B desde su checkpoint. Devuelve (modelo en eval(), clases)."""
    ckpt = torch.load(ruta, map_location="cpu", weights_only=False)
    clases = ckpt["clases"]
    cabeza = CabezaV9(EMB_MOBILENET, ckpt["params"], len(clases))
    modelo = ModeloV9(construir_backbone_mobilenet(), cabeza)
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo, clases


def cargar_modelo_c(ruta) -> tuple[ModeloC, list[str]]:
    """Reconstruye el modelo C desde su checkpoint. Devuelve (modelo en eval(), clases)."""
    ckpt = torch.load(ruta, map_location="cpu", weights_only=False)
    clases = ckpt["clases"]
    modelo = ModeloC(len(clases))
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo, clases
