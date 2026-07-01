"""Carga y validación del dataset para PyTorch v2.

Separa la lógica de datos de la de entrenamiento e incorpora chequeos de casos
borde con mensajes claros: carpeta inexistente, dataset vacío, alguna clase sin
imágenes, orden de clases inesperado y desbalance fuerte.
"""

from __future__ import annotations

import logging
from collections import Counter

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config

logger = logging.getLogger(__name__)

FACTOR_DESBALANCE = 10


def transform_train() -> transforms.Compose:
    """Transforms de entrenamiento (con data augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ]
    )


def transform_eval() -> transforms.Compose:
    """Transforms de evaluación/inferencia (sin augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ]
    )


def _validar(base: datasets.ImageFolder) -> None:
    """Valida estructura, clases y balance del dataset; aborta con mensaje claro."""
    if len(base) == 0:
        raise SystemExit(
            f"No hay imágenes en {config.DATASET_DIR}. Clasifica imágenes en "
            f"dataset/<clase>/ o genera datos de prueba: "
            f"python herramientas/crear_datos_prueba.py"
        )
    if base.classes != config.CLASES:
        raise SystemExit(
            f"Orden/numero de clases {base.classes} != {config.CLASES}. "
            f"Se esperan exactamente esas 3 carpetas en {config.DATASET_DIR}."
        )

    conteo = Counter(config.CLASES[t] for _, t in base.samples)
    vacias = [c for c in config.CLASES if conteo[c] == 0]
    if vacias:
        raise SystemExit(f"Estas clases no tienen imágenes: {vacias}.")

    menor, mayor = min(conteo.values()), max(conteo.values())
    if menor and mayor >= menor * FACTOR_DESBALANCE:
        logger.warning(
            "Dataset desbalanceado (de %d a %d por clase): %s.",
            menor,
            mayor,
            dict(conteo),
        )


def cargar_dataset() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Carga ImageFolder y lo parte 70% train · 15% val · 15% test.

    El split se siembra con ``config.SEED``. Train usa augmentation; val/test no.
    """
    base = datasets.ImageFolder(config.DATASET_DIR)
    _validar(base)

    n_total = len(base)
    n_train = int(n_total * config.SPLIT_TRAIN)
    n_val = int(n_total * config.SPLIT_VAL)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) == 0:
        raise SystemExit(
            f"Dataset demasiado pequeño para partir 70/15/15 ({n_total} imágenes). "
            f"Agrega más imágenes por clase."
        )

    g = torch.Generator().manual_seed(config.SEED)
    tr, va, te = random_split(base, [n_train, n_val, n_test], generator=g)

    # random_split devuelve Subsets sobre el MISMO dataset; reasignamos el
    # transform por split (train con augmentation, val/test sin).
    tr.dataset = datasets.ImageFolder(config.DATASET_DIR, transform=transform_train())
    eval_ds = datasets.ImageFolder(config.DATASET_DIR, transform=transform_eval())
    va.dataset = eval_ds
    te.dataset = eval_ds

    logger.info("Total: %d  |  Train: %d  Val: %d  Test: %d", n_total, n_train, n_val, n_test)
    return (
        DataLoader(tr, batch_size=config.BATCH_SIZE, shuffle=True),
        DataLoader(va, batch_size=config.BATCH_SIZE),
        DataLoader(te, batch_size=config.BATCH_SIZE),
    )
