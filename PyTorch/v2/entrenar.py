"""Entrenamiento del clasificador de huella de IA — PyTorch v2.

Equivalente en PyTorch a TensorFlow/v2/entrenar.py. Mismo flujo:
  - Transfer learning (MobileNetV3Large preentrenada en ImageNet, torchvision)
  - Data augmentation en el pipeline de entrenamiento
  - Dropout en la cabeza
  - Dos fases: cabeza congelada → fine-tuning de las últimas capas
  - Métricas manuales y mismas figuras que el lado TF (metricas.py)

Clases (orden fijo): 0_sin_ia · 1_rastro_ia · 2_saturada_ia

Genera, en esta carpeta:
  modelo_diapositivas.pt  — checkpoint {model_state, config} (lo usa consumidor.py)
  training_curves.png · confusion_matrix.png · roc_curves.png
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import datos
import metricas
from modelo import construir_modelo, guardar_checkpoint, set_features_trainable

logger = logging.getLogger("pytorch.entrenar")


def correr_epoca(
    net: nn.Module,
    dl: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Corre una época (entrena si hay optimizer) y devuelve (loss, accuracy)."""
    entrenando = optimizer is not None
    net.train(entrenando)
    total, correctos, suma = 0, 0, 0.0
    torch.set_grad_enabled(entrenando)
    for x, y in dl:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if entrenando:
            optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        if entrenando:
            loss.backward()
            optimizer.step()
        suma += loss.item() * x.size(0)
        correctos += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return suma / total, correctos / total


def _correr_fase(
    net: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    criterion: nn.Module,
    lr: float,
    epocas: int,
    etiqueta: str,
    hist: metricas.Historial,
) -> None:
    """Entrena una fase completa, registrando métricas por época en ``hist``."""
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    for e in range(1, epocas + 1):
        tl, ta = correr_epoca(net, train_dl, criterion, opt)
        vl, va = correr_epoca(net, val_dl, criterion)
        hist["acc"].append(ta)
        hist["val_acc"].append(va)
        hist["loss"].append(tl)
        hist["val_loss"].append(vl)
        logger.info(
            "[%s] %2d/%d  acc=%.3f val_acc=%.3f  loss=%.3f val_loss=%.3f",
            etiqueta,
            e,
            epocas,
            ta,
            va,
            tl,
            vl,
        )


def entrenar(net: nn.Module, train_dl: DataLoader, val_dl: DataLoader) -> metricas.Historial:
    """Entrena en dos fases (cabeza → fine-tuning) y devuelve el historial."""
    criterion = nn.CrossEntropyLoss()
    hist = metricas.Historial(acc=[], val_acc=[], loss=[], val_loss=[], corte=0)

    logger.info("FASE 1 — Cabeza (%d épocas), base congelada", config.EPOCHS_FASE1)
    set_features_trainable(net, trainable=False)
    _correr_fase(
        net, train_dl, val_dl, criterion, config.LR_FASE1, config.EPOCHS_FASE1, "cabeza", hist
    )
    hist["corte"] = len(hist["acc"])

    logger.info("FASE 2 — Fine-tuning (últimos %d bloques)", config.FINE_TUNE_BLOQUES)
    set_features_trainable(net, trainable=True, ultimos_bloques=config.FINE_TUNE_BLOQUES)
    _correr_fase(
        net, train_dl, val_dl, criterion, config.LR_FASE2, config.EPOCHS_FASE2, "fine", hist
    )
    return hist


def predecir_test(net: nn.Module, test_dl: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (y_true, y_pred, y_prob) sobre el conjunto de test."""
    net.eval()
    y_true_lotes, y_prob_lotes = [], []
    with torch.no_grad():
        for x, y in test_dl:
            probs = F.softmax(net(x.to(config.DEVICE)), dim=1).cpu().numpy()
            y_prob_lotes.append(probs)
            y_true_lotes.append(y.numpy())
    y_true = np.concatenate(y_true_lotes)
    y_prob = np.concatenate(y_prob_lotes)
    return y_true, np.argmax(y_prob, axis=1), y_prob


def main() -> None:
    config.configurar_logging()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    logger.info("Device: %s", config.DEVICE)

    train_dl, val_dl, test_dl = datos.cargar_dataset()
    net = construir_modelo(preentrenado=True).to(config.DEVICE)
    logger.info("Parámetros totales: %d", sum(p.numel() for p in net.parameters()))

    hist = entrenar(net, train_dl, val_dl)

    logger.info("Evaluando en test...")
    y_true, y_pred, y_prob = predecir_test(net, test_dl)
    cm = metricas.matriz_confusion(y_true, y_pred, config.N_CLASES)
    met = metricas.metricas_por_clase(cm)
    acc = metricas.accuracy(cm)

    # Salida orientada al usuario => print (no logging).
    metricas.imprimir_tabla(met, config.CLASES, acc)

    logger.info("Generando visualizaciones...")
    metricas.plot_curvas(hist, config.TITULO_CURVAS, config.ETIQUETA_LOSS, str(config.CURVAS_PATH))
    metricas.plot_matriz_confusion(
        cm, met, acc, config.CLASES, config.TITULO_MATRIZ, str(config.MATRIZ_PATH)
    )
    metricas.plot_roc(
        y_true, y_prob, config.CLASES, config.COLORES, config.TITULO_ROC, str(config.ROC_PATH)
    )

    guardar_checkpoint(net, config.MODELO_PATH)
    logger.info("Modelo guardado: %s", config.MODELO_PATH)
    print(f"\nListo. Modelo en {config.MODELO_PATH}")


if __name__ == "__main__":
    main()
