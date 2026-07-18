"""
reporte_metricas.py - Reporte por clase de los modelos v4 YA ENTRENADOS (sin re-entrenar).

Reconstruye el split de validación (mismo seed 123 que el entrenamiento) sobre dataset/
y calcula, para cada framework disponible, la matriz de confusión + el reporte por clase
(precision, recall, F1, soporte y macro avg), todo a mano desde la matriz (sin sklearn).

Es tolerante al entorno: cada framework vive en su propio venv, así que corré este script
en el venv que tengas a mano y reporta solo ese lado.

    # en el venv de TensorFlow
    python documentacion/reporte_metricas.py
    # en el venv de PyTorch
    python documentacion/reporte_metricas.py

Si el modelo v4 no está en disco (.keras/.pt están gitignoreados), avisa que corras
primero el 04_scripts.py correspondiente.
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
IMG_SIZE = (224, 224)
SEED = 123
BATCH_SIZE = 32
CLASS_NAMES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]


# --- Reporte por clase (a mano desde la matriz; duplicado igual que en los v5) ---
def reporte_por_clase(cm: np.ndarray, nombres: list[str]) -> None:
    total = cm.sum()
    print(f"{'clase':<16}{'precision':>10}{'recall':>9}{'f1':>7}{'soporte':>9}")
    print("-" * 51)
    precs, recs, f1s = [], [], []
    for i in range(len(nombres)):
        soporte = cm[i, :].sum()
        predichos = cm[:, i].sum()
        recall = cm[i, i] / soporte if soporte else 0.0
        precision = cm[i, i] / predichos if predichos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
        print(f"{nombres[i]:<16}{precision:>10.3f}{recall:>9.3f}{f1:>7.3f}{int(soporte):>9}")
    print("-" * 51)
    print(f"{'macro avg':<16}{np.mean(precs):>10.3f}{np.mean(recs):>9.3f}{np.mean(f1s):>7.3f}{int(total):>9}")
    acc = np.trace(cm) / total if total else 0.0
    print(f"accuracy: {acc:.3f}  ({int(np.trace(cm))}/{int(total)})")


def lado_tensorflow() -> None:
    try:
        import tensorflow as tf
    except Exception:
        print("[TensorFlow] no disponible en este venv; salteo el lado TF.")
        return

    model_path = ROOT / "TensorFlow" / "v4" / "modelotf_v4_mobilenetv3.keras"
    if not model_path.exists():
        print(f"[TensorFlow] falta el modelo {model_path.name}.")
        print("             Corré primero:  python TensorFlow/v4/04_scripts.py")
        return

    print("\n" + "#" * 60)
    print("# TensorFlow v4 (MobileNetV3-Small)")
    print("#" * 60)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR), validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    class_names = val_ds.class_names
    print("Orden de clases:", class_names)
    model = tf.keras.models.load_model(model_path)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy()
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    print("\nReporte por clase:")
    reporte_por_clase(cm, class_names)


def lado_pytorch() -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset, random_split
        from torchvision import datasets, transforms
        from torchvision.models import mobilenet_v3_small
    except Exception:
        print("[PyTorch] no disponible en este venv; salteo el lado PT.")
        return

    model_path = ROOT / "PyTorch" / "v4" / "modelopt_v4_mobilenetv3.pt"
    if not model_path.exists():
        print(f"[PyTorch] falta el modelo {model_path.name}.")
        print("          Corré primero:  python PyTorch/v4/04_scripts.py")
        return

    print("\n" + "#" * 60)
    print("# PyTorch v4 (MobileNetV3-Small)")
    print("#" * 60)
    ckpt = torch.load(model_path, map_location="cpu")
    class_names = ckpt.get("class_names", CLASS_NAMES)
    mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    torch.manual_seed(SEED)
    base = datasets.ImageFolder(str(DATA_DIR))
    print("Orden de clases:", base.classes)
    n_val = max(1, int(len(base) * 0.2))
    n_train = len(base) - n_val
    _, idx_val = random_split(
        range(len(base)), [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    ds_val = Subset(datasets.ImageFolder(str(DATA_DIR), transform=tfm), list(idx_val))
    val_dl = DataLoader(ds_val, batch_size=BATCH_SIZE)

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for x, y in val_dl:
            pred = model(x).argmax(1).numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t, p] += 1
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    print("\nReporte por clase:")
    reporte_por_clase(cm, class_names)


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"No existe {DATA_DIR}. Necesito el dataset para reconstruir el split de val.\n"
            "Opción de prueba:  python documentacion/crear_datos_prueba.py --por-clase 30"
        )
    lado_tensorflow()
    lado_pytorch()


if __name__ == "__main__":
    main()
