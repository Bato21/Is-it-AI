"""
Consumidor del modelo de diapositivas — TensorFlow v2

Carga modelo_diapositivas.keras (generado por entrenar.py) y predice la clase
de una o varias imágenes de diapositiva. Independiente del código de
entrenamiento: solo necesita el archivo .keras.

  python consumidor.py imagen.png
  python consumidor.py carpeta_con_imagenes/
  python consumidor.py img1.png img2.png ...
"""
from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # consola UTF-8 en Windows
except Exception:
    pass

import numpy as np
import tensorflow as tf

AQUI = Path(__file__).resolve().parent
MODELO = AQUI / "modelo_diapositivas.keras"
IMG_SIZE = (224, 224)
CLASES = ["0_sin_ia", "1_rastro_ia", "2_saturada_ia"]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def cargar_imagen(ruta: Path) -> np.ndarray:
    img = tf.keras.utils.load_img(ruta, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)        # 0-255; el preproc va dentro del modelo
    return np.expand_dims(arr, 0)


def imprimir(ruta: Path, probs: np.ndarray) -> None:
    idx = int(np.argmax(probs))
    barra = "█" * int(probs[idx] * 20)
    print(f"\n  {ruta.name}")
    print(f"  -> {CLASES[idx]}  ({probs[idx]:.1%})  {barra}")
    for i, c in enumerate(CLASES):
        print(f"     {c:<14} {probs[i]:.1%}")


def expandir(args: list[str]) -> list[Path]:
    rutas: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            rutas += [f for f in sorted(p.iterdir()) if f.suffix.lower() in EXTS]
        elif p.suffix.lower() in EXTS:
            rutas.append(p)
    return rutas


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python consumidor.py <imagen|carpeta> [...]")
    if not MODELO.exists():
        sys.exit(f"No existe {MODELO}. Entrena primero: python entrenar.py")

    rutas = expandir(sys.argv[1:])
    if not rutas:
        sys.exit("No se encontraron imágenes válidas.")

    modelo = tf.keras.models.load_model(MODELO)
    for r in rutas:
        probs = modelo.predict(cargar_imagen(r), verbose=0)[0]
        imprimir(r, probs)


if __name__ == "__main__":
    main()
