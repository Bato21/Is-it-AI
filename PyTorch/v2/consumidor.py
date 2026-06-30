"""
Consumidor del modelo de diapositivas — PyTorch v2

Carga modelo_diapositivas.pt (checkpoint {model_state, config} generado por
entrenar.py) y predice la clase de una o varias imágenes. Independiente del
código de entrenamiento.

  python consumidor.py imagen.png
  python consumidor.py carpeta/
  python consumidor.py img1.png img2.png ...

La arquitectura se redefine aquí (igual que en entrenar.py) para poder hacer
load_state_dict; en un proyecto grande viviría en un model.py compartido.
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # consola UTF-8 en Windows
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

AQUI = Path(__file__).resolve().parent
MODELO = AQUI / "modelo_diapositivas.pt"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def construir_modelo(num_clases: int) -> nn.Module:
    net = mobilenet_v3_large(weights=None)
    in_f = net.classifier[0].in_features
    net.classifier = nn.Sequential(
        nn.Linear(in_f, 256), nn.Hardswish(),
        nn.Dropout(0.3), nn.Linear(256, num_clases),
    )
    return net


def cargar_modelo(ruta: Path):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe '{ruta}'. Entrena primero: python entrenar.py")
    ckpt = torch.load(ruta, map_location="cpu")
    cfg = ckpt["config"]
    net = construir_modelo(len(cfg["clases"]))
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, cfg


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python consumidor.py <imagen|carpeta> [...]")
    net, cfg = cargar_modelo(MODELO)
    clases = cfg["clases"]
    tf = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    rutas: list[Path] = []
    for a in sys.argv[1:]:
        p = Path(a)
        if p.is_dir():
            rutas += [f for f in sorted(p.iterdir()) if f.suffix.lower() in EXTS]
        elif p.suffix.lower() in EXTS:
            rutas.append(p)
    if not rutas:
        sys.exit("No se encontraron imágenes válidas.")

    for r in rutas:
        x = tf(Image.open(r).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(net(x), dim=1)[0].numpy()
        idx = int(probs.argmax())
        barra = "█" * int(probs[idx] * 20)
        print(f"\n  {r.name}")
        print(f"  -> {clases[idx]}  ({probs[idx]:.1%})  {barra}")
        for i, c in enumerate(clases):
            print(f"     {c:<14} {probs[i]:.1%}")


if __name__ == "__main__":
    main()
