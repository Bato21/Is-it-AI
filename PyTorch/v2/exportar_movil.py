"""
Exporta el modelo entrenado para Android — PyTorch v2

  python exportar_movil.py

Genera, en esta carpeta:
  modelo_diapositivas.torchscript.pt  — TorchScript (genérico)
  modelo_diapositivas.ptl             — bundle Lite para PyTorch Mobile

En Android: org.pytorch:pytorch_android_lite.
Entrada: 1x3x224x224, normalizada con mean/std de ImageNet (ver config del checkpoint).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models import mobilenet_v3_large

AQUI = Path(__file__).resolve().parent
MODELO = AQUI / "modelo_diapositivas.pt"


def construir_modelo(num_clases: int) -> nn.Module:
    net = mobilenet_v3_large(weights=None)
    in_f = net.classifier[0].in_features
    net.classifier = nn.Sequential(
        nn.Linear(in_f, 256), nn.Hardswish(),
        nn.Dropout(0.3), nn.Linear(256, num_clases),
    )
    return net


def main() -> None:
    if not MODELO.exists():
        sys.exit(f"No existe {MODELO}. Entrena primero: python entrenar.py")
    ckpt = torch.load(MODELO, map_location="cpu")
    cfg = ckpt["config"]
    net = construir_modelo(len(cfg["clases"]))
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    ejemplo = torch.randn(1, 3, cfg["img_size"], cfg["img_size"])
    traced = torch.jit.trace(net, ejemplo)

    ts = AQUI / "modelo_diapositivas.torchscript.pt"
    traced.save(str(ts))
    print(f"Guardado TorchScript: {ts}")

    opt = optimize_for_mobile(traced)
    ptl = AQUI / "modelo_diapositivas.ptl"
    opt._save_for_lite_interpreter(str(ptl))
    print(f"Guardado Lite: {ptl}")
    print("Android: org.pytorch:pytorch_android_lite . Entrada 1x3x224x224, normalizar con mean/std ImageNet.")


if __name__ == "__main__":
    main()
