# Is-it-AI — Detector de huella de IA en diapositivas

## Propósito

A partir de la **foto de una diapositiva**, estimar cuánta **huella de generación
por IA** presenta, clasificándola en 3 niveles:

| Clase | Significado |
|-------|-------------|
| `0_sin_ia`      | Sin huella de IA visible; parece hecha por humano. |
| `1_rastro_ia`   | IA con clara intervención humana (datos reales, capturas, edición). |
| `2_saturada_ia` | Predominantemente generada por IA (plantilla, imágenes IA, texto genérico). |

## Objetivos

- Implementación **dual** en **TensorFlow** y **PyTorch**, para contrastar ambos
  frameworks sobre el mismo problema y mantener criterios parejos entre los dos.
- Progresión por versiones: `v1` baseline (CNN desde cero que sobreajusta a
  propósito) y `v2` con data augmentation para atacar ese sobreajuste.
- Modelo liviano orientado a **inferencia móvil (Android)**.

> Estado: avance inicial. Funciona de punta a punta; se irá puliendo por partes.
