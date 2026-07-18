# Is-it-AI — Detector de huella de IA en diapositivas

## Propósito

A partir de la **foto de una diapositiva**, estimar cuánta **huella de generación
por IA** presenta, clasificándola en 3 niveles (eje ordinal de saturación):

| Clase | Significado |
|-------|-------------|
| `0_sin_ia`      | Sin huella de IA visible; parece hecha por humano. |
| `1_rastro_ia`   | IA con clara intervención humana (datos reales, capturas, edición). |
| `2_saturada_ia` | Predominantemente generada por IA (plantilla, imágenes IA, texto genérico). |

> El proyecto **no mide procedencia ni proceso**, sino **densidad de artefactos
> visuales de IA** observables en la imagen — que es lo único que una CNN puede
> aprender de una foto. Ver `documentacion/avances_proyecto.md`.

## Objetivos

- Implementación **dual** en **TensorFlow/Keras** y **PyTorch** (scripts espejo, con
  comentarios de *contraste de frameworks* donde los dos difieren).
- Progresión por versiones, un cambio conceptual por versión:
  `v1` baseline (CNN desde cero que sobreajusta a propósito) → `v2` data augmentation
  → `v3` early stopping / convergencia → `v4` transfer learning con **MobileNetV3-Small**.
- Modelo liviano orientado a **inferencia en el navegador** (TensorFlow.js) dentro de
  una app **Ionic** (reemplaza el objetivo previo de TensorFlow Lite / Android).

## Pipeline de datos

La **obtención de datos está separada del entrenamiento**. El flujo, de punta a punta:

```
presentaciones (.pdf / .pptx)
   └─ deck_a_imagenes.py ─────────► 1 PNG por diapositiva  (documentacion/sin_clasificar/)
        └─ clasificación manual ──► dataset/{0_sin_ia, 1_rastro_ia, 2_saturada_ia}/
             └─ verificar_dataset.py (chequeo) ──► entrenamiento (TensorFlow/ · PyTorch/)
```

**Ningún script de entrenamiento descarga, genera ni modifica datos**: los scripts de
`TensorFlow/` y `PyTorch/` solo **leen** `dataset/`. Esa es la separación obtención ↔
entrenamiento que pide el punto 2 del anuncio de la 2ª entrega. El dataset vive **fuera
de git** (ver `.gitignore`); son ~300 imágenes reales, ~100 por clase.

Preparar y verificar el dataset:

```bash
python documentacion/deck_a_imagenes.py        # PDF/PPTX -> PNG por diapositiva
#   (clasificás las PNG a mano en dataset/0_sin_ia, 1_rastro_ia, 2_saturada_ia)
python documentacion/verificar_dataset.py      # tabla + avisos; corta si falta una clase

# ¿todavía sin datos reales? Probá el flujo completo con datos sintéticos:
python documentacion/crear_datos_prueba.py --por-clase 30   # crear
python documentacion/crear_datos_prueba.py --limpiar        # borrar al terminar
```

> **Ruta canónica del dataset:** `dataset/` en la raíz del repo. Todos los scripts de
> entrenamiento la resuelven igual (`parents[2] / "dataset"`). Si tu carpeta local se
> llama `data/`, renombrala una vez a `dataset/` (o dejá un symlink `dataset -> data`).

## Scripts (qué hace cada carpeta)

| Ruta | Rol |
|------|-----|
| `documentacion/deck_a_imagenes.py`    | **Obtención**: PDF/PPTX → 1 PNG por diapositiva. |
| `documentacion/verificar_dataset.py`  | **Chequeo** del dataset antes de entrenar (tabla, avisos, exit≠0 si falta una clase). |
| `documentacion/crear_datos_prueba.py` | Datos **sintéticos** para probar el flujo sin dataset real. |
| `TensorFlow/vN/`, `PyTorch/vN/`       | **Entrenamiento** espejo (v1→v4). Guardan modelo, curvas y matriz de confusión. |
| `modelos/vN/`                         | **Consumo** (inferencia) independiente del entrenamiento. |
| `modelos/v4/camara_tf.py`, `camara_pt.py` | Inferencia **en vivo por webcam** (OpenCV). |
| `imagenes_a_probar/`                  | Imágenes sueltas para probar el consumo. |

## Cómo correr

Entornos conda separados por framework: `tensorflow-ia` (TF) y `frameworks-ia` (PyTorch).
Ver `documentacion/SETUP.md`.

```bash
# Entrenamiento (ejemplo v4, transfer learning MobileNetV3-Small)
python TensorFlow/v4/04_scripts.py
python PyTorch/v4/04_scripts.py

# Consumo: clasifica las imágenes de imagenes_a_probar/ (vector completo de probabilidades)
python modelos/v4/modelopt_v4.py        # PyTorch
python modelos/v4/modelotf_v4.py        # TensorFlow

# Cámara en vivo (webcam apuntando a una diapo en pantalla)
python modelos/v4/camara_tf.py          # TensorFlow
python modelos/v4/camara_pt.py          # PyTorch
```

> Estado: entrega 2 en curso. Funciona de punta a punta; se pule por partes.
