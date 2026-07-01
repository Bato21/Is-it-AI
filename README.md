# Is-it-AI — Detector de huella de IA en diapositivas

Clasifica una **diapositiva** según su huella de IA en 3 niveles:

| Clase | Significado |
|-------|-------------|
| `0_sin_ia`      | Sin huella de IA visible; parece hecha por humano. |
| `1_rastro_ia`   | IA con clara intervención humana (datos reales, capturas, edición). |
| `2_saturada_ia` | Predominantemente generada por IA (plantilla, imágenes IA, texto genérico). |

Construido en **TensorFlow** y en **PyTorch** (para comparar) con
**MobileNetV3Large (transfer learning)**, pensado para correr en Android.

> Estado: avance inicial. Funciona de punta a punta; se irá puliendo por partes.

## Estructura

```
TensorFlow/    # lado TF, autocontenido: v1 baseline + v2 (config·modelo·datos·metricas·
               #   entrenar·consumidor·exportar_tflite·grad_cam) + requirements.txt
PyTorch/       # lado PyTorch, mismo esquema (v1 + v2 + requirements.txt)
dataset/       # imágenes etiquetadas en 0_sin_ia / 1_rastro_ia / 2_saturada_ia (gitignored)
documentacion/ # setup, rúbrica de etiquetado y prompt de la presentación
herramientas/  # conversor de presentaciones, utilidades de dataset y sus carpetas de trabajo
```

Cada framework es independiente y trae su propio `requirements.txt`. `v1/` es el
baseline (CNN desde cero que sobreajusta a propósito); `v2/` es transfer learning.

## Entornos

Cada framework usa su propio venv (TF y PyTorch fijan versiones en conflicto de
numpy/protobuf). Base: **Python 3.11**.

```powershell
./setup_envs.ps1     # crea los 3 venvs e instala cada requirements.txt
```

Detalle en `documentacion/SETUP.md`. Versiones exactas probadas en cada `requirements.txt`.

## Uso

```powershell
# 1. (opcional) probar el flujo sin datos reales
python herramientas/crear_datos_prueba.py --por-clase 30

# 2. entrenar (en el venv del framework)
python PyTorch/v2/entrenar.py        # o TensorFlow/v2/entrenar.py
#    genera el modelo + training_curves / confusion_matrix / roc_curves .png

# 3. predecir
python PyTorch/v2/consumidor.py una_diapositiva.png

# 4. exportar para Android
python PyTorch/v2/exportar_movil.py     # PyTorch -> .ptl
python TensorFlow/v2/exportar_tflite.py # TensorFlow -> .tflite

# 5. limpiar los datos de prueba
python herramientas/crear_datos_prueba.py --limpiar
```

Convertir presentaciones reales a imágenes: `python herramientas/deck_a_imagenes.py`
(deja PNGs en `herramientas/sin_clasificar/`; se clasifican a mano en
`dataset/<clase>/`, rúbrica en `documentacion/guia_etiquetado.md`).

El nivel de detalle de los logs se controla con `IS_IT_AI_LOG` (p. ej. `DEBUG`).
