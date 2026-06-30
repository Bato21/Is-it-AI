# Is-it-AI — Detector de huella de IA en diapositivas

Clasificador de aprendizaje profundo que puntúa **diapositivas** según su
**gradiente de huella de IA**, en 3 niveles:

| Clase | Significado |
|-------|-------------|
| `0_sin_ia`      | Sin huella de IA visible; parece hecha por humano. |
| `1_rastro_ia`   | IA con clara intervención humana (datos reales, capturas, edición). |
| `2_saturada_ia` | Predominantemente generada por IA (plantilla, imágenes IA, texto genérico). |

Construido **dos veces** — en **TensorFlow** y en **PyTorch** — para comparar
ambos frameworks (objetivo del curso) y con **MobileNetV3Large (transfer
learning)** para poder correr en **Android** más adelante.

## Estructura

```
Is-it-AI/
├── data/                       # dataset (gitignored). SOLO las 3 clases:
│   ├── 0_sin_ia/  1_rastro_ia/  2_saturada_ia/
├── raw_decks/                  # deja aquí las presentaciones .pdf / .pptx
├── unsorted/                   # el conversor deja aquí los PNG; tú los clasificas
├── tools/                      # conversor deck->imágenes y utilidades
│   ├── deck_a_imagenes.py · crear_datos_prueba.py · dividir_dataset.py
├── TensorFlow/
│   ├── v1/01_script.py         # baseline CNN (sobreajusta a propósito)
│   └── v2/                     # transfer MobileNetV3Large
│       ├── entrenar.py · consumidor.py · exportar_tflite.py · grad_cam.py
├── PyTorch/
│   ├── v1/01_script.py         # espejo del baseline en PyTorch
│   └── v2/
│       ├── entrenar.py · consumidor.py · exportar_movil.py
└── docs/                       # guía de etiquetado, setup, prompt presentación
```

Cada framework tiene **su propio entorno virtual** (TF y PyTorch chocan si se
mezclan). Ver `docs/SETUP.md`.

## Flujo de trabajo

```powershell
# 0. (una vez) crear los 3 venvs e instalar dependencias
./setup_envs.ps1

# 1. convertir presentaciones -> imágenes (venv tools)
python tools\deck_a_imagenes.py        # PDF/PPTX en raw_decks -> unsorted\

# 2. clasificar los PNG de unsorted\ en data\{0_sin_ia,1_rastro_ia,2_saturada_ia}
#    (rúbrica: docs\guia_etiquetado.md)

# 3. entrenar
python TensorFlow\v2\entrenar.py       # venv TensorFlow
python PyTorch\v2\entrenar.py          # venv PyTorch
#    -> cada uno genera: training_curves.png, confusion_matrix.png, roc_curves.png

# 4. predecir una diapositiva
python TensorFlow\v2\consumidor.py una_diapositiva.png
python PyTorch\v2\consumidor.py una_diapositiva.png

# 5. exportar para Android
python TensorFlow\v2\exportar_tflite.py
python PyTorch\v2\exportar_movil.py
```

### Probar el flujo sin datos reales
```powershell
python tools\crear_datos_prueba.py --por-clase 30   # imágenes sintéticas por clase
python PyTorch\v2\entrenar.py                        # corre todo el pipeline
python tools\crear_datos_prueba.py --limpiar         # borra las de prueba
```

## Versiones (pedagogía del curso)
- **v1** — CNN desde cero. Entrena y **sobreajusta a propósito**: ver esa brecha
  de generalización motiva la v2.
- **v2** — **Transfer learning** con MobileNetV3Large + augmentation + dropout +
  fine-tuning en dos fases. Métricas manuales (sin sklearn) y 3 figuras
  (curvas, matriz de confusión, ROC), igual que el material de clase.

## Stack
Python 3.11 · TensorFlow 2.15/2.16 · PyTorch 2.2 (CPU) · MobileNetV3Large.
Objetivo móvil: Android (TFLite + TorchScript-lite).
