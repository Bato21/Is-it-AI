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

## Estructura del repositorio

```
Is-it-AI/
├── README.md
├── setup_envs.ps1                  # crea los 3 entornos virtuales
├── requerimientos/                 # dependencias (una por framework)
│   ├── tensorflow.txt
│   ├── pytorch.txt
│   └── herramientas.txt
├── dataset/                        # imágenes etiquetadas (gitignored). SOLO 3 clases:
│   ├── 0_sin_ia/
│   ├── 1_rastro_ia/
│   └── 2_saturada_ia/
├── presentaciones_fuente/          # aquí dejas los .pdf / .pptx de origen
├── sin_clasificar/                 # el conversor deja aquí los PNG; tú los clasificas
├── herramientas/                   # conversor y utilidades de dataset
│   ├── deck_a_imagenes.py          # PDF/PPTX -> 1 PNG por diapositiva
│   ├── crear_datos_prueba.py       # imágenes sintéticas para probar el flujo
│   └── dividir_dataset.py          # reporte de balance de clases
├── TensorFlow/
│   ├── v1/01_script.py             # baseline CNN (sobreajusta a propósito)
│   └── v2/                         # transfer learning MobileNetV3Large
│       ├── entrenar.py · consumidor.py · exportar_tflite.py · grad_cam.py
├── PyTorch/
│   ├── v1/01_script.py             # espejo del baseline en PyTorch
│   └── v2/
│       ├── entrenar.py · consumidor.py · exportar_movil.py
└── documentacion/
    ├── SETUP.md                    # instalación detallada de los entornos
    ├── guia_etiquetado.md          # rúbrica de las 3 clases
    └── prompt_faces_presentacion.md# prompt para armar la presentación
```

## Requerimientos

Cada framework usa **su propio entorno virtual** (TensorFlow y PyTorch fijan
versiones en conflicto de `numpy`/`protobuf` y no pueden convivir). Base común:
**Python 3.11**. Los archivos están en `requerimientos/`.

**TensorFlow** (`requerimientos/tensorflow.txt`)
```
tensorflow>=2.15,<2.17
numpy<2.0
matplotlib>=3.8
pillow>=10.0
```

**PyTorch** (`requerimientos/pytorch.txt`)
```
torch==2.2.2
torchvision==0.17.2
numpy<2.0
matplotlib>=3.8
pillow>=10.0
```

**Herramientas / conversor** (`requerimientos/herramientas.txt`)
```
PyMuPDF>=1.24
pillow>=10.0
pywin32>=306 ; Windows (para convertir PPTX con PowerPoint)
```

Instalación de los tres entornos en un solo paso:
```powershell
./setup_envs.ps1
```
Detalle y activación de cada venv: `documentacion/SETUP.md`.

## Flujo de trabajo

```powershell
# 0. (una vez) crear los 3 venvs e instalar dependencias
./setup_envs.ps1

# 1. convertir presentaciones -> imágenes (venv herramientas)
python herramientas\deck_a_imagenes.py     # PDF/PPTX de presentaciones_fuente -> sin_clasificar\

# 2. clasificar los PNG de sin_clasificar\ en dataset\{0_sin_ia,1_rastro_ia,2_saturada_ia}
#    (rúbrica: documentacion\guia_etiquetado.md)

# 3. entrenar
python TensorFlow\v2\entrenar.py           # venv TensorFlow
python PyTorch\v2\entrenar.py              # venv PyTorch
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
python herramientas\crear_datos_prueba.py --por-clase 30   # imágenes sintéticas
python PyTorch\v2\entrenar.py                               # corre todo el pipeline
python herramientas\crear_datos_prueba.py --limpiar        # borra las de prueba
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
