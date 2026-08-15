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
  → `v3` early stopping / convergencia → `v4` transfer learning con **MobileNetV3-Small**
  → `v5` protocolo → `v6` Optuna → `v7` evaluación honesta → `v8` destilación
  → **`v9` producción**: brecha de dominio, fine-tuning en dos fases y despliegue.
- **Tres modelos desplegados** en una app **Ionic + Angular**, con inferencia en el
  dispositivo mediante **TensorFlow.js**, **ONNX Runtime Web** y **OpenCV.js**.

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
de git** (ver `.gitignore`).

### El dataset v9: fotos de pantalla + renders

A partir de la v9 el dataset mezcla dos dominios, y esa distinción manda en todo lo demás:

| clase | total | fotos de pantalla | renders |
|---|---:|---:|---:|
| `0_sin_ia` | 434 | 326 | 108 |
| `1_rastro_ia` | 435 | 341 | 94 |
| `2_saturada_ia` | 468 | 370 | 98 |
| **total** | **1337** | **1037** | **300** |

La app consume **fotos**, no renders. Por eso `documentacion/particion_v9.py` impone una
regla: **test y validación son 100% fotos; los renders solo entrenan**. El número que se
reporta es la *accuracy comercial*, no la de laboratorio. Ver `documentacion/analisis_v9.md`.

```bash
python documentacion/particion_v9.py            # genera particion_v9.json
python documentacion/particion_v9.py --mostrar  # inventario por clase y dominio
```

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
| `documentacion/dominio.py`            | **v9**: decide si cada imagen es foto de pantalla o render (EXIF + nombre). |
| `documentacion/particion_v9.py`       | **v9**: partición consciente del dominio (test y validación 100% fotos). |
| `documentacion/aug_pantalla.py`       | **v9**: simulación de foto de pantalla (moiré, glare, perspectiva, JPEG). |
| `documentacion/imagenes.py`           | **v9**: caché compartido de imágenes decodificadas; las vistas son idénticas en los 3 modelos. |
| `TensorFlow/vN/`, `PyTorch/vN/`       | **Entrenamiento** espejo (v1→v9). Guardan modelo, curvas y matriz de confusión. |
| `PyTorch/v9/modelos_v9.py`            | Fuente única de las arquitecturas PyTorch (la comparten entrenamiento y exportación). |
| `export/exportar_onnx.py`             | **v9**: PyTorch → ONNX, con verificación de paridad. |
| `export/generar_catalogo.py`          | **v9**: vuelca las métricas reales al catálogo que lee la app. |
| `export/verificar_paridad.py`         | **v9**: compara el `.keras` contra el modelo TensorFlow.js ya convertido. |
| `TensorFlow/export_tfjs/`             | **v9**: Keras → TensorFlow.js. |
| `app/`                                | **v9**: app Ionic 8 + Angular 18 (ver `app/README.md`). |
| `modelos/vN/`                         | **Consumo** (inferencia) independiente del entrenamiento. |
| `modelos/v9/consumir_v9.py`           | Consumo de los 3 modelos v9 + modo cámara + comparación entre modelos. |
| `imagenes_a_probar/`                  | Imágenes sueltas para probar el consumo. |

## La v9: tres modelos y una app

| | Modelo A | Modelo B | Modelo C |
|---|---|---|---|
| framework | TensorFlow / Keras | PyTorch | PyTorch |
| arquitectura | MobileNetV3-Small | MobileNetV3-Small | EfficientNet-B0 |
| entrenamiento | balanceado, fine-tuning 2 fases | ídem A (espejo) | **desequilibrado 9:1 + Focal Loss** |
| formato desplegado | TensorFlow.js | ONNX | ONNX |
| accuracy sobre fotos | **0.9478** | 0.9372 | 0.8647 |
| recall clase 2 | **0.9730** | 0.9378 | 0.8198 |
| peso | 4.1 MB | 4.6 MB | 16.0 MB |

Los tres se cargan en la app y se eligen desde un selector. **A** es el default (el más rápido,
único sobre WebGL); **B** verifica en producción la paridad entre frameworks y sirve de respaldo
sobre WASM; **C** aporta un voto de otra familia de arquitectura y es el "modelo desequilibrado"
del enunciado. El botón *Comparar los 3* corre los tres sobre la misma foto.

Análisis completo de métricas y plan de acción: **`documentacion/analisis_v9.md`**.

## Pipeline v9 de punta a punta

```bash
# 1) Partición consciente del dominio (una vez, o cuando cambie el dataset)
python documentacion/particion_v9.py

# 2) Entrenar los tres modelos
TensorFlow/.venv/Scripts/python TensorFlow/v9/09_scripts.py                    # A  (~7 min CPU)
PyTorch/.venv/Scripts/python    PyTorch/v9/09_scripts.py                       # B  (~7 min CPU)
PyTorch/.venv/Scripts/python    PyTorch/v9/09_modelo_c_desequilibrado.py       # C  (~50 min CPU)

# 3) Exportar a los formatos que consume la app
TensorFlow/export_tfjs/.venv/Scripts/python TensorFlow/export_tfjs/exportar_tfjs.py   # A -> TF.js
PyTorch/.venv/Scripts/python export/exportar_onnx.py                                  # B y C -> ONNX
python export/generar_catalogo.py                                                     # métricas -> app

# 3b) Verificar que lo desplegado diga lo mismo que lo entrenado
TensorFlow/.venv/Scripts/python export/verificar_paridad.py    # .keras vs TensorFlow.js

# 4) Levantar la app
cd app && npm install && npm start        # http://localhost:8100
```

> `export/exportar_onnx.py` **verifica** la exportación comparando PyTorch contra ONNX Runtime
> sobre fotos reales del test, y aborta si difieren más de 1e-4. Resultado actual:
> 3.15e-05 (B) y 6.56e-06 (C), con 12/12 clases coincidentes. Para el modelo A, que va por
> TensorFlow.js, la verificación equivalente la hace `export/verificar_paridad.py` levantando
> el modelo en Node: **1.55e-06** (ver `export/paridad_v9.txt`).

## Consumo independiente del modelo

```bash
# clasifica una carpeta con el modelo A
TensorFlow/.venv/Scripts/python modelos/v9/consumir_v9.py --modelo a --ruta imagenes_a_probar

# corre los TRES y reporta en qué imágenes discrepan
TensorFlow/.venv/Scripts/python modelos/v9/consumir_v9.py --modelo todos --ruta imagenes_a_probar

# cámara en vivo con OpenCV (requiere opencv-python)
PyTorch/.venv/Scripts/python modelos/v9/consumir_v9.py --modelo b --camara
```

No entrena, no toca el dataset y no importa nada de los scripts de entrenamiento. Además es la
**referencia** contra la que se verifica la app: si el navegador y este script no coinciden, el
problema está en el preprocesamiento del cliente.

## Cómo correr las versiones anteriores

Entornos separados por framework (`TensorFlow/.venv` y `PyTorch/.venv`).
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

> **Estado:** v9 completa. Los tres modelos entrenan, se exportan verificados y corren dentro
> de la app Ionic. Lo que falta está listado como plan de acción en
> `documentacion/analisis_v9.md` §4 — con la prioridad y el criterio de medición de cada punto.
