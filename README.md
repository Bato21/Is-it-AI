# Is-it-AI — Detector de huella de IA en diapositivas

## Propósito

A partir de la **foto de una diapositiva**, estimar cuánta **huella de generación
por IA** presenta, clasificándola en 3 niveles (eje ordinal de saturación) — y **reconocer
cuándo la foto no es una diapositiva**, para no responder en ese caso:

| Clase | Rol | Significado |
|-------|-----|-------------|
| `0_sin_ia`         | eje ordinal | Sin huella de IA visible; parece hecha por humano. |
| `1_rastro_ia`      | eje ordinal | IA con clara intervención humana (datos reales, capturas, edición). |
| `2_saturada_ia`    | eje ordinal | Predominantemente generada por IA (plantilla, imágenes IA, texto genérico). |
| `3_no_diapositiva` | **compuerta** | La foto no muestra una diapositiva. No corresponde estimar nada. |

> La cuarta clase (v10) **no es un cuarto nivel de saturación**: está fuera del eje. Un
> softmax de 3 salidas siempre reparte 1.0 entre sus 3 opciones, así que hasta la v9 el
> modelo no podía abstenerse: ante la foto de un escritorio devolvía un nivel de IA, y con
> confianza alta. Ver `documentacion/analisis_v10.md`.

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
  → `v9` producción: brecha de dominio, fine-tuning en dos fases y despliegue
  → **`v10` compuerta de rechazo**: una cuarta clase para poder abstenerse.
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

### El dataset: fotos de pantalla + renders

Desde la v9 el dataset mezcla dos dominios, y esa distinción manda en todo lo demás:

| clase | total | fotos de pantalla | renders |
|---|---:|---:|---:|
| `0_sin_ia` | 434 | 326 | 108 |
| `1_rastro_ia` | 435 | 341 | 94 |
| `2_saturada_ia` | 468 | 370 | 98 |
| `3_no_diapositiva` *(v10)* | 449 | 300 | 149 |
| **total** | **1786** | **1337** | **449** |

La app consume **fotos**, no renders. Por eso `documentacion/particion_v10.py` impone una
regla: **test y validación son 100% fotos; los renders solo entrenan**. El número que se
reporta es la *accuracy comercial*, no la de laboratorio. Ver `documentacion/analisis_v9.md`
y `documentacion/analisis_v10.md`.

```bash
python documentacion/negativos_v10.py            # baja la clase de rechazo (449 imgs)
python documentacion/particion_v10.py            # genera particion_v10.json
python documentacion/particion_v10.py --mostrar  # inventario por clase y dominio
```

**La clase de rechazo** se obtiene de datasets públicos, estratificada en cuatro familias
para cubrir las formas reales en que el usuario apunta la cámara a algo que no es una
diapositiva:

| familia | n | dominio | fuente | qué cubre |
|---|---:|---|---|---|
| escena | 160 | foto | SUN397 | habitaciones, oficinas, aulas: el encuadre accidental |
| objeto | 140 | foto | COCO | personas, objetos, escritorios |
| interfaz | 100 | render | wave-ui + website-screenshots | **negativo difícil**: una pantalla que no es una diapositiva |
| documento | 49 | render | DocVQA + FUNSD | **negativo difícil**: texto sobre fondo blanco |

Las 300 **fotos** pueden entrar al test (son fotografías de cámara reales, el mismo dominio
que la app ve); los 149 **renders** solo entrenan, y la augmentation los convierte en
"captura de pantalla fotografiada". La procedencia de cada archivo queda en
`dataset/3_no_diapositiva/_procedencia.json` y `documentacion/dominio.py` la respeta por
encima de sus heurísticos.

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
| `documentacion/negativos_v10.py`      | **v10**: obtención de la clase de rechazo desde datasets públicos + manifiesto de procedencia. |
| `documentacion/dominio.py`            | **v9**: decide si cada imagen es foto de pantalla o render (procedencia declarada + EXIF + nombre). |
| `documentacion/particion_v9.py`       | **v9**: partición consciente del dominio (test y validación 100% fotos). |
| `documentacion/particion_v10.py`      | **v10**: la misma partición con 4 clases + guard de que la compuerta tenga fotos. |
| `documentacion/aug_pantalla.py`       | **v9**: simulación de foto de pantalla (moiré, glare, perspectiva, JPEG). |
| `documentacion/imagenes.py`           | **v9**: caché compartido de imágenes decodificadas; las vistas son idénticas en los 3 modelos. |
| `documentacion/metricas.py`           | Métricas compartidas. Desde la v10 separa las ordinales (eje 0-1-2) de las de compuerta. |
| `TensorFlow/vN/`, `PyTorch/vN/`       | **Entrenamiento** espejo (v1→v10). Guardan modelo, curvas y matriz de confusión. |
| `PyTorch/v9/modelos_v9.py`            | Fuente única de las arquitecturas PyTorch (la v10 la reexporta: no cambió la red). |
| `export/exportar_onnx.py`             | PyTorch → ONNX, con verificación de paridad. |
| `export/generar_catalogo.py`          | Vuelca las métricas reales al catálogo que lee la app. |
| `export/verificar_paridad.py`         | Compara el `.keras` contra el modelo TensorFlow.js ya convertido. |
| `TensorFlow/export_tfjs/`             | Keras → TensorFlow.js. |
| `app/`                                | App Ionic 8 + Angular 18 (ver `app/README.md`). |
| `modelos/vN/`                         | **Consumo** (inferencia) independiente del entrenamiento. |
| `modelos/v10/consumir_v10.py`         | Consumo de los 3 modelos v10 + modo cámara + comparación entre modelos. |
| `imagenes_a_probar/`                  | Imágenes sueltas para probar el consumo. |

## La v10: tres modelos, cuatro clases y una app

| | Modelo A | Modelo B | Modelo C |
|---|---|---|---|
| framework | TensorFlow / Keras | PyTorch | PyTorch |
| arquitectura | MobileNetV3-Small | MobileNetV3-Small | EfficientNet-B0 |
| entrenamiento | balanceado, fine-tuning 2 fases | ídem A (espejo) | **desequilibrado + Focal Loss** |
| formato desplegado | TensorFlow.js | ONNX | ONNX |

Métricas sobre el **test de fotos** (267 fotos de cámara, ningún render), media de semillas:

| | Modelo A | Modelo B | Modelo C |
|---|---:|---:|---:|
| accuracy | **0.9558** | 0.9393 | 0.8989 |
| macro F1 | **0.9567** | 0.9407 | 0.9024 |
| recall clase 2 | **0.9730** | 0.9432 | 0.8333 |
| **recall de rechazo** | **0.9967** | 0.9700 | 0.9778 |
| **fugas** (de 60 no-diapositivas) | **0.2** | 1.8 | 1.3 |
| rechazos indebidos (de 207 diapos) | 0.2 | **0.0** | **0.0** |
| peso | 4.1 MB | 4.6 MB | 16.0 MB |

> **A es agresiva, B y C son conservadores.** B y C nunca descartan una diapositiva real y lo
> pagan fugando más; A ataja casi todo y descarta una cada tanto. Para el producto la fuga es
> el error caro —un veredicto inventado sobre el escritorio de alguien— así que **A es el
> default**. El selector deja de ser cosmético: es elegir entre *"no me molestes con rechazos"*
> y *"no me inventes respuestas"*.

Los tres se cargan en la app y se eligen desde un selector. **A** es el default (el más rápido,
único sobre WebGL); **B** verifica en producción la paridad entre frameworks y sirve de respaldo
sobre WASM; **C** aporta un voto de otra familia de arquitectura y es el "modelo desequilibrado"
del enunciado. El botón *Comparar los 3* corre los tres sobre la misma foto.

Los tres tienen la **compuerta de rechazo**: si la foto no es una diapositiva, la app lo dice
en vez de inventar un nivel de huella de IA. Es lo que agrega la v10 — ver
**`documentacion/analisis_v10.md`**, que además justifica con una ablación *cuántas* imágenes
de "no diapositiva" hacen falta.

Análisis de la versión anterior (brecha de dominio, fine-tuning en dos fases):
**`documentacion/analisis_v9.md`**.

## Pipeline v10 de punta a punta

```bash
# 0) Obtener la clase de rechazo (una vez; se reanuda si se corta)
python documentacion/negativos_v10.py

# 1) Partición consciente del dominio (una vez, o cuando cambie el dataset)
python documentacion/particion_v10.py

# 2) Entrenar los tres modelos
TensorFlow/.venv/Scripts/python TensorFlow/v10/10_scripts.py                   # A  (~15 min CPU)
PyTorch/.venv/Scripts/python    PyTorch/v10/10_scripts.py                      # B  (~10 min CPU)
PyTorch/.venv/Scripts/python    PyTorch/v10/10_modelo_c_desequilibrado.py      # C  (~60 min CPU)

# 3) Exportar a los formatos que consume la app
TensorFlow/export_tfjs/.venv/Scripts/python TensorFlow/export_tfjs/exportar_tfjs.py   # A -> TF.js
PyTorch/.venv/Scripts/python export/exportar_onnx.py                                  # B y C -> ONNX
python export/generar_catalogo.py                                                     # métricas -> app

# 3b) Verificar que lo desplegado diga lo mismo que lo entrenado
TensorFlow/.venv/Scripts/python export/verificar_paridad.py    # .keras vs TensorFlow.js

# 4) Levantar la app
cd app && npm install && npm start        # http://localhost:8100
```

**Desplegada en https://app-eta-one-19.vercel.app** (producción, sirviendo la v10). La
cámara necesita HTTPS, así que la red local no alcanza; el detalle del despliegue y su
verificación están en **`documentacion/DESPLIEGUE.md`**.

> `export/exportar_onnx.py` **verifica** la exportación comparando PyTorch contra ONNX Runtime
> sobre fotos reales del test, y aborta si difieren más de 1e-4. Para el modelo A, que va por
> TensorFlow.js, la verificación equivalente la hace `export/verificar_paridad.py` levantando
> el modelo en Node (ver `export/paridad_v10.txt`).

## Consumo independiente del modelo

```bash
# clasifica una carpeta con el modelo A
TensorFlow/.venv/Scripts/python modelos/v10/consumir_v10.py --modelo a --ruta imagenes_a_probar

# corre los TRES y reporta en qué imágenes discrepan
PyTorch/.venv/Scripts/python modelos/v10/consumir_v10.py --modelo todos --ruta imagenes_a_probar

# cámara en vivo con OpenCV (requiere opencv-python)
PyTorch/.venv/Scripts/python modelos/v10/consumir_v10.py --modelo b --camara
```

> **La prueba más corta de para qué sirve la v10:** pasarle una foto cualquiera que NO sea una
> diapositiva. El modelo de la v9 no puede abstenerse y contesta con confianza alta; el de la
> v10 responde `NO ES UNA DIAPOSITIVA` y no informa nivel.

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

> **Estado:** v10 completa. Los tres modelos entrenan con las 4 clases, se exportan
> verificados y corren dentro de la app Ionic, que ya distingue "no es una diapositiva" de un
> nivel de huella de IA. Limitaciones conocidas y qué falta:
> `documentacion/analisis_v10.md` §7 y `documentacion/analisis_v9.md` §4.
