# Plan de la entrega final — v9 + app Ionic

*Escrito al cierre de la v8. Define el trabajo que falta: una versión de modelado nueva
(v9, robustez a foto real) y la aplicación móvil en Ionic con OpenCV.js y TensorFlow.js.
Formato: un commit por tarea, igual que las entregas anteriores.*

---

## 0. Las tres decisiones tomadas

1. **Se hace la v9** (robustez al *domain gap*). Es la única versión conceptual nueva.
2. **PyTorch llega al navegador vía ONNX** (`onnxruntime-web`), no se descarta el requisito
   dual en el despliegue. Fallback si falla: el tercer modelo pasa a ser TF v4.
3. **Se fotografían diapositivas reales** con el teléfono, para medir el gap.

**Lo que NO se hace** (queda documentado como trabajo futuro en `explicacion.md` §9):
fine-tuning en dos fases, escenario SeqKD, `weight_decay`/`label_smoothing`/scheduler.
Son interesantes pero no cambian nada de la app, y el tiempo va a Ionic.

---

## 1. La v9 en una página

**El problema.** Todos los números del proyecto (v1→v8) son sobre **renders limpios**.
La app va a comer **fotos de teléfono**: perspectiva, reflejo de proyector, moiré de
pantalla, desenfoque, compresión JPEG, balance de blancos. Ese *domain gap* está
identificado como deuda #1 en `explicacion.md` §9, `analisis_metricas.md` §7 y
`avances_proyecto.md`, y **nunca se midió ni una vez**. Los números actuales son un
**techo optimista** respecto del despliegue real.

**El cambio conceptual (uno solo, como siempre).** La augmentation pasa de "domain-aware
suave" (v2→v8) a **simulación de fotografía**. Todo lo demás queda igual: misma partición
canónica, mismos hiperparámetros heredados de la v7, misma receta de entrenamiento de la
v8, mismas 5 semillas.

**El diseño experimental — 2 brazos × 2 tests:**

| | test-render (63 imgs limpias) | test-foto (las MISMAS 63, fotografiadas) |
|---|---|---|
| **Brazo A** — augmentation v8 (control) | ya medido (~0.87 TF / ~0.82 PT) | **mide el domain gap** |
| **Brazo B** — augmentation de foto (v9) | mide lo que se paga | **mide lo que se recupera** |

**Por qué el test-foto son las mismas 63 imágenes del test apartado:** las etiquetas
salen gratis (ya están), la partición canónica se respeta (esas imágenes nunca entraron
a entrenamiento ni a la búsqueda de hiperparámetros), y la comparación queda **pareada**
imagen por imagen — se puede mirar la degradación caso a caso, no solo el promedio.

**Regla dura: las fotos NO entran al entrenamiento.** Si entran, se pierde la medición,
que es todo el valor de la v9. El lado de entrenamiento lo cubre la augmentation.

**Criterio de veredicto, declarado ANTES de correr** (misma disciplina que la v8): una
diferencia entre brazos cuenta solo si `|B − A|` supera la suma de los desvíos de los dos
brazos sobre las 5 semillas. Con 63 imágenes, el resto es ruido.

### 1.1 Qué simula la augmentation de foto

| Efecto | Por qué | TensorFlow | PyTorch |
|---|---|---|---|
| Perspectiva leve | OpenCV.js rectifica, pero no perfecto: queda residual | `tf.raw_ops.ImageProjectiveTransformV3` (no hay capa Keras) | `transforms.RandomPerspective` (nativo) |
| Brillo no uniforme / glare | reflejo de proyector, gradiente de luz | máscara de gradiente a mano | máscara de gradiente a mano |
| Desenfoque + motion blur | mano alzada | `tfa`-style conv o kernel manual | `transforms.GaussianBlur` |
| Ruido gaussiano | ISO alto | `tf.random.normal` | `torch.randn` |
| Compresión JPEG | la cámara del teléfono comprime | `tf.io.encode_jpeg` / `decode_jpeg` | PIL round-trip |
| Balance de blancos / tinte | temperatura de la pantalla | `ColorJitter` equivalente | `transforms.ColorJitter` |
| Moiré | rejilla de píxeles de la pantalla | rejilla de alta frecuencia a baja opacidad (**aproximación, decirlo así**) | ídem |
| Recorte leve | la rectificación no queda exacta | `RandomCrop` tras padding | `RandomResizedCrop` acotado |

**Contraste de framework nuevo para §8 de `explicacion.md`:** PyTorch trae
`RandomPerspective` de fábrica; Keras **no tiene capa de perspectiva** y hay que bajar a
`tf.raw_ops.ImageProjectiveTransformV3` armando la matriz de homografía a mano. Es el
mismo patrón que ya apareció en la v8 (torchvision trae hecho lo que en Keras hay que
construir).

---

## 2. Los tres modelos que van a la app

Se despliegan las **variantes v9** de las tres arquitecturas (los pesos entrenados con la
augmentation de foto). Los pesos v8 quedan como el "antes" de la historia.

| # | Modelo | Por qué está | Runtime | Entrada |
|---|---|---|---|---|
| **M1** | **CNN propia destilada** (E3 del brazo cadena, 180px) | El **mejor medido** sobre el test apartado (TF 0.8857 / PT 0.8698). Es el hallazgo incómodo de la v8: la CNN chica le gana a las features congeladas de ImageNet. | TF.js | 180×180, 0-255 crudo |
| **M2** | **MobileNetV3-Small + cabeza Optuna + destilación** (v8 brazo B) | El arco completo del proyecto, y el **mejor calibrado y más estable** (ECE 0.092, ±0.0078 entre semillas). | TF.js | 224×224, 0-255 crudo |
| **M3** | **El espejo PyTorch de M2** | Único camino para que el requisito **dual** sobreviva al despliegue. Extiende `paridad.txt`: la app muestra los dos frameworks sobre la misma foto. | ONNX + `onnxruntime-web` | 224×224, **mean/std ImageNet a mano** |

**Dato de despliegue contraintuitivo, y buen material de oral:** M1 es el **más preciso y
el más pesado** (~15 MB fp32; el `Flatten → Dense(64)` tiene 3.8M parámetros contra ~1.5M
de MobileNetV3, que pesa 3.7 MB). Con cuantización uint8 del converter baja a ~4 MB.
*El modelo más preciso no es el más liviano.*

**El preprocesamiento distinto de M3 no es un detalle, es el §8 apareciendo en
producción:** en TF la normalización ImageNet viaja **dentro** del modelo
(`include_preprocessing=True`), así que el navegador manda 0-255 crudo; en PyTorch vive en
el *transform*, así que la app tiene que reproducir `mean`/`std` a mano o se
des-sincroniza en silencio.

**Opcional (stretch):** un cuarto slot con el **M2 en versión v8** (sin augmentation de
foto), para demostrar en vivo el efecto de la v9: la misma foto, el mismo modelo, con y
sin entrenamiento robusto. Es la demo más contundente que permite este proyecto — pero
solo si sobra tiempo y presupuesto de descarga.

---

## 3. Plan de commits

### Bloque A — modelos (Python, este repo)

**T9 — protocolo y verificación de fotos reales**
- `documentacion/protocolo_fotos.md`: instructivo de captura (ver §4).
- `documentacion/verificar_fotos.py`: chequea el pareo 1:1 contra la lista `test` de
  `particion.json`, formato y tamaño mínimo; `exit != 0` si falta alguna. Mismo espíritu
  que `verificar_dataset.py`.
- `dataset_fotos/` al `.gitignore` (las fotos viven fuera de git, como el dataset).
- No entrena nada: es el commit que **habilita** la medición.

**T10 — v9: augmentation de fotografía (espejo TF + PyTorch)**
- `TensorFlow/v9/09_scripts.py` · `PyTorch/v9/09_scripts.py`.
- Brazo A (control, augmentation v8) vs Brazo B (augmentation de foto), 5 semillas,
  evaluados sobre **test-render** y **test-foto** → la tabla 2×2.
- Métricas: las de `documentacion/metricas.py` (accuracy, F1 macro, QWK, MAE ordinal,
  errores 0↔2, AUC macro OvR, ECE). El **ECE en foto real** es el número que más importa
  para la app: si el modelo está mal calibrado sobre fotos, la UI miente.
- Figuras: `Figure_gap_v9.png` (barras del 2×2), `Figure_matriz_*_v9.png`,
  `Figure_semillas_v9.png`, `Figure_calibracion_foto_v9.png`.
- Salidas: `Resultado_9.txt` (log real completo), `resultados_v9.json`.
- Guarda los modelos **end-to-end** (backbone + cabeza + softmax, entrada 0-255):
  `modelotf_v9_foto.keras` · `modelopt_v9_foto.pt`.

**T11 — guardado end-to-end de la CNN propia + consumo v9**
- Parche a `v8/08_scripts.py` (los dos frameworks) para guardar también **E3** end-to-end
  — hoy solo guarda E5, y E3 es el mejor modelo del proyecto.
- `modelos/v9/modelotf_v9.py` · `modelopt_v9.py`: consumo espejo, vector completo de
  probabilidades, como toda la carpeta `modelos/`.

**T12 — export de despliegue de los 3 modelos + paridad**
- `despliegue/exportar_tfjs.py`: generaliza `TensorFlow/export_tfjs/exportar_tfjs.py`
  (hoy tiene v4 hardcodeado) para recibir el modelo por parámetro, y agrega cuantización
  uint8 del converter.
- `despliegue/exportar_onnx.py`: PyTorch → ONNX (opset fijo, shape fijo, `dynamic_axes`
  solo en batch), con verificación `onnxruntime` vs PyTorch en Python.
- `despliegue/verificar_paridad.py` + `despliegue/paridad.md`: los 3 modelos, Python vs
  navegador, **backends CPU y WebGL** — la paridad actual (`web/paridad.txt`) es solo CPU
  y en el celular vas a querer WebGL por velocidad. Criterio `< 0.02`.
- `web/index.html` actualizado a los 3 modelos: **smoke test antes de tocar Ionic**, igual
  que se hizo con v4 en la entrega 2.

### Bloque B — aplicación

**T13 — esqueleto Ionic**
- `ionic start is-it-ai blank --type=angular --standalone` en `app/`.
- `node_modules/` y `app/www/` al `.gitignore`.
- Páginas: *Capturar*, *Resultado*. Servicios: `inferencia.service.ts`, `vision.service.ts`.
- Modelos como **assets locales** en `app/src/assets/modelos/{m1,m2,m3}/`.
- TF.js y onnxruntime-web por **npm, no CDN**: la app tiene que funcionar sin internet en
  el examen.

**T14 — captura de imagen**
- Capacitor Camera (es lo que usa el tutorial que pasó el profe), con fallback a
  `getUserMedia` para probar en desktop.
- Foto → `<canvas>` → `ImageData`.

**T15 — OpenCV.js: rectificación de perspectiva**
- `opencv.js` como asset local (~9 MB WASM) con pantalla de carga: tarda.
- Pipeline: `cvtColor` → `GaussianBlur` → `Canny` → `findContours` →
  `approxPolyDP` (quedarse con el cuadrilátero de mayor área) → `warpPerspective` a
  180 o 224 px según el modelo.
- **Fallback explícito**: si no aparece un cuadrilátero razonable, resize directo y
  avisarlo en la UI. Sin esto la demo se cuelga con la primera foto rara.
- Preview del recorte rectificado antes de predecir.
- *Este paso es lo que hace que OpenCV.js no sea decorativo: la app corrige la perspectiva
  y la v9 entrena con las distorsiones que la rectificación no saca. App y modelo son una
  sola decisión de diseño.*

**T16 — inferencia y comparación de los 3 modelos**
- Warmup de cada modelo al arrancar (la primera inferencia en WebGL es lentísima).
- Preprocesamiento **por modelo** (0-255 crudo para M1/M2, mean/std para M3).
- Selector de modelo + **modo comparación**: los 3 sobre la misma foto, con las 3
  probabilidades y el tiempo de inferencia de cada uno. Esa pantalla es el examen entero
  en una imagen.
- **Vector completo de 3 probabilidades, nunca solo el argmax** — convención de la casa.
- Aviso de incertidumbre en la UI: con ECE ~0.10 esas probabilidades no son certezas.

**T17 — empaquetado y documentación**
- `npx cap add android` + APK, para no depender de la red del aula.
- Sección de app en `DetalleProyecto.md` y `explicacion.md`; nueva fila en el catálogo de
  contrastes §8 (perspectiva Keras vs torchvision; normalización dentro vs fuera del
  modelo, ahora con consecuencia en producción).
- `analisis_metricas.md`: incorporar el domain gap medido al plan de acción.

---

## 4. Protocolo de captura de fotos (tarea de Bato)

**Qué fotografiar:** las **63 imágenes de la lista `test`** de `documentacion/particion.json`.
Ni una del set de desarrollo.

**Cómo:**
1. Mostrar cada imagen a pantalla completa en un monitor o proyector.
2. Fotografiar con el teléfono, **sin flash**, a mano alzada.
3. **Variar a propósito**: ángulo (frontal y 10-30° fuera de eje), distancia, iluminación
   de la sala, monitor y proyector si hay los dos.
4. **No buscar la foto perfecta — buscar la foto realista.** Una foto de laboratorio mide
   un gap que no existe en el despliegue.

**Nombre del archivo:** exactamente el mismo que el original, en
`dataset_fotos/<clase>/<mismo_nombre>.jpg`. Así el pareo es automático y
`verificar_fotos.py` puede chequearlo.

**Verificación:** `python documentacion/verificar_fotos.py` antes de correr la v9.

---

## 5. Riesgos y cómo se mitigan

| Riesgo | Mitigación |
|---|---|
| **Peso total de la app**: opencv.js (~9 MB) + 3 modelos + tfjs + onnxruntime | Presupuestar desde T12 y cuantizar uint8. Medir el bundle en cada commit del bloque B. |
| **Paridad en WebGL**: la verificada es en CPU; WebGL puede correr en float16 | T12 la verifica en los dos backends antes de que la app dependa de eso. |
| **ONNX de PyTorch falla** | Fallback declarado: M3 pasa a ser TF v4 (el baseline histórico). Costo: ~10 minutos, y la app cuenta el arco `baseline → transfer+destilado → CNN propia`. |
| **iOS**: restricciones de `getUserMedia` y WASM | Si se demuestra en iPhone, probarlo **temprano**, no la semana del examen. |
| **La v9 no mejora nada** | Sigue siendo un resultado publicable: sería la primera medición del domain gap del proyecto, y la disciplina de "todas las diferencias son ruido" ya se ejerció en la v8. |
| **Sin red en el aula** | APK + todos los assets locales. Nada de CDN. |

---

## 6. Preguntas de la interrogación oral que este plan deja contestadas

- *¿Por qué esos tres modelos?* — El mejor medido, el mejor calibrado, y el espejo del
  otro framework. Con números del test apartado, no de la validación.
- *¿Para qué usás OpenCV.js?* — Para rectificar la perspectiva de la diapositiva; y la v9
  entrena con el residuo que la rectificación no corrige.
- *¿Cómo sabés que el modelo del navegador es el mismo que el de Python?* — `paridad.md`:
  3 modelos, 2 backends, criterio `< 0.02`, medido.
- *¿Tus números valen para el uso real?* — La v9 lo mide: acá está el gap y acá está lo
  que recupera la augmentation.
- *¿Por qué el modelo más preciso no es el que menos pesa?* — Porque el `Flatten` de la
  CNN propia tiene 3.8M parámetros y MobileNetV3 tiene 1.5M.
