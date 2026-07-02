# Detección de huella de IA en diapositivas — Documento de avances

*Estado al día de la presentación. Compilado para no dejar nada afuera.*

---

## 0. Resumen en una línea

Tenemos el pipeline de datos definido, un dataset inicial balanceado (300 imágenes, 100 por clase) y **dos versiones del modelo en TensorFlow corridas y diagnosticadas**: la v1 demostró el sobreajuste esperado y la v2 lo eliminó con data augmentation. La v2 quedó sub-entrenada (hallazgo esperable), y ese es el punto de partida para la siguiente iteración.

---

## 1. Objetivo del proyecto

Construir un clasificador **móvil** que, a partir de una **foto de teléfono de una diapositiva**, estime cuánta huella de generación por IA presenta esa diapositiva.

Requisitos del curso:
- Implementación **dual**: TensorFlow/Keras y PyTorch (contraste explícito entre frameworks).
- Base **MobileNetV3** vía *transfer learning* (arquitectura liviana, pensada para móvil).
- Exportación a **TensorFlow Lite** para inferencia en celular.

---

## 2. La decisión conceptual clave (el corazón del proyecto)

El instinto inicial era clasificar "hecha con IA / no hecha con IA" (binario). **El profe objetó, con razón**, y esa objeción reencuadró todo el proyecto:

- La **procedencia** de una presentación es **multidimensional** (la IA puede haber diseñado el layout, escrito el texto, generado las imágenes, o todo junto) y **parcial** (casos como "esqueleto generado + texto humano encima" no caen limpio en un sí/no).
- El punto decisivo: **una foto contiene el resultado visual, no el proceso.** Una CNN solo ve píxeles, así que solo puede aprender clases que *se vean distintas*. La huella de plantilla generada y las imágenes generadas dejan rastro visual; el origen del texto, no.

**Conclusión de diseño:** el proyecto **NO mide procedencia/proceso.** Mide **densidad de artefactos visuales de IA** observables en la imagen. El label dice lo que el modelo realmente puede aprender.

> Este es el argumento intelectual más fuerte que tenemos para la presentación: entendimos *qué puede y qué no puede aprender el modelo* antes de escribir una línea de código.

---

## 3. Las tres clases (eje ordinal)

| Índice | Carpeta | Clase | Definición operacional (lo que se ve en la foto) |
|--------|---------|-------|---------------------------------------------------|
| 0 | `0_sin_ia/` | **Sin rastro** | Sin artefactos de IA. Deck a mano o con plantilla clásica, espaciado "humano", fotos de stock o sin imágenes generadas. |
| 1 | `1_rastro_ia/` | **Hay rastro** | Densidad intermedia: p. ej. una imagen generada insertada en un deck por lo demás humano, o una plantilla de generador rellenada a mano. |
| 2 | `2_saturada_ia/` | **Saturada** | Firma de generador *end-to-end*: estética uniforme tipo Gamma/Tome, ilustraciones generadas, ritmo de layout parejo, tipografía característica. |

**Nota crítica:** la clase 1 se define por **aspecto (densidad de artefactos)**, NO por proceso. Definirla por proceso la convierte en un cajón de sastre visualmente heterogéneo que el modelo no puede aprender. La frontera **1↔2 es el punto más frágil del esquema.**

---

## 4. Decisiones de modelado

- **Arquitectura final prevista:** MobileNetV3 + transfer learning. *(Todavía no implementada; las v1/v2 usan una CNN mínima desde cero, a propósito, para ver el arco de sobreajuste primero.)*
- **Salida:** softmax de 3 unidades.
- **Loss:** `sparse_categorical_crossentropy` (correcta para labels enteros que entrega `image_dataset_from_directory`).
- **Tradeoff asumido conscientemente:** se tratan las clases como **nominales** aunque tengan orden. Confundir 0↔2 se penaliza igual que 0↔1, lo cual conceptualmente no es equivalente (confundir vecinos es "barato"; confundir extremos es grave). Se acepta el costo por simplicidad; la **regresión ordinal** sería lo teóricamente correcto pero es over-engineering para esta etapa.
- **Instrumento de evaluación principal: la matriz de confusión.** Si el error se concentra en vecinos (0↔1, 1↔2) → esperable y aceptable. Si aparece confusión **0↔2** → problema serio que invalida el modelo.

---

## 5. Estrategia de datos

**Estructura de carpetas** (idiomática para `image_dataset_from_directory` / `ImageFolder`):

```
data/               <- fuera del repo, ignorada por git
├── 0_sin_ia/
├── 1_rastro_ia/
└── 2_saturada_ia/
```

> **Gotcha resuelto:** Keras y PyTorch asignan índices ordenando los nombres de carpeta **alfabéticamente**. Los prefijos `0_/1_/2_` fuerzan que el índice coincida con el eje de saturación. Se verifica siempre con `print(class_names)` — confirmado en cada corrida: `['0_sin_ia', '1_rastro_ia', '2_saturada_ia']`.

**Fuentes por clase:**
- **Clase 0 (sin rastro):** datasets pre-2022 de HF/Zenodo (candidato principal `Forceless/Zenodo10K`, CC-BY 4.0, con `.pptx` manipulables). La fecha de subida sirve de label automático.
- **Clase 2 (saturada):** generación manual con Gamma, Copilot en PowerPoint, Canva Magic Design. **No existe dataset público.** Procedencia conocida = labels perfectos. **Esta clase, por ser la más cara de producir, fija el techo de tamaño de las otras dos.**
- **Clase 1 (hay rastro):** se construye deliberadamente; definida por aspecto, no por proceso.

**Principio de etiquetado:** invertir el problema. Dejar que la procedencia entregue el label automáticamente, en vez de scrapear genérico y etiquetar después (esto último maximiza esfuerzo manual y no da señal de label).

**Renderizado:** LibreOffice headless (`.pptx` → PNG por diapo, con layout completo). `python-pptx` solo extrae imágenes embebidas sin layout — útil para filtrar metadata, no para generar imágenes de entrenamiento.

**Formato:** JPEG ~calidad 90 (match de dominio con fotos de teléfono). No pre-redimensionar por debajo de ~256px en disco, para preservar flexibilidad hacia el input 224×224 de MobileNetV3.

**Estado actual del dataset:** ~100 imágenes por clase, **300 en total, balanceado.** Primer lote con señal real.

---

## 6. Brecha de dominio (pendiente de atacar de fondo)

Se entrena con **renders limpios**, pero se va a usar con **fotos de celular**: perspectiva, reflejos de proyector, moiré de pantalla, iluminación, compresión JPEG. Este *domain gap* **todavía no se probó ni una vez** — todos los resultados actuales son sobre renders limpios, así que son **optimistas** respecto del despliegue real. Mitigación prevista: data augmentation fuerte (perspectiva, brillo, blur, compresión JPEG) y/o fotografiar parte del set.

---

## 7. Infraestructura y entorno

| Componente | Elección |
|------------|----------|
| Plataforma | Linux Mint, local |
| Entornos | conda: `tensorflow-ia` (TF) y `frameworks-ia` (PyTorch) |
| Control de versiones | Git (nivel sistema) + GitHub |
| Estructura del repo | Carpetas `TensorFlow/` y `PyTorch/`; `data/` **fuera** del repo |
| `.gitignore` | `data/`, `__pycache__/`, modelos (`.h5`, `.pth`, `.tflite`) |
| Renderizado | LibreOffice headless |
| Compartir datos | Hugging Face Hub (preferido, versionado + integración directa); Google Drive (fallback) |
| Cómputo cloud | AWS S3 / $50 en créditos **reservados** para entrenamiento en la nube más adelante |

---

## 8. Trabajo técnico realizado — el arco experimental

Este es el núcleo empírico de la presentación. El arco es **deliberado y pedagógico**: ver el sobreajuste primero, después corregirlo, leyendo el estado del modelo en cada paso.

### v1 — Baseline (`01_baseline.py`)

CNN mínima desde cero: 2 bloques Conv+MaxPool (16→32), Flatten, Dense(64), Dense(3, softmax). ~3.79M parámetros (el 99% en la primera capa densa). **Sin augmentation, sin regularización, sin transfer learning — diseñada a propósito para sobreajustar.**

**Corrida 1 (placeholder, ~67 imágenes):** train y val al 100%. **Sin señal** — artefacto de tener solo 13 imágenes de validación (cada una vale ~7.7%) y un modelo que memoriza 54 ejemplos trivialmente. → Se decidió pausar y juntar datos reales.

**Corrida 2 (300 imágenes reales, 100/clase):**

| Métrica | Train | Val |
|---------|-------|-----|
| Accuracy final | 1.0000 | 0.9667 |
| Loss final | 0.0099 | 0.1026 |

**Diagnóstico:** el sobreajuste **está, pero se ve en la loss, no en la accuracy.** La accuracy casi no muestra brecha porque las clases son *muy separables* en renders limpios; pero la loss delata la memorización (0.01 vs 0.10, ~10×). El 96.7% es **engañoso**: no es la métrica de despliegue.

### v2 — Data augmentation (`02_augmentation.py`)

Cambios sobre v1, uno conceptual y dos de instrumentación:
1. **Data augmentation domain-aware**: `RandomRotation`, `RandomZoom`, `RandomBrightness`, `RandomContrast`. **Deliberadamente SIN flip horizontal** — una diapo espejada tiene el texto en espejo, algo que nunca ocurre en el mundo real. (Lección: las transformaciones deben ser realistas *para el dominio*; copiar el flip del clasificador de Santa habría sido un error.)
2. Se agrega el gráfico de **loss** (no solo accuracy), porque ahí estaba la señal.
3. Se agrega la **matriz de confusión** sobre validación (instrumento del §4), con `tf.math.confusion_matrix` (nativo, sin dependencia de sklearn).

**Corrida (300 imágenes, 10 épocas):**

| Métrica | Train | Val |
|---------|-------|-----|
| Accuracy final | 0.7708 | 0.7500 |
| Loss final | 0.5285 | 0.5501 |

**Matriz de confusión (split 24/18/18):**

```
                 Predicho
              0    1    2
Real  0  [   20    4    0  ]
      1  [    4   13    1  ]
      2  [    3    3   12  ]
```

**Diagnóstico (tres lecturas separadas):**
- **El overfitting se murió.** Loss train (0.53) y val (0.55) quedaron pegadas — firma de un modelo bien regularizado. La augmentation funcionó.
- **La accuracy "bajó" a 75%, pero eso es honestidad, no retroceso.** El 96.7% de v1 era memorización de renders casi duplicados; el 75% es real.
- **Pero el modelo quedó sub-entrenado.** Train llegó apenas a 0.77 (en v1 llegaba a 1.0) y las curvas **seguían mejorando en la época 10, sin aplanarse.** Con augmentation cada época cuesta más; 10 no alcanzan.

> **Lección de método (la más importante):** el próximo paso NO es agregar más regularización. Dropout o L1/L2 ahora sería lo contrario de lo que hace falta — no sobreajustamos, sub-entrenamos. La receta no es "siempre sumar técnicas"; es leer el estado y responder a lo que se ve.

**Sobre la matriz:** la mayoría del error está entre **vecinos** (esperable y barato). El punto de atención es la esquina **0↔2**: **3 diapos `saturada` clasificadas como `sin rastro`** (asimétrico: de 0 hacia 2 no se escapa ninguna). Es exactamente la confusión que el §4 marca como grave. **Salvedad:** esta matriz se calculó sobre un modelo sub-entrenado, así que esos 3 casos pueden ser ruido de no-convergencia. Hay que re-entrenar hasta converger antes de sacar conclusiones.

---

## 9. Logros (qué conseguimos)

1. **Reencuadre conceptual sólido:** pasamos de un binario ingenuo a un esquema ordinal de 3 clases fundamentado en lo que una CNN *puede* aprender (artefactos visibles, no procedencia).
2. **Pipeline de datos definido de punta a punta:** fuentes por clase, estrategia de labels automáticos por procedencia, renderizado con LibreOffice, formato y resolución, estructura de carpetas con el gotcha alfabético resuelto.
3. **Dataset inicial real y balanceado:** 300 imágenes, 100 por clase.
4. **Infraestructura reproducible:** repo estructurado, entornos separados por framework, `.gitignore` correcto, plan de almacenamiento/compartición.
5. **Dos iteraciones del modelo corridas y correctamente diagnosticadas:** demostramos el sobreajuste (v1) y lo eliminamos (v2), con la matriz de confusión ya funcionando como instrumento.
6. **Método de trabajo demostrado:** desarrollo incremental y versionado, leyendo el estado del modelo en cada paso en vez de amontonar técnicas.

---

## 10. Puntos débiles y pendientes (honestos — en un avance, suman)

1. **Protocolo de etiquetado de la frontera 1↔2: todavía sin escribir.** Es el pendiente más importante desde el inicio (§7 del contexto original). Sin una regla concreta y reproducible de qué/cuántos artefactos hacen `1` vs `2`, cualquier ambigüedad se convierte en ruido de label que el modelo aprende como confusión. **Test de validación previsto:** etiquetar ~10 casos límite por separado entre integrantes; si coinciden, la frontera existe; si no, hay que afinar la regla.
2. **La confusión 0↔2 (3 casos) sin resolver.** Puede ser ruido de sub-entrenamiento o algo real (diapos "saturadas" que visualmente se ven limpias → nos devuelve al protocolo de etiquetado). Se decide re-entrenando hasta converger.
3. **Domain gap sin probar.** Todos los números son sobre renders limpios; no hay una sola foto de teléfono en el circuito todavía. Los resultados actuales son un techo optimista.
4. **Split no estratificado.** `image_dataset_from_directory` agarró 60 imágenes de validación sin balancear por clase (quedó 24/18/18, no 20/20/20). Menor, pero conviene estratificar al comparar porcentajes.
5. **Dataset todavía chico.** 300 imágenes es un primer lote; para transfer learning serio conviene más volumen, con el techo puesto por la clase 2 (la más cara de producir).
6. **Sin PyTorch todavía.** El requisito dual del curso sigue pendiente; hoy solo hay TensorFlow.
7. **Sin MobileNetV3 ni TFLite todavía.** Las v1/v2 son CNN desde cero (intencional para el arco pedagógico); la arquitectura final y la exportación móvil son etapas posteriores.

---

## 11. Roadmap (qué sigue)

**Inmediato — cerrar la v2 correctamente:**
- Re-entrenar con `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)` y techo alto de épocas, para llevar el modelo a convergencia.
- Volver a leer la matriz de confusión: si los 3 casos 0↔2 desaparecen → era falta de entrenamiento; si persisten → hay algo de fondo (probable problema de etiquetado en esas imágenes).

**Según ese resultado, la v3 será una de dos:**
- Si la matriz queda sana → **atacar el domain gap** (augmentation pesada: perspectiva, compresión JPEG, blur, moiré) y empezar a fotografiar parte del set.
- Si persiste la confusión 0↔2 → **escribir y validar el protocolo de etiquetado 1↔2** antes de seguir modelando.

**Etapas siguientes (orden tentativo):**
- Estratificar el split (o pasar a split en disco train/val/test) al comparar frameworks.
- Regularización adicional (dropout, L1/L2) **solo si** vuelve a aparecer sobreajuste tras converger.
- **Transfer learning con MobileNetV3** (reemplaza la CNN desde cero).
- **Implementación paralela en PyTorch**, con las diferencias de framework explícitas (channels-last vs channels-first, `model.train()`/`eval()` vs modo automático, `weight_decay` vs suma manual de L1, augmentation en el `transform` del loader vs capas del modelo).
- **Exportación a TensorFlow Lite** para inferencia en celular.
- Ampliar el dataset (subiendo el techo de la clase 2).

---

## 12. Mensajes clave para la presentación (los 4 que no se pueden pasar)

1. **No medimos procedencia, medimos densidad de artefactos visuales** — porque es lo único que una CNN puede aprender de una foto. (El reencuadre por la objeción del profe.)
2. **El arco v1→v2 es una demostración de método:** vimos el sobreajuste (v1), lo eliminamos con augmentation domain-aware (v2), y ahora leemos que el modelo quedó sub-entrenado. No amontonamos técnicas: respondemos a lo que el diagnóstico muestra.
3. **La matriz de confusión —no la accuracy— es el instrumento**, porque la accuracy esconde *qué* se confunde, y en un eje ordinal confundir extremos (0↔2) es cualitativamente distinto de confundir vecinos.
4. **Lo que falta es tan importante como lo hecho:** protocolo de etiquetado 1↔2, prueba del domain gap real, y el requisito dual PyTorch. Están identificados y priorizados.
