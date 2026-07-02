# Notas del expositor — presentacion_is_it_ai.pptx

Una sección por diapositiva (8 en total). Tiempo objetivo: **8–10 minutos** + preguntas.
Los elementos de cada diapositiva entran solos (animación automática); no hay que
hacer clicks dentro de una diapositiva, solo avanzar entre diapositivas.

---

## Slide 1 — Presentación (≈20 s)

- Presentarse y dar el nombre del proyecto: **Is-it-AI**, detector de huella de IA en diapositivas.
- Gancho: "cuando vemos una presentación hecha con IA, todos la reconocemos… ¿puede una red neuronal?"
- No detenerse: la portada es tránsito.

## Slide 2 — Descripción del proyecto (≈70 s)

- Objetivo: un clasificador **móvil** que, desde una **foto de teléfono** de una diapositiva, estime cuánta huella de IA presenta.
- **La decisión de diseño clave**: no medimos procedencia (el proceso), medimos **densidad de artefactos visuales**. Una foto contiene el resultado, no el proceso; una CNN solo ve píxeles → solo puede aprender clases que se vean distintas. El origen del texto, por ejemplo, no deja rastro visual.
- Contexto: el instinto inicial era un binario IA/no-IA; la objeción del profesor reencuadró todo el proyecto hacia este esquema.
- Requisitos del curso que ordenan el trabajo: dual TensorFlow + PyTorch, MobileNetV3 vía transfer learning, exportación móvil (TFLite).
- Estrategia deliberada: CNN simple primero (v1/v2) para diagnosticar con control; la arquitectura final llega en v3.

## Slide 3 — Separación en 3 categorías (≈50 s)

- Recorrer la barra: `0_sin_ia` (sin artefactos, deck humano) → `1_rastro_ia` (densidad intermedia: ej. una imagen generada dentro de un deck humano) → `2_saturada_ia` (firma de generador end-to-end tipo Gamma/Tome: estética uniforme, ilustraciones generadas, ritmo de layout parejo).
- Por qué 3 y no 2: una etiqueta binaria sería deshonesta con el enorme medio asistido.
- Nota técnica si preguntan: las clases se tratan como nominales aunque el eje tiene orden; confundir extremos (0↔2) es cualitativamente más grave que confundir vecinos — por eso la matriz de confusión es el instrumento central (se ve en las slides 5 y 6).
- Ser transparente: la frontera 1↔2 es el punto más frágil; el protocolo de etiquetado está pendiente.

## Slide 4 — Datos usados (≈50 s)

- 300 imágenes reales, balanceadas (~100 por clase). Primer lote con señal.
- El truco del etiquetado: **invertir el problema** — en vez de scrapear y etiquetar a mano, la procedencia entrega el label automáticamente: datasets de decks pre-2022 (Zenodo) → clase 0; generar nosotros con Gamma/Copilot/Canva → clase 2 (procedencia conocida = labels perfectos).
- Render: LibreOffice headless convierte .pptx → un PNG por diapositiva con el layout completo.
- Gotcha resuelto: Keras y PyTorch asignan índices ordenando carpetas alfabéticamente → los prefijos 0_/1_/2_ fuerzan que índice = nivel de saturación (verificado en cada corrida).
- La clase 2 es la más cara de producir → fija el techo de tamaño de las otras dos.

## Slide 5 — PyTorch (≈80 s)

- Estructura del experimento (aplica igual a la slide siguiente): **v1** = CNN desde cero (3.79M parámetros), sin regularización, diseñada para sobreajustar; **v2** = misma red + data augmentation, todo lo demás igual (comparación A/B limpia).
- v1: accuracy 0.996 train vs 0.833 val, y loss 0.029 vs 0.406 — **memoriza**.
- v2: accuracy 0.708/0.717, loss 0.669/0.707 — **curvas pegadas, el sobreajuste desapareció**.
- Augmentation en PyTorch: va en el transform del DataLoader y **solo en el split de train** (val/test usan el transform limpio). Es la diferencia idiomática con Keras.
- Matriz: el error se concentra entre vecinos; solo 1 caso extremo (2→0).
- Detalle si preguntan por qué las curvas se ven distintas a TF: split distinto (random_split con seed propia) y detalles de las transforms; la conclusión es la misma.

## Slide 6 — TensorFlow (≈80 s)

- Mismo experimento, mismo arco. v1: accuracy 1.000/0.967 pero **loss 0.010 vs 0.103 — brecha de ~10×**. El hallazgo no obvio: el sobreajuste casi no se ve en la accuracy (las clases son muy separables en renders limpios); **la loss lo delata**. El 96.7% es engañoso: memorización, no métrica de despliegue.
- v2: accuracy 0.771/0.750, loss 0.529/0.550 — augmentation funcionó; "bajar" a 75% es honestidad, no retroceso.
- Augmentation en Keras: capas del modelo (RandomRotation, RandomZoom, RandomBrightness, RandomContrast) que se apagan solas en inferencia. **Sin flip a propósito**: una diapo espejada tiene texto en espejo — no existe en el mundo real.
- Matriz: 3 casos extremos 2→0 (asimétrico). Salvedad importante: la matriz sale de un modelo **sub-entrenado**, puede ser ruido de no-convergencia — se decide re-entrenando.
- Remate: dos implementaciones independientes → **misma conclusión**. La lectura del experimento es robusta.

## Slide 7 — Estado actual general (≈60 s)

- Columna verde (logrado): pipeline de datos punta a punta, dataset balanceado, v1+v2 corridas y diagnosticadas en ambos frameworks, matriz de confusión funcionando como instrumento.
- Columna teal (diagnóstico): el arco completo — vimos el sobreajuste (v1), lo eliminamos (v2), y la lectura fina es que ahora el modelo está **sub-entrenado**: train llegó apenas a 0.77 y las curvas seguían subiendo en la época 10. Con augmentation cada época cuesta más; 10 no alcanzan.
- Columna roja (pendiente): protocolo de la frontera 1↔2 (el más importante), domain gap (todo es sobre renders limpios: los números actuales son un techo optimista), la confusión 0↔2, y MobileNetV3 + export móvil.
- Mensaje: **lo que falta es tan importante como lo hecho — está identificado y priorizado.**

## Slide 8 — Próximos pasos (≈40 s)

- Roadmap en orden: (1) converger la v2 con EarlyStopping y techo alto de épocas → (2) re-leer la matriz: si los casos 0↔2 desaparecen era falta de entrenamiento; si persisten, hay que atacar el etiquetado → (3) MobileNetV3 con transfer learning → (4) exportación móvil y app Android.
- **La lección de método del avance**: no sobreajustamos, sub-entrenamos — agregar dropout o L1/L2 ahora sería lo contrario de lo que hace falta. La receta no es "siempre sumar técnicas"; es leer el estado y responder.
- Cierre: "¿Preguntas?"

---

### Preguntas probables y respuestas cortas

- **¿Por qué no MobileNetV3 de una?** → El arco v1→v2 con una CNN simple permite ver y corregir el sobreajuste de forma controlada; transfer learning llega en v3 con el pipeline ya validado.
- **¿Por qué la accuracy de PyTorch difiere de TF?** → Split distinto y detalles de las transforms; el orden de magnitud y la conclusión son idénticos.
- **¿Cómo garantizan que las clases significan algo?** → Labels por procedencia (no por juicio visual) + pendiente explícito: protocolo de la frontera 1↔2 con test de acuerdo entre etiquetadores (~10 casos límite etiquetados por separado; si coinciden, la frontera existe).
- **¿Sirve para fotos reales de proyector?** → Aún no probado (domain gap); la augmentation de v2 apunta en esa dirección y es el siguiente experimento tras converger.
- **¿Por qué accuracy y no F1/AUC?** → En este avance el instrumento es la matriz de confusión (posición del error en el eje ordinal); métricas por clase llegan cuando el modelo converja.
