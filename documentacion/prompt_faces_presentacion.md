# Prompt para generar la presentación en Faces.app — diapositiva por diapositiva

Cómo usar este archivo:
1. Primero pega el **PROMPT DE ESTILO GLOBAL** (define tema, paleta, idioma y tono).
2. Luego genera **una diapositiva a la vez**: copia el prompt de "Diapositiva N",
   genérala, revísala, y recién ahí pasa a la siguiente.
3. Cada prompt es autónomo: repite el estilo para que las diapositivas queden
   coherentes aunque las generes por separado.

Idioma: **español**. Audiencia: profesor y compañeros del curso *Frameworks de IA*.
Tono: técnico pero claro, sobrio, nada de relleno.

---

## PROMPT DE ESTILO GLOBAL (pegar una vez al inicio)

```
Vas a ayudarme a crear una presentación técnica en español para un curso
universitario de "Frameworks de IA". Tema: un detector de huella de IA en
diapositivas (clasificador de deep learning). Estilo visual para TODAS las
diapositivas:
- Tema oscuro, fondo azul-noche (#0B0E14) con un acento violeta-cian
  (degradado #7C5CFF -> #19D3C5).
- Tipografía sans-serif moderna, títulos en negrita, mucho espacio en blanco.
- Minimalista y técnico: pocas palabras por diapositiva, jerarquía clara.
- Cuando pida un gráfico o diagrama, hazlo limpio y plano (flat design), sin
  imágenes genéricas tipo "stock de negocios".
- Idioma: español de Chile, neutro y profesional.
Voy a pedirte las diapositivas de a una. Espera mi prompt de cada diapositiva.
```

---

## Diapositiva 1 — Portada

```
Crea la diapositiva 1 (portada). Título grande: "¿Es hecha por IA?".
Subtítulo: "Detector de huella de IA en diapositivas — gradiente de 3 niveles".
Pie: "Frameworks de IA · UDD · Avances". Estilo oscuro con degradado violeta-cian
en el título. Minimalista, sin imágenes de relleno.
```

## Diapositiva 2 — El problema

```
Crea la diapositiva 2. Título: "Las diapositivas con IA tienen un 'look'… pero
no es blanco o negro". Tres viñetas cortas:
- Las presentaciones hechas con IA comparten patrones reconocibles (plantillas,
  imágenes genéricas, texto de 'voz IA').
- Pero casi siempre hay edición humana encima: no podemos decir que todo es IA.
- Una etiqueta binaria (IA / no IA) sería deshonesta con el enorme medio asistido.
Cierra con una frase destacada: "Por eso lo modelamos como un gradiente".
```

## Diapositiva 3 — La idea: 3 niveles

```
Crea la diapositiva 3. Título: "Tres niveles de huella de IA".
Muestra una barra horizontal de gradiente con 3 segmentos etiquetados:
"0_sin_ia"  ->  "1_rastro_ia"  ->  "2_saturada_ia".
Debajo, tres tarjetas breves:
- 0_sin_ia: sin huella de IA visible; fotos/datos reales, layout humano.
- 1_rastro_ia: andamiaje de IA + clara intervención humana (datos, capturas).
- 2_saturada_ia: predominantemente IA (plantilla, imágenes IA, texto genérico).
Nota al pie: "El softmax entrega confianza, no un sí/no rígido".
```

## Diapositiva 4 — Dos frameworks

```
Crea la diapositiva 4. Título: "El mismo modelo, en dos frameworks".
Dos columnas comparativas:
- TensorFlow: MobileNetV3Large (Keras) · 2 fases · export a .tflite.
- PyTorch: MobileNetV3Large (torchvision) · 2 fases · export a .ptl (lite).
Abajo, frase: "Mismo orden de clases en ambos -> resultados comparables (objetivo
del curso: comparar frameworks)". Usa los logos/colores naranja (TF) y rojo (PyTorch).
```

## Diapositiva 5 — Por qué transfer learning + MobileNetV3

```
Crea la diapositiva 5. Título: "MobileNetV3Large + Transfer Learning".
Tres puntos:
- Preentrenada en ImageNet -> aprende con un dataset pequeño (nuestra etapa).
- Diseñada para celulares -> el objetivo Android está incorporado desde el día 1.
- Entrenamiento en dos fases: (1) cabeza con base congelada, (2) fine-tuning de
  las últimas capas con learning rate bajo.
Incluye 3 métricas destacadas tipo tarjeta: "224² entrada", "3 clases",
"~3-5M parámetros".
```

## Diapositiva 6 — Pipeline del dataset

```
Crea la diapositiva 6. Título: "De presentaciones a diapositivas etiquetadas".
Muestra un flujo horizontal de 4 pasos con flechas:
1) Dejar .pdf/.pptx en herramientas/presentaciones_fuente/  ->
2) Conversor: 1 PNG por diapositiva (herramientas/sin_clasificar/)  ->
3) Clasificar en dataset/ {0_sin_ia / 1_rastro_ia / 2_saturada_ia} (rúbrica)  ->
4) Revisar balance + split 70/15/15.
Nota: "Sin capturas manuales. Una rúbrica escrita mantiene etiquetas consistentes
(40-80 por clase, balanceadas, de presentaciones variadas)".
```

## Diapositiva 7 — v1: el baseline que sobreajusta

```
Crea la diapositiva 7. Título: "v1 — CNN desde cero (sobreajusta a propósito)".
Explica en 2-3 viñetas: una CNN simple (Conv-Conv-Dense), sin augmentation ni
dropout ni transfer learning. Entrena bien en train pero mal en validación:
esa "brecha de generalización" es la lección que motiva la v2.
Incluye un mini-gráfico ilustrativo de dos curvas (train sube, val se estanca).
```

## Diapositiva 8 — v2: dos fases de entrenamiento

```
Crea la diapositiva 8. Título: "v2 — Transfer learning en dos fases".
Muestra como código/pseudocódigo limpio:
  Fase 1: base congelada -> entrenar solo la cabeza (15 épocas, lr 1e-3)
  Fase 2: descongelar últimas capas -> fine-tuning (10 épocas, lr 1e-4)
Más augmentation (flip, rotación, zoom, contraste) y dropout 0.3.
Nota al pie: "Mismo flujo en TensorFlow y PyTorch".
```

## Diapositiva 9 — Cómo medimos (métricas)

```
Crea la diapositiva 9. Título: "Cómo evaluamos el modelo".
Tres bloques con íconos:
- Curvas de entrenamiento (accuracy y loss, train vs val) -> ver overfitting.
- Matriz de confusión + precision/recall/F1 por clase (calculadas a mano, sin sklearn).
- Curvas ROC one-vs-rest + AUC por clase.
Frase: "Accuracy, loss y AUC se leen EN CONJUNTO entre train y validación: su
diferencia es la brecha de generalización".
```

## Diapositiva 10 — Explicabilidad: Grad-CAM

```
Crea la diapositiva 10. Título: "¿Dónde mira el modelo? (Grad-CAM)".
Explica que Grad-CAM genera un mapa de calor sobre la diapositiva mostrando qué
zonas pesaron en la decisión (layout/plantilla, imágenes generadas, texto).
Incluye una ilustración esquemática de una diapositiva con un mapa de calor
superpuesto en el área de una imagen. Nota: "Respalda el criterio de las 3 clases".
```

## Diapositiva 11 — Objetivo móvil: Android

```
Crea la diapositiva 11. Título: "Rumbo a Android".
Dos tarjetas:
- TensorFlow -> TFLite (tensorflow-lite), entrada 224x224x3 RGB 0-255.
- PyTorch -> Lite (.ptl, pytorch_android_lite), entrada 1x3x224x224 normalizada.
Frase: "Elegimos MobileNetV3 justamente por esto: el modelo es de tamaño celular
desde el inicio; la cuantización int8 lo achica aún más".
```

## Diapositiva 12 — Estado actual y limitaciones

```
Crea la diapositiva 12. Título: "Dónde estamos (honesto)".
Columna "Listo" (con checks): estructura del repo, conversor de presentaciones,
ambos pipelines (TF + PyTorch) con métricas y export móvil, rúbrica de etiquetado,
flujo verificado de punta a punta.
Columna "Limitaciones": dataset pequeño -> accuracy moderada; modelo ve solo la
imagen (sin OCR todavía); el límite de 1_rastro_ia es subjetivo.
```

## Diapositiva 13 — Próximos pasos y cierre

```
Crea la diapositiva 13 (cierre). Título: "Próximos pasos".
Flujo de 3 hitos: 1) crecer y balancear el dataset + matriz de confusión real,
2) agregar señales de texto/OCR para los casos 1_rastro_ia difíciles,
3) construir la app Android sobre el modelo exportado.
Cierra grande con: "¿Preguntas?" y el pie "Frameworks de IA · UDD".
```

---

### Notas para el expositor
- Si Faces.app permite generar todo de una, igual conviene revisar diapositiva a
  diapositiva para que no invente datos (ej: que no ponga un accuracy concreto que
  no medimos aún).
- Reemplaza cualquier número de ejemplo por los reales una vez que entrenes con el
  dataset definitivo (matriz de confusión y AUC salen de `entrenar.py`).
