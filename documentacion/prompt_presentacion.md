# Prompt para generar la presentación (.pptx) de este repositorio

Pega el bloque de abajo **tal cual** en Claude (Claude Code, con acceso al
repositorio y al MCP de Stitch habilitado). Es un único prompt autónomo: pide una
presentación completa sobre este directorio, usando todas las skills disponibles
más el toolkit de Stitch, con animaciones y transiciones, y entregada como
archivo `.pptx`.

---

```
Contexto: estás dentro del repositorio "Is-it-AI" (detector de huella de IA en
diapositivas; clasificador de deep learning en 3 niveles, construido en paralelo
en TensorFlow y PyTorch con MobileNetV3Large + transfer learning, con destino
Android). Antes de generar nada, LEE el repositorio para basarte solo en hechos
reales: README.md, documentacion/ (SETUP.md, guia_etiquetado.md), TensorFlow/ y
PyTorch/ (v1 baseline y v2: config, datos, modelo, entrenar, metricas,
consumidor, grad_cam, exportar_*), y herramientas/. No inventes métricas ni
resultados que el código no produzca; si un número aún no se midió, dilo como
"pendiente de entrenar".

Objetivo: producir una PRESENTACIÓN TÉCNICA COMPLETA sobre este directorio y
entregármela como un archivo .pptx real y descargable.

Usa TODAS las capacidades que tengas disponibles, orquestándolas:
- Todas las skills de Claude que apliquen (diseño de frontend/deck, generación de
  imágenes/diagramas, escritura técnica, etc.).
- El toolkit MCP de Stitch para diseñar el sistema visual y las pantallas/slides
  (design system, generación de screens desde texto, variantes). Deriva de ahí la
  identidad visual coherente de todo el deck.
- Animaciones y transiciones: cada diapositiva con transición de entrada y los
  elementos clave (títulos, viñetas, diagramas) animados en secuencia. Que se vea
  fluido, no estático.

Idioma: español de Chile, técnico pero claro, sobrio, sin relleno.
Audiencia: profesor y compañeros del curso "Frameworks de IA" (UDD).

Estilo visual: tema oscuro, fondo azul-noche (#0B0E14), acento en degradado
violeta-cian (#7C5CFF -> #19D3C5), tipografía sans-serif moderna, minimalista,
mucho espacio en blanco, diagramas planos (flat design), sin imágenes tipo stock.
Naranja para lo de TensorFlow, rojo para lo de PyTorch cuando compares.

Estructura sugerida (ajústala si el repo lo amerita, ~13 diapositivas):
1.  Portada: "¿Es hecha por IA?" — detector de huella de IA en diapositivas.
2.  El problema: la huella de IA no es binaria; casi siempre hay edición humana.
3.  La idea: gradiente de 3 niveles (0_sin_ia / 1_rastro_ia / 2_saturada_ia).
4.  Dos frameworks en paralelo: TensorFlow vs PyTorch (mismo modelo, comparables).
5.  Por qué MobileNetV3Large + transfer learning (ImageNet, tamaño celular).
6.  Pipeline del dataset: presentaciones -> PNG por slide -> etiquetado -> split.
7.  v1: CNN desde cero que sobreajusta a propósito (brecha de generalización).
8.  v2: transfer learning en dos fases (cabeza congelada -> fine-tuning).
9.  Cómo medimos: curvas train/val, matriz de confusión, ROC/AUC (sin sklearn).
10. Explicabilidad: Grad-CAM (dónde mira el modelo) — implementado en ambos lados.
11. Rumbo a Android: export a TFLite (TF) y a Lite/.ptl (PyTorch).
12. Estado actual y limitaciones (honesto: dataset pequeño, sin OCR aún).
13. Próximos pasos y cierre ("¿Preguntas?" · Frameworks de IA · UDD).

Requisitos de entrega (importante):
- El resultado final DEBE ser un archivo .pptx (PowerPoint) real, guardado en el
  repo (p. ej. documentacion/presentacion_is_it_ai.pptx) y listo para abrir.
- Conserva las animaciones y transiciones en el .pptx en la medida en que el
  formato lo permita; si alguna animación de Stitch no es exportable a .pptx,
  reprodúcela con las animaciones/transiciones nativas de PowerPoint y anótalo.
- Incluye los diagramas/heatmaps como imágenes generadas, no como texto plano.
- Deja los placeholders de métricas claramente marcados (ej. "AUC: pendiente")
  para reemplazar por los valores reales tras entrenar con el dataset definitivo.

Flujo de trabajo esperado:
1. Lee el repo y resume en 5 bullets qué vas a presentar (confírmame el guion).
2. Define el design system en Stitch y genera las pantallas/slides.
3. Ensambla el .pptx con animaciones y transiciones.
4. Entrégame la ruta del .pptx y una lista de qué quedó pendiente/aproximado.
```

---

### Notas
- Reemplaza cualquier número de ejemplo por los reales una vez que entrenes con el
  dataset definitivo (matriz de confusión, AUC y curvas salen de `entrenar.py` /
  `metricas.py` en cada framework).
- Si Stitch no puede exportar directo a `.pptx`, el prompt ya instruye reconstruir
  el deck en PowerPoint conservando animaciones/transiciones equivalentes.
