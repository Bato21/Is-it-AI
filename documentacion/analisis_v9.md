# Análisis de métricas v9 y plan de acción

> Qué se midió, qué significa cada número, y qué se hace con esa información.
>
> Todos los valores salen de `TensorFlow/v9/resultados_v9.json`,
> `PyTorch/v9/resultados_v9.json` y `PyTorch/v9/resultados_modelo_c_v9.json`.
> Ninguno está escrito a mano.

---

## 0. Lo primero: la v9 cambió qué se mide, no solo cuánto

Las versiones v1–v8 evaluaban sobre **renders limpios**: páginas de PDF/PPTX exportadas por
software, con píxeles exactos. La app que se entrega consume otra cosa: **fotografías de una
pantalla**, con moiré, glare, perspectiva y compresión de cámara.

Por eso la v9 rehizo la partición (`documentacion/particion_v9.py`) con una regla explícita:

> **El test y la validación son 100% fotos. Los renders solo entrenan.**

**Consecuencia que hay que decir en voz alta:** la accuracy de la v9 **no es comparable** con
la de la v8. No porque el modelo empeorara, sino porque hasta la v8 se estaba midiendo un
problema más fácil que el que el producto resuelve. El número de la v9 es el que un cliente
puede exigir.

### El dataset

| clase | total | fotos | renders |
|---|---:|---:|---:|
| `0_sin_ia` | 434 | 326 | 108 |
| `1_rastro_ia` | 435 | 341 | 94 |
| `2_saturada_ia` | 468 | 370 | 98 |
| **total** | **1337** | **1037** | **300** |

| bloque | imgs | composición | uso |
|---|---:|---|---|
| TEST | 207 | 100% fotos | el número que se reporta |
| TEST_RENDER | 61 | 100% renders | solo para medir la brecha de dominio |
| folds 0–4 | 830 | 100% fotos | validación cruzada |
| solo entrenar | 239 | 100% renders | nunca validan |

La detección foto/render es un heurístico auditable (`documentacion/dominio.py`): EXIF de
cámara → nombre de archivo → si nada dispara, se asume render. El caso conservador es
etiquetar una foto como render (pierde una foto de entrenamiento); el caso peligroso sería el
inverso, porque metería un render en el test y contaminaría la métrica comercial.

> **Detalle que casi arruina la partición:** 326 fotos de la clase 0 se llamaban
> `Copia de 20260815_124529.jpg`. El prefijo "Copia de" rompía el patrón de nombre de cámara,
> y sin el respaldo del EXIF esas 326 fotos habrían entrado como renders — dejando a la clase
> 0 sin ninguna foto en el test. La regla ahora normaliza prefijos de duplicado.

---

## 1. Modelos A y B — el experimento de las dos fases

Ambos son MobileNetV3-Small. **A** en TensorFlow, **B** en PyTorch, con la misma partición,
las mismas vistas aumentadas (semilla fija 999) y las mismas métricas. 5 semillas, comparación
pareada.

- **Fase 1** — backbone congelado + cabeza con los hiperparámetros que Optuna eligió en la v7.
- **Fase 2** — se descongela el último bloque convolucional, `lr = 1e-5`, 4 épocas,
  BatchNorm congeladas.

### Resultados sobre el test de fotos (media ± desvío entre 5 semillas)

| | A · TensorFlow F1 | A · TF F2 | veredicto | B · PyTorch F1 | B · PT F2 | veredicto |
|---|---:|---:|---|---:|---:|---|
| accuracy | 0.9256 ± 0.0128 | **0.9478 ± 0.0083** | GANA FT | 0.9208 ± 0.0117 | **0.9372 ± 0.0031** | GANA FT |
| F1 macro | 0.9234 | **0.9475** | GANA FT | 0.9205 | **0.9374** | GANA FT |
| recall clase 2 | 0.9676 | 0.9730 | ruido | 0.9216 | 0.9378 | ruido |
| QWK (ordinal) | 0.8655 | 0.8948 | ruido | 0.8808 | 0.8993 | ruido |
| MAE ordinal | 0.1092 | 0.0821 | ruido | 0.1063 | 0.0870 | GANA FT |
| AUC macro | 0.9941 | 0.9960 | ruido | 0.9898 | 0.9924 | ruido |
| ECE (calibración) | 0.0421 | 0.0294 | ruido | 0.0452 | 0.0375 | ruido |
| errores 0↔2 | 7.2 | 6.2 | ruido | 5.6 | 5.0 | ruido |

**Criterio de "ruido", declarado antes de correr** (el mismo de la v8): una diferencia solo
cuenta si supera la suma de los desvíos de las dos fases. Con 207 imágenes de test, una imagen
vale 0.48 puntos de accuracy; festejar diferencias menores que el ruido entre semillas sería
inventar precisión que el dataset no soporta.

### Lectura

1. **El fine-tuning funcionó, y funcionó en los dos frameworks.** Accuracy y F1 macro superan
   el umbral de ruido en A y en B. La hipótesis declarada antes de correr —que soltar el
   último bloque le permitiría a los filtros de ImageNet adaptarse a texturas de pantalla que
   nunca vieron— queda sostenida.

2. **Todo lo demás mejora pero se queda en el ruido.** Es la lectura honesta: con 207 imágenes
   no hay resolución para afirmar que el fine-tuning mejora la calibración o el error ordinal,
   aunque las medias vayan en esa dirección en las 7 métricas. Se anota como tendencia, no
   como resultado.

3. **El fine-tuning también estabiliza.** El desvío de la accuracy de B baja de 0.0117 a
   0.0031 (casi 4×). Partir de una cabeza ya entrenada y mover el backbone con `lr = 1e-5`
   deja mucho menos margen a la semilla. Para un producto eso vale tanto como la media: un
   modelo reproducible es un modelo que se puede versionar.

### La brecha de dominio: el resultado propio de la v9

El **mismo** modelo, evaluado sobre los dos tests:

| | test FOTOS | test RENDERS | brecha |
|---|---:|---:|---:|
| A · fase 1 | 0.9256 | 0.8426 | **−0.0830** |
| A · fase 2 | 0.9478 | 0.8623 | **−0.0855** |
| B · fase 1 | 0.9208 | 0.8131 | **−0.1077** |
| B · fase 2 | 0.9372 | 0.8361 | **−0.1011** |

La brecha es **negativa en los cuatro casos**: los modelos aciertan entre 8 y 11 puntos *más*
sobre fotos que sobre renders.

Esto es lo contrario de lo que anticipaba `SUGERENCIAS_V9.md §2`, que advertía que los renders
—al ser más fáciles— inflarían la métrica. **La advertencia era correcta para el dataset que
existía cuando se escribió.** Con el dataset v9 el balance se invirtió, y por dos razones
concretas:

- El **78% del entrenamiento son fotos** (830 de 1069). El modelo se especializó donde tiene
  datos.
- La augmentation aplica simulación de foto de pantalla **con fuerza 1.0 sobre los renders**
  (moiré, glare, perspectiva, JPEG). Durante el entrenamiento el modelo casi no ve renders
  limpios: los ve disfrazados de foto.

**Para el producto esto es correcto**, porque la app solo recibe fotos. Pero hay que decirlo
explícitamente: **el modelo v9 ya no es un buen clasificador de PDFs**. Si mañana el producto
quisiera aceptar archivos subidos además de fotos, no alcanza con reusar este modelo.

### El riesgo de `SUGERENCIAS_V9.md §4`: la confusión 0 ↔ 2

Matriz acumulada del modelo A, fase 2 (5 semillas × 207 = 1035 casos):

```
                     predicho
                 0      1      2
real   0       300      3     22
       1        12    321      7
       2         9      1    360
```

El error extremo existe y es **asimétrico**: 22 casos de `0 → 2` contra 9 de `2 → 0`. O sea,
2.4 veces más frecuente **acusar de IA a una diapositiva humana** que dejar pasar una saturada.

Ese es exactamente el riesgo que anticipaba §4: el modelo ve un patrón de moiré en una
presentación humana limpia y lo lee como artefacto de generación. La mitigación aplicada
—aplicar la simulación de moiré **a las tres clases por igual**, para que deje de correlacionar
con la etiqueta— redujo el error de 5.2 a 4.4 casos por semilla, pero **no lo eliminó**.

En términos de producto: sobre 1035 fotos, 22 falsos positivos graves = **2.1%**. Para una
herramienta que puede terminar señalando el trabajo de un alumno, ese número es el que hay que
poner sobre la mesa, no la accuracy del 94.8%.

### A vs B: los dos frameworks se equivocan distinto

```
Modelo A (TensorFlow)              Modelo B (PyTorch)
   0     1     2                      0     1     2
 300     3    22                    305     1    19
  12   321     7                     14   318     8
   9     1   360                      6    17   347
```

A es mejor en accuracy global (0.9478 vs 0.9372), pero **B comete menos errores extremos**
(5.0 vs 6.2 por semilla) porque su error dominante es `2 → 1` (17 casos) en vez de `0 → 2`.
B se equivoca *entre vecinos del eje ordinal*, que es el error barato; A salta al extremo, que
es el caro.

Esto no es una curiosidad: **es la justificación empírica del selector de modelos de la app**.
Dos redes con errores correlacionados no aportan información al combinarse. Estas tienen
perfiles de error distintos, así que su desacuerdo señala imágenes genuinamente ambiguas.
Medido sobre `imagenes_a_probar/`: **7 de 10 con acuerdo unánime entre los tres modelos**, y
las 3 discordantes son las visualmente limítrofes.

---

## 2. Modelo C — qué pasa cuando las clases están desequilibradas

EfficientNet-B0 entrenada sobre un subconjunto deliberadamente desequilibrado **9.4 : 2.8 : 1**
(se conserva el 100% de la clase 0, el 30% de la 1 y el 10% de la 2). La clase que se vuelve
rara es `2_saturada_ia` **a propósito**: es la que el producto necesita detectar, y es el
escenario comercial realista.

Tres estrategias sobre **exactamente los mismos datos**, evaluadas sobre el **mismo test de
fotos balanceado** (3 semillas):

| estrategia | macro F1 | recall clase 2 | accuracy | AUC macro | ECE | errores 0↔2 |
|---|---:|---:|---:|---:|---:|---:|
| C0 · cross-entropy plana | 0.7820 | 0.6351 | 0.7826 | 0.9609 | **0.0561** | 19.7 |
| C1 · class weights | 0.8374 | 0.7342 | 0.8374 | 0.9662 | 0.0578 | 17.3 |
| C2 · Focal Loss + sampler | **0.8649** | **0.8198** | **0.8647** | 0.9649 | 0.1373 | **13.3** |

### Las tres cosas que dicen estos números

**(a) El desequilibrio no destruyó la capacidad de discriminar: destruyó dónde se pone la
frontera.**

El AUC macro es prácticamente el mismo en las tres (0.961 / 0.966 / 0.965). El AUC mide la
calidad del *ranking* — si el modelo le da más score a una clase 2 que a una clase 0, sin
importar el umbral. Ese ranking casi no se movió.

Lo que sí se movió, y mucho, es el recall: **0.635 → 0.820**. Es decir, el modelo desequilibrado
*sabía* distinguir la clase rara; lo que hacía mal era **decidir**, porque el argmax hereda la
frontera que la pérdida le enseñó, y una pérdida plana sobre datos 9:1 pone la frontera donde
conviene a la clase mayoritaria.

Esta distinción es la que explica por qué las correcciones funcionan: no le enseñan al modelo a
ver mejor, le corrigen dónde corta.

**(b) La accuracy fue inútil como diagnóstico, pero no de la forma esperada.**

Se anticipaba el patrón clásico "accuracy alta, recall cerca de cero". No pasó: la accuracy de
C0 (0.7826) y su recall (0.6351) bajaron *juntos*. La razón es que el test **no** está
desequilibrado, así que la accuracy no puede esconderse detrás de la clase mayoritaria.

Dónde sí se ve la patología es en la **matriz de confusión** de C0:

```
                     predicho
                 0      1      2
real   0       192      0      3
       1        45    153      6
       2        56     25    141      <- 56 diapositivas saturadas clasificadas como "sin IA"
```

56 de 222 fotos de la clase 2 fueron empujadas a la clase 0 — la mayoritaria. C2 baja esos 56
a 16. **La lección metodológica es que un solo escalar no diagnostica un desequilibrio: hay que
mirar la matriz.**

**(c) Focal Loss mejora la decisión y ARRUINA la calibración. Este es el hallazgo incómodo.**

El ECE salta de 0.0561 (C0) a **0.1373** (C2) — se multiplica por 2.4. C2 acierta más pero
**miente sobre cuánto sabe**: cuando dice 90%, no acierta el 90%.

La causa es directa: el `WeightedRandomSampler` hace que el modelo entrene sobre una
distribución de clases artificial (balanceada) distinta de la real, y las probabilidades que
aprende son las de esa distribución inventada. Focal Loss agrega lo suyo al desinflar los
ejemplos fáciles.

**Consecuencia concreta para el producto:** el aviso de "confianza baja" de la app usa un
umbral de 0.60, y ese umbral es confiable en A y B (ECE ≈ 0.03) pero **no** en C. Está anotado
en el plan de acción.

---

## 3. Los tres modelos, lado a lado

| | A · TensorFlow.js | B · ONNX | C · ONNX |
|---|---|---|---|
| arquitectura | MobileNetV3-Small | MobileNetV3-Small | EfficientNet-B0 |
| entrenamiento | balanceado, FT 2 fases | ídem A | desequilibrado + Focal Loss |
| accuracy (fotos) | **0.9478** | 0.9372 | 0.8647 |
| F1 macro | **0.9475** | 0.9374 | 0.8649 |
| recall clase 2 | **0.9730** | 0.9378 | 0.8198 |
| ECE | **0.0294** | 0.0375 | 0.1373 |
| peso desplegado | 4.1 MB | 4.6 MB | 16.0 MB |
| motor | WebGL (GPU) | WASM | WASM |

C es claramente el más débil — **y está en el catálogo a propósito**, con sus métricas reales a
la vista en el selector. Un usuario que lo elige ve que rinde menos. Su función en el producto
no es ganar: es aportar un voto de una arquitectura no emparentada cuando A y B discrepan, y
ser la demostración en vivo de qué le hace el desequilibrio a un modelo.

### Verificación de la exportación

| modelo | formato | max&#124;diferencia&#124; vs original | clases coincidentes |
|---|---|---:|---|
| B | ONNX opset 17 | 3.15 × 10⁻⁵ | 12/12 |
| C | ONNX opset 17 | 6.56 × 10⁻⁶ | 12/12 |

Medido sobre 12 fotos reales del test, no sobre ruido aleatorio. `export/exportar_onnx.py`
**aborta** si la diferencia supera 10⁻⁴: exportar es silencioso y produce un archivo aunque el
grafo esté mal, así que sin esta verificación el error aparecería recién en el teléfono.

---

## 4. Plan de acción

Ordenado por relación impacto / costo.

### Prioridad alta

**1. Atacar el falso positivo `0 → 2` (2.1% de las fotos).**
Es el error que más daña al usuario: acusar de IA a una diapositiva humana.
*Acción:* fotografiar 80–120 diapositivas **humanas** en las pantallas que más moiré generan
(paneles con rejilla fina, TV) e incorporarlas a `0_sin_ia`. Es el error con la causa más
identificada del proyecto y se ataca con datos, no con hiperparámetros.
*Medición:* `sin_ia_como_saturada` en `resultados_v9.json` tiene que bajar de 4.4 a < 2.0 por
semilla.

**2. Recalibrar el modelo C antes de confiar en su confianza.**
*Acción:* aplicar *temperature scaling* sobre un conjunto de calibración apartado (una sola
temperatura escalar ajustada por NLL: no cambia el argmax, solo aplana el softmax).
*Medición:* ECE de C por debajo de 0.06, alineado con A y B.
*Mientras tanto:* la app debería subir el umbral de "confianza baja" a 0.75 cuando el modelo
activo es C. **Está identificado y no implementado.**

**3. Ampliar el test.** 207 fotos hacen que una imagen valga 0.48 puntos, y por eso 7 de las 11
métricas de la comparación A/B quedaron en "ruido".
*Acción:* llegar a ~500 fotos de test, priorizando pantallas y condiciones de luz no cubiertas.

### Prioridad media

**4. Decidir explícitamente si el producto acepta archivos además de fotos.**
Hoy el modelo rinde 8–10 puntos peor sobre renders. Si se quiere soportar "subir un PDF", hay
dos caminos: entrenar una cabeza separada para ese dominio, o bajar la fuerza de la simulación
de pantalla sobre los renders (`aug_pantalla.augmentar`, hoy 1.0) y aceptar perder algo en
fotos. **Es una decisión de producto, no técnica.**

**5. Cuantificar el aporte real del recorte de pantalla con OpenCV.**
`OpenCvService.prepararParaModelo()` detecta el marco del monitor y corrige la perspectiva.
Es defendible en teoría, pero **no está medido**: falta un experimento que compare accuracy con
y sin recorte sobre el mismo lote de fotos desencuadradas. Si no aporta, es complejidad que se
puede sacar.

**6. Ensamble de los tres como modo por defecto.**
El botón "Comparar los 3" ya existe. Falta medir si el voto mayoritario supera a A solo sobre
el test de fotos. Con perfiles de error distintos (§1) es plausible, pero **no está medido** y
cuesta 3× en latencia.

### Prioridad baja

**7. Cuantización int8 del modelo C.** 16 MB es mucho para datos móviles. ONNX Runtime soporta
cuantización dinámica post-entrenamiento; bajaría a ~4 MB. Verificar la pérdida de accuracy
antes de adoptarlo.

**8. Reproducir el fracaso clásico del desequilibrio.** Con 10:3:1 el recall bajó a 0.635, no a
~0. Para el material didáctico, un experimento con el 2% de la clase 2 mostraría el patrón
"accuracy alta, recall ≈ 0" de forma más nítida.

---

## 5. Limitaciones que hay que declarar

1. **Test chico.** 207 fotos. Los intervalos son anchos y está reflejado en los veredictos de
   "ruido".
2. **Un solo fotógrafo, un puñado de pantallas.** Las fotos se capturaron en una sesión. La
   variedad de paneles, cámaras y condiciones de luz del mundo real es mucho mayor.
3. **Las etiquetas son un juicio, no un hecho.** El proyecto mide *densidad de artefactos
   visuales de IA*, no procedencia. La frontera `1_rastro_ia` / `2_saturada_ia` es
   genuinamente difusa, y parte del error residual es desacuerdo de etiquetado, no error del
   modelo.
4. **El modelo C tiene 3 semillas, no 5.** Su comparación interna es pareada y válida, pero sus
   barras de error son más anchas que las de A y B.
5. **La brecha de dominio se mide contra 61 renders.** Es suficiente para ver el signo y el
   orden de magnitud, no para afirmar el valor exacto.
