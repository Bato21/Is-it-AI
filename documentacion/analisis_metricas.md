# Análisis de métricas — entrega 2

> **Cómo se generó:** los números de este documento son reales, de corridas con
> `seed=123` sobre `dataset/` (300 imgs, 100/clase) y `dataset_desbalanceado/` (v5).
> Las matrices v4 se reproducen sin re-entrenar con
> `python documentacion/reporte_metricas.py` (en el venv de cada framework). Los logs
> completos están en `TensorFlow/v*/Resultado_*.txt` y `PyTorch/v*/Resultado_*.txt`.
> Corridas: PyTorch v3/v4 (2026-06, Bato); TF v3/v4 y ambos v5 (2026-07).
>
> **La prosa interpretativa y el plan de acción los completa Bato** (bloques
> `<!-- COMPLETAR: Bato -->`): el profe avisó que la interrogación oral sobre estas
> métricas es individual, así que el documento tiene que ser defendible línea por línea.
>
> **Convención de matriz:** filas = clase real, columnas = clase predicha. Clases en
> orden `0_sin_ia`, `1_rastro_ia`, `2_saturada_ia` (el eje ordinal de saturación).

---

## 0. Tabla resumen (todos los números reales)

| Modelo | val_acc | esquina 0↔2 | recall clase 2 | nota |
|--------|:------:|:-----------:|:--------------:|------|
| PyTorch v3 (CNN baseline) | 83.3% | **4 errores** (1×0→2, 3×2→0) | 18/22 | confusión extrema persiste tras converger |
| PyTorch v4 (MobileNetV3)  | **90.0%** | **0** | **22/22** | features preentrenadas separan los extremos |
| TF v3 (CNN baseline)      | 88.3% | 0 | 18/18 | error solo entre vecinos (0↔1) |
| TF v4 (MobileNetV3)       | 80.0% | 2 (1×0→2, 1×2→0) | 17/18 | tocó el techo de 60 épocas (val_loss aún bajaba) |
| TF v5 sin pesos           | 71.4% | 0 | 6/7 | recall clase 1 colapsa: 3/10 |
| TF v5 con pesos           | 80.0% | 1 (1×0→2) | 6/7 | recall clase 1 se recupera: 7/10 |
| PyTorch v5 sin pesos      | 82.9% | 2 (2×2→0) | 6/8 | recall clase 1: 7/10 |
| PyTorch v5 con pesos      | 85.7% | 1 (1×2→0) | 6/8 | recall clase 1: 8/10 |

> **Salvedad de lectura (vale para TODO el documento):** la validación son ~60 imágenes
> (v3/v4) o ~35 (v5), sin estratificar. Con soportes así de chicos, **1-2 imágenes mueven
> varios puntos de %**. Leer los recalls de la clase minoritaria en NÚMEROS ABSOLUTOS.

---

## 1. Qué cambió de v3 → v4 (misma data, distinto modelo)

Mismas imágenes, mismo split, mismo early stopping: lo único que cambia entre v3 y v4 es
el modelo (CNN desde cero → MobileNetV3-Small preentrenada, backbone congelado).

**PyTorch — el resultado limpio (el titular del proyecto):**

```
PyTorch v3 (83.3%)          PyTorch v4 (90.0%)
        0   1   2                   0   1   2
   0 [ 14   3   1 ]            0 [ 14   4   0 ]
   1 [  1  18   1 ]            1 [  1  18   1 ]
   2 [  3   1  18 ]            2 [  0   0  22 ]
   -> 0↔2: 4 errores          -> 0↔2: 0 errores ; clase 2 recall 22/22
```

La esquina 0↔2 pasa de **4 errores a 0** con las MISMAS imágenes. Las features de ImageNet
separan lo que la CNN propia confundía en los extremos: el límite era la **capacidad de
representación del modelo, no las etiquetas** (las etiquetas eran aprendibles).

**TensorFlow — el resultado que NO calza con esa narrativa (hay que explicarlo):**

```
TF v3 (88.3%)               TF v4 (80.0%)
        0   1   2                   0   1   2
   0 [ 21   3   0 ]            0 [ 21   2   1 ]
   1 [  4  14   0 ]            1 [  8  10   0 ]
   2 [  0   0  18 ]            2 [  1   0  17 ]
   -> 0↔2: 0 ; acc 88.3%      -> 0↔2: 2 ; acc 80.0% ; recall clase 1 = 10/18
```

En TF la v4 **no supera** a la v3 en este split: la CNN baseline ya tenía la esquina 0↔2
limpia y más accuracy. Dos pistas objetivas para interpretarlo (no inventadas):
- **TF v4 cortó en la época 60/60** (el techo), con `val_loss` todavía bajando → la cabeza
  quedó *sub-entrenada* (el early stopping nunca disparó). PyTorch v4 convergió en la 12/20.
- **El split no es el mismo entre frameworks**: `image_dataset_from_directory` (TF) y
  `random_split` (PyTorch) parten distinto, así que evalúan sobre imágenes distintas.

<!-- COMPLETAR: Bato -->
<!--
  Preguntas guía para tu prosa:
  1. ¿El titular "features preentrenadas > CNN propia" se sostiene con AMBOS frameworks,
     o es un resultado de PyTorch que en TF queda tapado por el sub-entrenamiento de la v4
     (techo de épocas) y el split distinto? ¿Cómo lo defendés sin exagerar?
  2. Si TF v4 tocó el techo de 60 épocas con val_loss bajando, ¿qué corrida repetirías
     (subir EPOCHS) antes de sacar conclusiones del lado TF?
  3. ¿Qué te dice que la CNN baseline (v3) ya rinda tan bien sobre renders limpios? ¿Cuánto
     de esto es "el problema es fácil en renders limpios" vs "el modelo es bueno"?
-->

---

## 2. Lectura por clase: precision vs recall según el costo del negocio

Reporte por clase de los dos modelos v4 (calculado a mano desde la matriz):

```
PyTorch v4                              TensorFlow v4
clase          prec  recall  f1  sop    clase          prec  recall  f1  sop
0_sin_ia      0.933  0.778 0.848 18     0_sin_ia      0.700  0.875 0.778 24
1_rastro_ia   0.818  0.900 0.857 20     1_rastro_ia   0.833  0.556 0.667 18
2_saturada_ia 0.957  1.000 0.978 22     2_saturada_ia 0.944  0.944 0.944 18
macro         0.903  0.893 0.894        macro         0.826  0.792 0.796
accuracy      0.900                     accuracy      0.800
```

Para el uso del producto (auditoría de huella de IA), el error **caro** es el falso
"sin rastro" — una diapo saturada clasificada como 0 (**2→0**): deja pasar justo lo que se
quería detectar. En PyTorch v4 ese error es **0** (recall clase 2 = 22/22); en TF v4 es
**1** (recall clase 2 = 17/18). El recall de la clase 2 es la métrica de negocio a vigilar.

<!-- COMPLETAR: Bato -->
<!--
  Preguntas guía:
  1. ¿Qué preferís que el modelo minimice: falsos "sin rastro" (2→0, deja pasar IA) o
     falsos positivos (0→2, acusa de más)? Justificá según a quién le vendés el producto.
  2. TF v4 tiene recall de clase 1 bajo (10/18): ¿es un problema del modelo o del soporte
     chico + frontera 1↔2 ambigua (la más frágil del esquema)?
-->

---

## 3. Efecto del desbalance y de los pesos de clase (v5, A/B)

v5 = mismo modelo que v4, entrenado sobre `dataset_desbalanceado/` (100/50/25). Dos
variantes en una corrida: sin corrección vs. con pesos `w_c = n_total/(n_clases·n_c)`.

```
TensorFlow v5                            PyTorch v5
sin pesos (71.4%)  con pesos (80.0%)     sin pesos (82.9%)  con pesos (85.7%)
   0   1   2          0   1   2             0   1   2          0   1   2
0[16   2   0]      0[15   2   1]         0[16   1   0]      0[16   1   0]
1[ 7   3   0]      1[ 3   7   0]         1[ 3   7   0]      1[ 2   8   0]
2[ 1   0   6]      2[ 1   0   6]         2[ 2   0   6]      2[ 1   1   6]
recall clase 1:    recall clase 1:       recall clase 1:    recall clase 1:
   3/10 (colapsa)     7/10 (recupera)       7/10               8/10
```

- **TF v5 = efecto de libro:** sin corrección el modelo deriva hacia la clase mayoritaria
  (0) y el recall de la clase 1 **colapsa (3/10)** aunque la accuracy global se mantenga
  decente; con pesos, el recall minoritario **se recupera (7/10)** y sube la accuracy.
- **PyTorch v5 = efecto sutil:** las features preentrenadas ya hacen la clase 2 muy
  separable (6/8 en ambas variantes), así que la corrección mueve poco (recall clase 1
  7→8). **Un efecto chico también es un hallazgo**: el desbalance duele menos cuando las
  clases son separables en el espacio de features.

<!-- COMPLETAR: Bato -->
<!--
  Preguntas guía:
  1. ¿Por qué el mismo experimento da un efecto FUERTE en TF y SUTIL en PyTorch? (pista:
     convergencia + separabilidad de features + split). ¿Qué conclusión honesta sacás?
  2. Los pesos suben el recall minoritario a costa de la precision. ¿Ese trade-off te
     conviene para el producto de auditoría? ¿O preferirías más datos de la clase 2?
  3. La clase 2 en val tiene ~7-8 casos: ¿cuánta confianza le das a su recall?
-->

---

## 4. Límites honestos

- **Validación chica y sin estratificar:** 60 imgs (v3/v4), ~35 (v5). Los % son ruidosos;
  por eso todo el documento insiste en leer números absolutos.
- **Todo sobre renders limpios:** ni una foto real de teléfono en el circuito de métricas.
  Los números son un **techo optimista**; el *domain gap* (moiré, reflejos, perspectiva)
  no está medido. Los scripts de cámara (`modelos/v4/camara_*.py`, tecla `s`) son el primer
  banco de pruebas para empezar a medirlo.
- **Split distinto entre frameworks:** TF y PyTorch no evalúan sobre las mismas imágenes,
  así que comparar sus % lado a lado tiene un asterisco.
- **Clase 2, la más cara de producir**, fija el techo de tamaño del dataset.

<!-- COMPLETAR: Bato -->
<!--
  Preguntas guía:
  1. ¿Cuál de estos límites es el más urgente de atacar para que los números sean creíbles
     de cara al despliegue? (candidato fuerte: el domain gap sin medir).
  2. ¿Vale la pena estratificar el split / pasar a train-val-test en disco antes de seguir
     comparando frameworks?
-->

---

## 5. Plan de acción priorizado

> Regla de la casa (y del profe): **más datos primero, técnica después.**

Orden propuesto (Bato prioriza y justifica):

1. **Medir el domain gap:** fotografiar diapos con `camara_*.py`, guardar capturas y
   ver cuánto cae el modelo fuera de los renders limpios. Sin esto, los números son teóricos.
2. **Más datos de la clase 2** (la que fija el techo) y de la frontera 1↔2 (la más frágil).
3. **Cerrar el asterisco de TF v4:** subir el techo de épocas (cortó en 60/60 sub-entrenada)
   y/o estratificar el split, para una comparación v3↔v4 justa del lado TF.
4. **Recién después, técnica:** fine-tuning en dos fases (v6) o búsqueda de hiperparámetros
   (v7), que ya están esbozados como opcionales.

<!-- COMPLETAR: Bato -->
<!--
  Preguntas guía:
  1. Ordená estos 4 pasos con TU criterio y defendé el orden. ¿Por qué "más datos" antes
     que "más técnica"?
  2. ¿Qué UNA acción concreta cerrarías en la interrogación si te piden "¿y ahora qué?".
-->
