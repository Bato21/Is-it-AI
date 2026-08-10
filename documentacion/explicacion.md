# Explicación del proyecto — guía completa de lo que se hizo

*Guía de lectura del repositorio **Is-it-AI**. Recorre, versión por versión y separado por
framework, qué se agregó en cada paso, **cómo se llama cada función**, qué hace y qué
resultado dio. Escrita al cierre de la **v7**, con números de corridas reales.*

> **Cómo usar este documento.** Si venís a entender el proyecto de cero, leé §1 y §2.
> Si venís a defenderlo oralmente, leé §3 (resultados) y §8 (contrastes de framework).
> Si venís a tocar código, andá directo a §5 (TensorFlow), §6 (PyTorch) o §7 (módulos
> compartidos), que son el índice de funciones.
>
> Documentos hermanos: `avances_proyecto.md` (el porqué conceptual y el histórico),
> `analisis_metricas.md` (el análisis para la interrogación oral),
> `DetalleProyecto.md` (resumen ejecutivo del repo), `caso_comercial.md` (uso comercial).

---

## 1. Qué es el proyecto, en una página

Un clasificador que, a partir de la **foto de una diapositiva**, estima su **densidad de
artefactos visuales de IA** en tres niveles ordinales:

| Índice | Carpeta | Clase | Qué se ve |
|:---:|---|---|---|
| 0 | `0_sin_ia/` | Sin rastro | Deck humano o plantilla clásica. |
| 1 | `1_rastro_ia/` | Hay rastro | Densidad intermedia (una imagen generada en un deck humano). |
| 2 | `2_saturada_ia/` | Saturada | Firma de generador end-to-end (estética Gamma/Tome). |

**La decisión conceptual que ordena todo:** el proyecto **no mide procedencia ni proceso**,
mide **huella visual**, porque es lo único que una CNN puede aprender de una imagen. Las
clases se definen por **aspecto**, no por cómo se fabricó la diapositiva.

**El método de trabajo:** un **cambio conceptual por versión**. Cada versión modifica una
sola cosa respecto de la anterior, para que el A/B sea limpio y se pueda atribuir la
diferencia a algo concreto. No se amontonan técnicas: se lee el diagnóstico y se responde
a lo que se ve.

**La implementación es dual y espejo:** todo existe en TensorFlow/Keras y en PyTorch, con
los contrastes entre frameworks documentados donde aparecen (§8).

---

## 2. Mapa del repositorio

```
Is-it-AI/
├── dataset/                    (fuera de git) 313 imágenes: 108 / 94 / 111
├── dataset_desbalanceado/      (fuera de git) subset 100/50/25 para la v5
│
├── TensorFlow/  v1..v7/        entrenamiento — un script por versión
│   └── export_tfjs/            exportación a TensorFlow.js (venv aparte)
├── PyTorch/     v1..v7/        entrenamiento — espejo exacto
│
├── modelos/                    scripts de CONSUMO (no entrenan, solo predicen)
│   ├── v2/  v4/  v6/           consumo por versión + cámara OpenCV en v4
│
├── documentacion/
│   ├── particion_datos.py      [v7] partición canónica COMPARTIDA por los 2 frameworks
│   ├── metricas.py             [v7] métricas COMPARTIDAS por los 2 frameworks
│   ├── particion.json          [v7] el manifiesto de la partición (versionado en git)
│   ├── deck_a_imagenes.py      obtención: PDF/PPTX -> PNG por diapositiva
│   ├── verificar_dataset.py    chequeo del dataset antes de entrenar
│   ├── crear_datos_prueba.py   datos sintéticos para probar el flujo sin dataset real
│   ├── crear_subset_desbalanceado.py   genera el subset 100/50/25 de la v5
│   ├── dividir_dataset.py      utilidad de split en disco
│   ├── reporte_metricas.py     reporte por clase de los v4 sin re-entrenar
│   ├── explicacion.md          ESTE DOCUMENTO
│   ├── avances_proyecto.md · analisis_metricas.md · caso_comercial.md · guia_etiquetado.md
│
├── web/                        smoke test del modelo en el navegador (TF.js)
├── imagenes_a_probar/          imágenes sueltas para probar el consumo
└── README.md · DetalleProyecto.md
```

**Convención de nombres:** `TensorFlow/vN/0N_scripts.py` y `PyTorch/vN/0N_scripts.py` son
espejos. Cada carpeta de versión guarda su `Resultado_N.txt` (log completo de la corrida
real), sus figuras `Figure_*.png` y su modelo entrenado (los modelos están fuera de git).

**Convención de matriz de confusión, en TODO el proyecto:**
`cm[i, j]` = casos cuya clase **real** es `i` y fueron **predichos** como `j`.
Filas = real, columnas = predicho, en orden `0_sin_ia` / `1_rastro_ia` / `2_saturada_ia`.

---

## 3. El arco de versiones y sus resultados

### 3.1 Qué cambió en cada versión

| Ver | Cambio conceptual (uno solo) | Qué pregunta responde |
|:---:|---|---|
| v1 | CNN desde cero, sin nada | ¿Se ve el sobreajuste? |
| v2 | + Data augmentation domain-aware | ¿La augmentation lo mata? |
| v3 | + Early stopping (converger de verdad) | La confusión residual, ¿era ruido o real? |
| v4 | CNN propia → MobileNetV3-Small preentrenada | ¿El límite era el modelo o las etiquetas? |
| v5 | Dataset desbalanceado, sin/con pesos de clase | ¿Cuánto duele el desbalance y cuánto lo corrigen los pesos? |
| v6 | Hiperparámetros a ojo → **Optuna** | ¿Buscar hiperparámetros mejora el modelo? |
| **v7** | **Protocolo de evaluación honesto** | **¿Cuánto de lo reportado hasta acá era real?** |

### 3.2 Resultados — versiones v1 a v6

> Estos números son de un protocolo **viejo**: split 80/20 sin estratificar, **sin test
> apartado**, una sola semilla, y splits **distintos** entre TF y PyTorch. La v7 los
> re-mide bien; ver §3.3 antes de citarlos.

**TensorFlow / Keras**

| Ver | Resultado (val) | Matriz | Lectura |
|:---:|---|---|---|
| v1 | train 1.00 / val 0.97 · loss 0.01 vs 0.10 | — | El sobreajuste se ve en la **loss**, no en la accuracy. |
| v2 | train 0.77 / val 0.75 | `[[20,4,0],[4,13,1],[3,3,12]]` | Mató el sobreajuste pero quedó **sub-entrenado**. |
| v3 | **88.3%** | `[[21,3,0],[4,14,0],[0,0,18]]` | Convergió; esquina 0↔2 **limpia**. |
| v4 | **80.0%** | `[[21,2,1],[8,10,0],[1,0,17]]` | Cortó en la época 60/60 (**sub-entrenada**); recall clase 1 = 10/18. |
| v5 | 71.4% → **80.0%** (sin/con pesos) | `[[16,2,0],[7,3,0],[1,0,6]]` → `[[15,2,1],[3,7,0],[1,0,6]]` | Sin pesos el recall de clase 1 **colapsa (3/10)**; con pesos **se recupera (7/10)**. |
| v6 | **93.3%** (best trial 96.7%) | `[[23,0,1],[2,15,1],[0,0,18]]` | 2·128·drop0.2·adam·lr4.8e-3. Recall clase 2 = 18/18. |

**PyTorch**

| Ver | Resultado (val) | Matriz | Lectura |
|:---:|---|---|---|
| v1 | train 0.996 / val 0.83 · loss 0.03 vs 0.41 | — | Sobreajuste visible en la loss. |
| v2 | train 0.71 / val 0.72 | `[[12,6,0],[4,16,0],[1,6,15]]` | Regularizado pero sub-entrenado. |
| v3 | **83.3%** | `[[14,3,1],[1,18,1],[3,1,18]]` | Tras converger **persisten 4 errores 0↔2** → no era ruido. |
| v4 | **90.0%** | `[[14,4,0],[1,18,1],[0,0,22]]` | Esquina 0↔2 **en cero**; clase 2 recall **22/22**. |
| v5 | 82.9% → **85.7%** (sin/con pesos) | `[[16,1,0],[3,7,0],[2,0,6]]` → `[[16,1,0],[2,8,0],[1,1,6]]` | Efecto **sutil**: las features preentrenadas ya separan bien. |
| v6 | **93.3%** | `[[15,2,1],[1,19,0],[0,0,22]]` | 2·256·drop0.45·rmsprop·lr9.2e-4. Recall clase 2 = 22/22. |

### 3.3 Resultados — v7 (protocolo honesto)

Test **apartado** de 63 imágenes que la búsqueda de hiperparámetros nunca vio, **5 semillas**,
**mismo split** para los dos frameworks. Media ± desvío:

| Métrica | | TensorFlow v7 | PyTorch v7 |
|---|:---:|---|---|
| accuracy | ↑ | **0.8000 ± 0.0819** | **0.7905 ± 0.0119** |
| F1 macro | ↑ | 0.7950 ± 0.0849 | 0.7870 ± 0.0081 |
| recall clase 2 | ↑ | 0.7818 ± 0.2120 | 0.7182 ± 0.1012 |
| errores 0↔2 | ↓ | 2.8 ± 2.2 | 3.6 ± 0.8 |
| QWK (ordinal) | ↑ | 0.7566 ± 0.1410 | 0.7198 ± 0.0372 |
| MAE ordinal | ↓ | 0.2444 ± 0.1165 | 0.2667 ± 0.0211 |
| AUC macro (OvR) | ↑ | 0.9312 ± 0.0176 | 0.9392 ± 0.0056 |
| ECE (calibración) | ↓ | 0.1681 ± 0.0650 | 0.1304 ± 0.0107 |

Accuracy de la validación cruzada durante la búsqueda: **TF 0.8800**, **PyTorch 0.8482**.

**Los cuatro hallazgos de la v7 — esto es lo que hay que saber defender:**

1. **El 93.3% de la v6 era, en buena parte, sesgo de selección.** Medido con test
   apartado, el mismo modelo da **~0.79–0.80**. La caída es de **−13 a −14 puntos** en
   ambos frameworks, de forma independiente. Optuna probó 30 combinaciones maximizando
   una validación de 60 imágenes y después se reportó esa misma validación: el ganador de
   30 intentos sobre un examen chico gana en parte por mérito y en parte por suerte.

2. **TensorFlow y PyTorch son equivalentes.** Sobre el mismo split: 0.8000 vs 0.7905, una
   diferencia de ~0.6 imágenes de 63, muy dentro del ruido. **La brecha histórica de
   "PyTorch v4 90% vs TF v4 80%" era el split, no el framework.** El asterisco que
   `analisis_metricas.md` §4 arrastraba desde la entrega 2 queda cerrado.

3. **El desvío entre semillas de TensorFlow es enorme: ±0.0819** (de 0.6667 a 0.8889 con
   los MISMOS hiperparámetros, cambiando solo la semilla). PyTorch es mucho más estable
   (±0.0119). Una sola corrida de TF podía reportar 88.9% o 66.7% con igual honestidad
   aparente. **Esto solo, justifica el multi-semilla.**

4. **El error caro no desapareció.** La esquina 0↔2 sigue viva (~3 de 63) y el recall de
   la clase 2 bajó a ~0.72–0.78, lejos del 22/22 que celebraba la v4. Ese 22/22 era el
   split favorable, no una propiedad del modelo.

> **Salvedad honesta:** el dataset creció de 300 a **313 imágenes** (108/94/111) entre la
> v6 y la v7, así que la comparación v6↔v7 no es 100% a data constante. Pero 13 imágenes
> no explican 14 puntos, y los dos frameworks cayeron lo mismo por separado.

---

## 4. Cómo correr todo

```bash
# 0) Verificar el dataset
python documentacion/verificar_dataset.py

# 1) Generar la partición canónica (UNA vez; los dos frameworks leen este archivo)
python documentacion/particion_datos.py
python documentacion/particion_datos.py --mostrar     # ver la que ya existe

# 2) Entrenar (venv tensorflow-ia para TF, frameworks-ia para PyTorch)
python TensorFlow/v7/07_scripts.py
python PyTorch/v7/07_scripts.py

# 3) Consumo y cámara
python modelos/v4/modelotf_v4.py ; python modelos/v4/camara_pt.py
python modelos/v6/modelopt_v6.py

# 4) Export a TensorFlow.js + smoke test (venv aparte)
python TensorFlow/export_tfjs/exportar_tfjs.py
cd web && python -m http.server        # abrir http://localhost:8000/
```

---

## 5. TensorFlow / Keras — versión por versión

### v1 — `TensorFlow/v1/01_script.py` · baseline que sobreajusta a propósito

**Sin funciones propias**: es un script lineal, a propósito, para que se lea de arriba a
abajo como un tutorial.

| Qué hace | Cómo |
|---|---|
| Carga de datos | `tf.keras.utils.image_dataset_from_directory(validation_split=0.2, subset=...)` |
| Modelo | `Sequential`: `Rescaling(1/255)` → `Conv2D(16)` → `MaxPool` → `Conv2D(32)` → `MaxPool` → `Flatten` → `Dense(64)` → `Dense(3)` (~3.79 M parámetros, el 99% en la primera densa) |
| Verificación | `print("Orden de clases:", train_ds.class_names)` — el chequeo del gotcha alfabético, presente en TODO script que carga datos |
| Salida | `Figure_1.png` (curvas), `Resultado_1.txt` |

**Resultado:** train 1.00 / val 0.97, loss 0.0099 vs 0.1026. El sobreajuste **está** pero
se ve en la **loss**, no en la accuracy: las clases son muy separables en renders limpios.

---

### v2 — `TensorFlow/v2/02_scripts.py` · data augmentation domain-aware

**Nuevo:** un bloque `data_augmentation` de capas Keras dentro del modelo, la gráfica de
**loss** (no solo accuracy) y la **matriz de confusión** con `tf.math.confusion_matrix`.

La augmentation es `RandomRotation(0.05)` (±18°), `RandomZoom(0.1)`, `RandomBrightness(0.2)`,
`RandomContrast(0.2)`. **Sin flip horizontal, a propósito:** una diapositiva espejada tiene
el texto en espejo, algo que no existe en el mundo real. Las transformaciones tienen que ser
realistas *para el dominio*.

**Resultado:** train 0.77 / val 0.75, loss 0.53 vs 0.55 (pegadas = bien regularizado). La
accuracy "bajó" respecto de la v1, pero eso es **honestidad, no retroceso**: el 96.7% era
memorización. El modelo quedó **sub-entrenado** (las curvas seguían subiendo en la época 10).

> **La lección de método más importante del proyecto:** el paso siguiente **no** es agregar
> más regularización. No sobreajustamos, sub-entrenamos. La receta no es "sumar técnicas",
> es leer el estado y responder a lo que se ve.

---

### v3 — `TensorFlow/v3/03_scripts.py` · early stopping

**Nuevo:** `tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
restore_best_weights=True)` y techo de 60 épocas en vez de 10 fijas. Ninguna técnica nueva:
solo dejar que el modelo **converja**.

**Resultado: 88.3%**, matriz `[[21,3,0],[4,14,0],[0,0,18]]`. La esquina 0↔2 quedó **limpia**:
el error es solo entre vecinos. Del lado TF, la confusión extrema **era** falta de
entrenamiento.

---

### v4 — `TensorFlow/v4/04_scripts.py` · transfer learning MobileNetV3-Small

**Nuevo:** backbone `tf.keras.applications.MobileNetV3Small(include_top=False,
weights="imagenet")` con `base_model.trainable = False`, entrada 224×224 y el
**preprocesamiento ImageNet DENTRO del modelo** (`include_preprocessing=True` por defecto).

> **Decisión de diseño clave:** que la normalización viva dentro del `.keras` significa que
> el modelo recibe imágenes **0-255 crudas** y normaliza solo. Eso hace que el consumo (y el
> navegador, vía TF.js) no tenga que replicar nada. En PyTorch es al revés (§8).

**Resultado: 80.0%**, matriz `[[21,2,1],[8,10,0],[1,0,17]]`. **No superó a la v3**, y hay dos
razones objetivas documentadas: cortó en la **época 60/60** (el techo) con `val_loss` todavía
bajando → sub-entrenada, y el split difería del de PyTorch. La v7 confirma que el problema
era metodológico, no del framework.

---

### v5 — `TensorFlow/v5/05_scripts.py` · clases desbalanceadas + pesos

Entrena sobre `dataset_desbalanceado/` (100/50/25) y corre **dos variantes en una sola
corrida** para el A/B.

| Función | Qué hace |
|---|---|
| `contar_labels(ds)` | Cuenta casos por clase recorriendo un `tf.data.Dataset` (necesario para calcular los pesos sobre TRAIN, no sobre el total). |
| `reporte_por_clase(cm, nombres)` | Precision / recall / f1 / soporte por clase, calculado **a mano** desde la matriz. Primera aparición; se repite en v6 y se unifica en la v7. |
| `construir_modelo()` | Arma el MobileNetV3 congelado + cabeza. Se llama dos veces (una por variante) para que las dos partan de cero. |
| `matriz_confusion(model)` | Matriz de confusión sobre validación. |
| `entrenar_variante(usar_pesos)` | El corazón del A/B: entrena con o sin `class_weight` y devuelve las métricas. |

Fórmula de pesos: `w_c = n_total / (n_clases · n_c)`. En Keras se aplica con un argumento de
`fit()`: `class_weight={0: w0, 1: w1, 2: w2}`.

**Resultado: 71.4% → 80.0%.** El **efecto de libro**: sin corrección el recall de la clase 1
**colapsa (3/10)** aunque la accuracy global se vea decente; con pesos **se recupera (7/10)**.

---

### v6 — `TensorFlow/v6/06_scripts.py` · Optuna

**Nuevo:** búsqueda automática de hiperparámetros de la **cabeza** con Optuna (TPESampler +
MedianPruner), y el **pre-cómputo de embeddings** que la hace viable.

| Función | Qué hace |
|---|---|
| `construir_extractor()` | MobileNetV3 congelada + `GlobalAveragePooling2D` → salida de **576 dimensiones**. |
| `embeddings_de(ds)` | Pasa el dataset por el extractor **una sola vez** y guarda los vectores 576-d. |
| `construir_cabeza(params)` | Arma la MLP que Optuna propone (n capas, units, dropout, optimizador, lr) y la compila. |
| `objective(trial)` | La función objetivo: propone hiperparámetros, entrena la cabeza y devuelve el mejor `val_accuracy`. |
| `ejecutar_estudio()` | Crea el `study` (TPE + MedianPruner) e imprime el progreso trial a trial. |
| `reentrenar_mejor(best_params)` | Reentrena con los mejores hiperparámetros y **une extractor + cabeza** en un `.keras` de punta a punta. |
| `plot_historia(study, salida)` | `val_accuracy` por trial + histograma de la distribución. |
| `plot_importancia(study, salida)` | Importancia de cada hiperparámetro (fANOVA). |

> **La optimización que hace que esto corra en minutos:** el backbone está congelado, así que
> sus features **no cambian entre trials**. Se calculan una vez y los 30 trials entrenan una
> MLP chica sobre vectores de 576 números, en vez de re-pasar 300 imágenes por la CNN 30
> veces. Este es el patrón correcto de Optuna sobre feature extraction.

**Resultado: 93.3%** (best trial 96.7%), hiperparámetros `2 capas · 128 units · dropout 0.2 ·
adam · lr 4.8e-3`. **Ver §3.3: la v7 muestra que este número estaba inflado.**

---

### v7 — `TensorFlow/v7/07_scripts.py` · protocolo de evaluación honesto

**Nuevo — cuatro cambios, todos de medición, ninguno de modelo:**

1. **Test apartado** (20%, 63 imgs) que Optuna nunca ve.
2. **Validación cruzada 5-fold** como objetivo de Optuna, en vez de una sola validación.
3. **Split estratificado y compartido** con PyTorch (`documentacion/particion.json`).
4. **5 semillas** con media ± desvío.

Además: **las curvas ROC llegan por fin al lado TensorFlow.** Hasta acá existían solo en
PyTorch (6 figuras) y la paridad "espejo" del proyecto estaba rota.

| Función | Qué hace |
|---|---|
| `dataset_de_rutas(rutas, etiquetas)` | Arma un `tf.data.Dataset` desde una **lista explícita de rutas** del manifiesto, no barriendo la carpeta. Es lo que garantiza que TF y PyTorch vean las mismas imágenes. Usa `decode_image(channels=3, expand_animations=False)`. |
| `construir_extractor()` | Igual que v6: MobileNetV3 congelada + GAP → 576-d. |
| `embeddings_de(rutas, etiquetas)` | Pre-cómputo de embeddings para una lista de rutas. |
| `construir_cabeza(params)` | La MLP de Optuna. **Mismo espacio de búsqueda que la v6**, a propósito. |
| `entrenar_cabeza(...)` | Entrena con `EarlyStopping`; devuelve `(modelo, mejor val_acc, mejor época)`. Si `X_va is None` entrena **a ciegas** N épocas: ese es el refit final. |
| `probabilidades(modelo, X)` | Vector completo de probabilidades softmax (requisito de ROC y ECE). |
| `validacion_cruzada(params, semilla, trial)` | **El cambio central.** Entrena en los 5 folds y devuelve la **media**. Reporta parciales a Optuna para que `MedianPruner` pode por fold. |
| `objective(trial)` | Propone hiperparámetros y devuelve la media de CV (ya no una sola validación). |
| `ejecutar_estudio()` | El `study` con TPE + MedianPruner. |
| `plot_historia` · `plot_importancia` | Figuras de Optuna sobre el nuevo objetivo. |

**Salidas:** `Resultado_7.txt`, `resultados_v7.json` (todas las métricas, agregadas y por
semilla), `mejores_hiperparametros.json`, y seis figuras: `Figure_optuna_historia_v7.png`,
`Figure_optuna_importancia_v7.png`, `Figure_matriz_v7.png`, `Figure_roc_v7.png`,
`Figure_calibracion_v7.png`, `Figure_semillas_v7.png`.

**Resultado: 0.8000 ± 0.0819** sobre test. Hiperparámetros elegidos por CV: `2 capas ·
64 units · dropout 0.2 · rmsprop · lr 5.9e-3` (CV = 0.8800). 18/30 trials completados,
**12 podados** por el pruner.

---

## 6. PyTorch — versión por versión

### v1 — `PyTorch/v1/01_script.py` · baseline espejo

| Elemento | Qué hace |
|---|---|
| `class BaselineCNN(nn.Module)` | Replica **exactamente** la arquitectura de Keras: `Conv(16)`→`Pool`→`Conv(32)`→`Pool`→`Flatten`→`Dense(64)`→`Dense(3)`. Con entrada 180×180 el flatten da `32·43·43 = 59168`, igual que en Keras. |
| `correr_epoca(dl, entrenar)` | Una época de entrenamiento **o** de evaluación según el flag. Encapsula el bucle manual que en Keras hace `fit()`. Es el patrón que se repite en v2, v3 y v4. |
| `roc_binaria(y_bin, scores)` | Curva ROC binaria **a mano**: ordena por score y acumula TP/FP; AUC por regla trapezoidal. |
| `curvas_roc_ovr(y_true, y_prob)` | ROC One-vs-Rest para las 3 clases + micro-promedio + AUC macro. |

**Resultado:** train 0.996 / val 0.83, loss 0.03 vs 0.41.

---

### v2 — `PyTorch/v2/02_scripts.py` · data augmentation

**Nuevo:** la augmentation va en el `transform` del loader de **train únicamente**:
`RandomAffine(degrees=18, scale=(0.9, 1.1))` + `ColorJitter(brightness=0.2, contrast=0.2)`.
Equivale al bloque de capas de Keras, pero vive **fuera** del modelo (§8).

**Resultado:** train 0.71 / val 0.72, matriz `[[12,6,0],[4,16,0],[1,6,15]]`.

---

### v3 — `PyTorch/v3/03_scripts.py` · early stopping manual

**Nuevo:** el early stopping **a mano**, porque PyTorch no tiene callbacks: se guarda un
`copy.deepcopy(modelo.state_dict())` cuando `val_loss` mejora, se cuentan las épocas sin
mejora, se corta al llegar a `patience=8` y se restauran los mejores pesos antes de evaluar.

**Resultado: 83.3%**, matriz `[[14,3,1],[1,18,1],[3,1,18]]`. **Persisten 4 errores 0↔2 tras
converger** → del lado PyTorch, la confusión extrema **no** era ruido de no-convergencia.
Ese hallazgo es lo que motiva la v4.

---

### v4 — `PyTorch/v4/04_scripts.py` · MobileNetV3-Small · **el titular del proyecto**

**Nuevo:** `mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)`, backbone
congelado (`requires_grad = False`), `classifier[3]` reemplazado por un `Linear(576, 3)`,
entrada 224×224 y **normalización ImageNet en el transform**.

> **El gotcha de BatchNorm — el contraste de frameworks de oro del proyecto.**
> `requires_grad=False` congela los **pesos** del backbone, pero **no** congela las
> estadísticas de las capas BatchNorm. En modo `train()`, BatchNorm sigue actualizando su
> `running_mean`/`running_var` con nuestros datos y **corrompe en silencio** las estadísticas
> de ImageNet que vienen con los pesos preentrenados. La solución está dentro de
> `correr_epoca()`: después de `modelo.train()`, forzar `modelo.features.eval()`.
> **En Keras esto no pasa:** `trainable=False` ya pone las BN en modo inferencia.

**Resultado: 90.0%**, matriz `[[14,4,0],[1,18,1],[0,0,22]]`. La esquina 0↔2 pasó de **4
errores a 0** con las mismas imágenes, y el recall de la clase 2 fue **22/22**.

**La lectura:** las features de ImageNet separan lo que la CNN propia confundía en los
extremos. **El límite era la capacidad de representación del modelo, no las etiquetas.**
*(La v7 matiza esto: sobre un test apartado la esquina 0↔2 vuelve a aparecer, así que el
0/22-22 era también un split favorable.)*

---

### v5 — `PyTorch/v5/05_scripts.py` · desbalance + pesos

| Función | Qué hace |
|---|---|
| `reporte_por_clase(cm, nombres)` | Precision / recall / f1 / soporte a mano. |
| `roc_binaria` · `curvas_roc_ovr` | ROC One-vs-Rest. |
| `construir_modelo()` | MobileNetV3 congelada + cabeza nueva; se llama una vez por variante. |
| `correr_epoca(modelo, optimizer, criterion, dl, entrenar)` | Versión parametrizada del bucle (hay dos modelos, así que ya no puede usar variables globales). |
| `matriz_confusion(modelo)` | Matriz sobre validación. |
| `probabilidades_val(modelo)` | Probabilidades softmax de validación (insumo de la ROC). |
| `entrenar_variante(usar_pesos)` | El A/B: con o sin pesos en la pérdida. |

En PyTorch la corrección se aplica con `nn.CrossEntropyLoss(weight=...)`. La alternativa
equivalente sería re-muestrear con un `WeightedRandomSampler` en el DataLoader.

**Resultado: 82.9% → 85.7%.** Efecto **sutil** (recall clase 1: 7/10 → 8/10) porque las
features preentrenadas ya separan bien. **Un efecto chico también es un hallazgo:** el
desbalance duele menos cuando las clases son separables en el espacio de features.

---

### v6 — `PyTorch/v6/06_scripts.py` · Optuna

| Función | Qué hace |
|---|---|
| `construir_extractor()` | `nn.Sequential(modelo.features, modelo.avgpool, nn.Flatten())` congelado y en `eval()` (gotcha de BatchNorm). |
| `embeddings_de(ds)` | Forward con `torch.no_grad()` → vectores 576-d, una sola vez. |
| `construir_cabeza(params)` | La MLP que Optuna propone, como `nn.Sequential`. |
| `crear_optimizer(cabeza, params)` | Adam / RMSprop / SGD(momentum=0.9) según el trial. |
| `evaluar(cabeza)` | Accuracy + matriz de confusión sobre validación. |
| `probabilidades_val(cabeza)` | Probabilidades softmax (insumo de ROC). |
| `entrenar_cabeza(params, epochs, paciencia)` | Bucle de entrenamiento con early stopping manual. |
| `objective(trial)` | La función objetivo de Optuna. |
| `ejecutar_estudio()` | El `study` con TPE + MedianPruner. |
| `plot_historia` · `plot_importancia` · `plot_roc` | Figuras. |

**Resultado: 93.3%**, hiperparámetros `2 capas · 256 units · dropout 0.45 · rmsprop ·
lr 9.2e-4`. **Ver §3.3.**

> **Inconsistencia detectada:** `PyTorch/v6/mejores_hiperparametros.json` guarda
> `lr = 5.29e-4` pero `Resultado_6.txt` reporta `lr = 9.16e-4`. Son de corridas distintas.
> Conviene regenerar la v6 o anotar cuál vale antes de defenderla.

---

### v7 — `PyTorch/v7/07_scripts.py` · protocolo de evaluación honesto

Espejo exacto del lado TF. **Mismo manifiesto de partición, mismo módulo de métricas, mismo
espacio de búsqueda.**

| Elemento | Qué hace |
|---|---|
| `class ImagenesDeLista(Dataset)` | Dataset desde una **lista explícita de rutas**, no desde una carpeta (reemplaza a `ImageFolder`). `convert("RGB")` es obligatorio: hay PNGs con canal alfa. |
| `construir_extractor()` | Backbone congelado en `eval()`. |
| `embeddings_de(rutas, etiquetas)` | Pre-cómputo de embeddings 576-d. |
| `construir_cabeza(params)` · `crear_optimizer(...)` | Igual que v6. |
| `evaluar_acc(cabeza, X, y)` | Accuracy sobre un tensor de embeddings. |
| `probabilidades(cabeza, X)` | Softmax completo. |
| `entrenar_cabeza(params, X_tr, y_tr, X_va, y_va, epochs, paciencia, semilla)` | Early stopping manual; devuelve también la **mejor época** (la necesita el refit). Con `X_va=None` entrena a ciegas. |
| `validacion_cruzada(params, semilla, trial)` | **El cambio central**: 5 folds, devuelve la media y reporta parciales para el pruning. |
| `objective(trial)` · `ejecutar_estudio()` | Optuna sobre el nuevo objetivo. |
| `plot_historia` · `plot_importancia` | Figuras. |

**Resultado: 0.7905 ± 0.0119** sobre test. Hiperparámetros por CV: `2 capas · 256 units ·
dropout 0.30 · adam · lr 3.1e-3` (CV = 0.8482). 24/30 trials completados, 6 podados.

> **Gotcha de Optuna 4.x descubierto en esta versión:** un trial **podado** no tiene
> `value = None` — Optuna le deja el **último valor intermedio** reportado. Filtrar por
> `value is not None` cuenta los podados como completados y ensucia las estadísticas del
> estudio con medias de 2-3 folds. Hay que mirar `trial.state`. Los dos scripts v7 lo hacen
> bien y lo documentan en el código.

---

## 7. Módulos compartidos (nuevos en la v7)

Antes de la v7 cada script se copiaba su propio `reporte_por_clase()` y su propio bloque de
ROC. Eso produjo dos problemas reales: las curvas ROC quedaron **solo** del lado PyTorch, y
cada métrica nueva había que escribirla dos veces con riesgo de que las implementaciones no
coincidieran y las comparaciones TF↔PyTorch mintieran.

Los dos módulos son **numpy puro** (sin torch, sin tf, sin sklearn) justamente para poder
importarse desde los dos venvs.

### 7.1 `documentacion/particion_datos.py`

Genera y lee la partición canónica. **Es el archivo que cierra el asterisco del split.**

| Función | Qué hace |
|---|---|
| `listar_por_clase(data_dir)` | `{clase: [rutas ordenadas]}`. El `sorted()` no es cosmético: garantiza la misma partición en máquinas distintas. |
| `huella_dataset(por_clase)` | SHA-256 del listado completo. **Guard**: si se agregan o borran imágenes, `cargar_particion()` avisa en vez de entrenar en silencio sobre una partición que ya no corresponde. |
| `construir_particion(data_dir, seed, n_folds, prop_test)` | Arma test + folds **estratificados** (reparto clase por clase, round-robin sobre la lista barajada). |
| `guardar_particion` · `cargar_particion` | Escriben y leen `particion.json`, con guards de existencia y de huella. |
| `rutas_y_etiquetas(particion, subconjunto, fold)` | Devuelve `(rutas, etiquetas)` de `"test"`, `"desarrollo"`, `"entrenamiento"` o `"validacion"`. La etiqueta sale del índice alfabético de la clase: **idéntico criterio en los dos frameworks**. |
| `conteos(particion, rutas)` | Cuántas imágenes de cada clase hay en una lista (para imprimir soportes). |
| `mostrar(particion)` | Tabla por consola + **guard de solapamiento** entre test y desarrollo. |

Estructura resultante sobre 313 imágenes:

```
dataset/  (313)
  ├── TEST        63 imgs  (22 / 19 / 22)   <- apartado, se usa UNA vez
  └── DESARROLLO 250 imgs  (86 / 75 / 89)
        └── 5 folds de ~50 (17-18 / 15 / 17-18 cada uno)
```

### 7.2 `documentacion/metricas.py`

| Función | Qué mide |
|---|---|
| `matriz_confusion(y_true, y_pred, n_clases)` | La matriz, a mano. |
| `reporte_por_clase(cm, nombres, imprimir)` | Precision / recall / f1 / soporte + macro + accuracy. Devuelve dict para poder agregar. |
| `errores_extremos(cm)` | **El error caro:** cuenta la esquina 0↔2 y la separa en `2→0` (deja pasar IA) y `0→2` (acusa de más). |
| `qwk(cm)` | **Kappa cuadrático ponderado.** Pondera el error por la **distancia al cuadrado**: confundir extremos pesa 4× más que confundir vecinos, y descuenta el acuerdo por azar. Es el criterio de negocio del proyecto convertido en métrica; levanta el tradeoff "clases ordinales tratadas como nominales" que `avances_proyecto.md` §4 declaraba asumido. |
| `mae_ordinal(cm)` | Media de \|real − predicho\| en la escala de clases. Interpretable: 0.27 = el modelo se corre 0.27 niveles en promedio. |
| `roc_binaria(y_bin, scores)` | ROC binaria a mano; AUC por trapecios (compatible numpy 1.x y 2.x, que renombró `trapz`→`trapezoid`). |
| `curvas_roc_ovr(y_true, y_prob, n_clases)` | One-vs-Rest + micro-promedio + AUC macro. |
| `ece(y_true, y_prob, n_bins)` | **Expected Calibration Error**: ¿cuando el modelo dice 90%, acierta el 90%? Un modelo puede tener buena accuracy y ECE malo (siempre dice 99%, aun equivocándose). |
| `resumen_completo(y_true, y_prob, nombres)` | Todas las métricas en un dict, listo para promediar. |
| `agregar_semillas(resumenes)` | Media, desvío, mín y máx de cada métrica sobre N semillas + **matriz acumulada**. |
| `imprimir_agregado(agregado, titulo)` | La tabla `media ± std` con flechas ↑/↓ según qué es mejor. |
| `plot_matriz` · `plot_roc` · `plot_calibracion` · `plot_semillas` | Las figuras. `plot_calibracion` dibuja el diagrama de fiabilidad; `plot_semillas` hace **visible el ruido**: si los puntos de dos modelos se solapan, la diferencia de medias no significa nada. |

---

## 8. El catálogo de contrastes TensorFlow ↔ PyTorch

Este es el material del requisito "contraste explícito entre frameworks". Todos salieron de
problemas reales encontrados escribiendo el código, no de un libro.

| Tema | TensorFlow / Keras | PyTorch |
|---|---|---|
| **Data augmentation** | Capas **dentro** del modelo (`RandomRotation`, `RandomZoom`…), viajan con el `.keras`. | En el **transform** del DataLoader de train, **fuera** del modelo. |
| **Normalización ImageNet** | **Dentro** del modelo (`include_preprocessing=True`): recibe 0-255 crudo y normaliza sola. El consumo no replica nada. | En el **transform**: la inferencia tiene que reproducir `mean`/`std` a mano o se des-sincroniza. Por eso van dentro del checkpoint. |
| **BatchNorm con backbone congelado** | `trainable=False` **ya** pone las BN en modo inferencia. | `requires_grad=False` congela los pesos pero **no** las estadísticas: hay que forzar `features.eval()` o se corrompen en silencio. **El gotcha más importante del proyecto.** |
| **Early stopping** | Callback de una línea con `restore_best_weights=True`. | A mano: `copy.deepcopy(state_dict())`, contador de épocas sin mejora, restaurar al final. |
| **Bucle de entrenamiento** | `model.fit()`. | Bucle explícito (`correr_epoca`), con `zero_grad` / `backward` / `step`. |
| **Pesos de clase** | Argumento de `fit()`: `class_weight={...}`. | `CrossEntropyLoss(weight=...)` **o** `WeightedRandomSampler`. No hay un argumento de "fit". |
| **Softmax** | La última capa lo aplica: `predict()` devuelve probabilidades. | `CrossEntropyLoss` lo incluye internamente, así que el modelo devuelve **logits**: hay que aplicar `torch.softmax` para el vector de probabilidades. |
| **Orden de clases** | `image_dataset_from_directory` ordena alfabéticamente. | `ImageFolder` ordena alfabéticamente. **Igual en los dos** — de ahí los prefijos `0_/1_/2_`. |
| **Carga de datos (v7)** | `tf.data.from_tensor_slices` sobre la lista de rutas + `map(decode_image)`. | `Dataset` propio con PIL + `convert("RGB")`. |
| **Estabilidad entre semillas (v7)** | **±0.0819** de accuracy. Mucho más sensible a la semilla. | **±0.0119**. Notoriamente más estable con la misma receta. |

---

## 9. Qué falta

**Inmediato**
- **v8 — la cadena de destilación** (el bonus track): entrenar v1, destilarlo, reentrenar
  con lo que agrega v2, destilar de nuevo, etc., y comparar contra un modelo entrenado
  directo con todo. La v7 existe justamente para que esa comparación se pueda medir: con
  test apartado y 5 semillas, una diferencia de 2 puntos ya se puede declarar ruido o señal.
  Los hiperparámetros que quedaron fuera del espacio de la v7 (`weight_decay`,
  `label_smoothing`, scheduler, fine-tuning en 2 fases) entran ahí.
- **T7 — esqueleto Ionic** (Angular + OpenCV.js + TF.js) para la entrega final.

**Deudas metodológicas conocidas**
1. **Domain gap sin medir.** Todos los números son sobre renders limpios; no hay una sola
   foto de teléfono en el circuito de métricas. Son un **techo optimista**.
2. **Riesgo de atajo por procedencia.** La clase 0 sale de datasets pre-2022 y la clase 2 se
   generó con Gamma/Canva: son **pipelines de render distintos** (resolución, compresión,
   perfil de color). El modelo podría estar aprendiendo el pipeline y no la huella de IA.
   Un **Grad-CAM** lo contestaría.
3. **Protocolo de etiquetado 1↔2 sin escribir.** Es el pendiente más viejo del proyecto; la
   clase 1 sigue siendo la de peor F1 (~0.75) en los dos frameworks.
4. **Dataset chico** (313 imágenes). La clase 2, la más cara de producir, fija el techo.
5. `PyTorch/v6/mejores_hiperparametros.json` no coincide con su `Resultado_6.txt` (§6, v6).
6. Los `Resultado_*.txt` de PyTorch v1-v6 **no** incluyen la salida de ROC-AUC, aunque las
   figuras existen: se regeneraron las figuras sin volcar los logs.

**Tareas de Bato** (heredadas de `DetalleProyecto.md` §6)
- Prosa del análisis y plan de acción en `analisis_metricas.md` (bloques `COMPLETAR: Bato`).
- Renombrar la carpeta local `data/` → `dataset/` (hoy hay un symlink).
- Fotos reales con los scripts de cámara, para la primera medición del domain gap.
