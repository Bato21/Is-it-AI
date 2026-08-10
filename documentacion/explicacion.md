# Explicación del proyecto — guía completa de lo que se hizo

*Guía de lectura del repositorio **Is-it-AI**. Recorre, versión por versión y separado por
framework, qué se agregó en cada paso, **cómo se llama cada función**, qué hace y qué
resultado dio. Escrita al cierre de la **v8**, con números de corridas reales.*

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
├── TensorFlow/  v1..v8/        entrenamiento — un script por versión
│   └── export_tfjs/            exportación a TensorFlow.js (venv aparte)
├── PyTorch/     v1..v8/        entrenamiento — espejo exacto
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
| v7 | **Protocolo de evaluación honesto** | ¿Cuánto de lo reportado hasta acá era real? |
| **v8** | **Destilación en cadena vs. entrenamiento directo** | **¿Conviene transferir el conocimiento de versión en versión?** |

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

### 3.4 Resultados — v8 (destilación en cadena vs. directo)

Mismo test apartado de 63 imágenes, mismas 5 semillas. Los dos brazos terminan en la
**misma arquitectura** (MobileNetV3 congelada + cabeza con los hiperparámetros de la v7)
y ven **los mismos datos** (1 vista limpia + 2 aumentadas por imagen). Lo único que
cambia es el procedimiento de entrenamiento.

**Criterio de veredicto, declarado ANTES de correr:** una diferencia solo cuenta si
`|B − A|` supera la suma de los desvíos de los dos brazos. Con 63 imágenes, todo lo demás
es ruido, y decirlo es más honesto que festejar decimales.

| Métrica | | **TF** A directo | **TF** B cadena | **PT** A directo | **PT** B cadena |
|---|:---:|---|---|---|---|
| accuracy | ↑ | 0.8698 ± 0.0394 | 0.8667 ± **0.0078** | 0.8190 ± 0.0215 | 0.8063 ± **0.0119** |
| F1 macro | ↑ | 0.8681 | 0.8639 | 0.8181 | 0.8054 |
| recall clase 2 | ↑ | 0.9273 | **0.9545** | 0.7364 | **0.7909** |
| errores 0↔2 | ↓ | 2.0 | 1.8 | 3.2 | 4.0 |
| QWK | ↑ | 0.8405 | 0.8452 | 0.7516 | 0.7243 |
| AUC macro | ↑ | 0.9475 | **0.9545** | 0.9360 | **0.9465** |
| ECE | ↓ | 0.1097 | **0.0924** | 0.1380 | **0.1107** |

**Veredicto formal: TODAS las diferencias son ruido**, en los dos frameworks. Con este
tamaño de test no se puede afirmar que la cadena sea mejor ni peor que el directo. Esa es
la respuesta defendible, y es la que hay que dar si preguntan.

**Pero hay un patrón que sí vale la pena mirar.** Cinco efectos apuntan en la MISMA
dirección en **dos implementaciones independientes** (frameworks distintos, augmentation
distinta, hiperparámetros distintos):

| Efecto | TensorFlow | PyTorch |
|---|---|---|
| accuracy | −0.0032 | −0.0127 | 
| recall clase 2 | **+0.0273** | **+0.0545** |
| AUC macro | **+0.0070** | **+0.0106** |
| ECE (calibración) | **−0.0173** | **−0.0273** |
| **desvío de accuracy** | **0.0394 → 0.0078** | **0.0215 → 0.0119** |

Que cinco signos coincidan en dos corridas independientes es más informativo que
cualquiera de los deltas por separado. La lectura honesta: **la cadena paga un poco de
accuracy y compra calibración, AUC, recall de la clase cara y —sobre todo— ESTABILIDAD.**

El efecto de estabilidad es el más marcado y el menos esperado: en TensorFlow el desvío
entre semillas cae de **±0.0394 a ±0.0078, cinco veces menos**. Es el mismo problema que
la v7 había destapado (TF variaba de 0.6667 a 0.8889 con la misma configuración), y la
destilación lo amortigua: los targets blandos del profesor son una señal mucho más rica y
consistente que las etiquetas duras, así que el punto de llegada depende menos de la
inicialización. **Para un producto que hay que desplegar, esto vale más que 1 punto de
accuracy.**

### 3.5 El hallazgo incómodo: la cadena se rompe en el eslabón 4

Progresión eslabón a eslabón sobre el test (media de 5 semillas):

| Eslabón | TF acc | TF QWK | TF ECE | PT acc | PT QWK | PT ECE | PT 0↔2 |
|---|---|---|---|---|---|---|---|
| E1 (v1: CNN pelada) | 0.8825 | 0.8473 | 0.0763 | 0.8444 | 0.8021 | 0.0695 | 2.6 |
| E2 (v2: + augmentation) | 0.8825 | 0.8541 | 0.0863 | 0.8667 | 0.8569 | 0.0688 | 1.4 |
| **E3 (v3: + converger)** | **0.8857** | 0.8521 | **0.0599** | **0.8698** | **0.8659** | 0.0691 | **1.2** |
| E4 (v4: → MobileNetV3) | 0.8349 | 0.8153 | 0.0923 | 0.7905 | 0.7705 | 0.0944 | 2.2 |
| E5 (v7: cabeza Optuna) | 0.8667 | 0.8452 | 0.0924 | 0.8063 | 0.7243 | 0.1107 | 4.0 |

Dos conclusiones, y la segunda es fuerte:

**1. La destilación entre CNNs funciona.** En PyTorch, E1→E3 sube 2.5 pp de accuracy,
+6.4 pp de QWK y **baja los errores 0↔2 de 2.6 a 1.2** — el error caro del proyecto,
más que a la mitad, con la misma arquitectura y los mismos datos. Eso no es ruido de
inicialización: es transferencia de conocimiento.

**2. El salto a MobileNetV3 (E4) es lo que ROMPE la cadena, y eso invierte el titular
del proyecto.** En los DOS frameworks, el mejor eslabón es **E3 — la CNN chica entrenada
desde cero**, y cae al pasar a las features congeladas de ImageNet. La v4 había celebrado
"transfer learning > CNN propia" con 90.0% vs 83.3% en PyTorch; medido sobre el test
apartado, con la misma partición y 5 semillas, **la CNN propia iguala o gana**.

Tiene sentido conceptual: las features de ImageNet están entrenadas para distinguir perros
de aviones, no para detectar regularidad de plantilla, paleta uniforme y tipografía de
generador. Una CNN chica entrenada sobre este dominio puede aprender justamente eso.

> **El matiz que hay que decir sí o sí:** E4/E5 usan el backbone **CONGELADO** (feature
> extraction). Esto NO dice que MobileNetV3 sea peor que la CNN propia — dice que
> *las features de ImageNet sin adaptar* no son mejores. Con fine-tuning de las últimas
> capas la conclusión podría darse vuelta, y eso **todavía no está medido**. Es el
> experimento más urgente que queda pendiente.

### 3.6 Sensibilidad a la temperatura y al peso de la destilación

Barrido medido sobre **validación cruzada de desarrollo, nunca sobre el test** (el barrido
es información, no un mecanismo de selección). `alpha=0.0` equivale a no destilar y es la
línea base dentro del mismo barrido.

| | α=0.0 | α=0.3 | α=0.5 | α=0.7 | α=0.9 |
|---|---|---|---|---|---|
| **PyTorch** T=2 | 0.8122 | 0.8043 | 0.8161 | 0.8201 | 0.8161 |
| **PyTorch** T=4 | 0.8122 | 0.8202 | 0.8121 | 0.8082 | 0.8241 |
| **PyTorch** T=8 | 0.8122 | 0.8281 | 0.8162 | 0.8203 | **0.8321** |
| **TensorFlow** T=2 | 0.8479 | **0.8561** | 0.8237 | 0.8157 | 0.8121 |
| **TensorFlow** T=4 | 0.8401 | 0.8197 | 0.8241 | 0.7957 | 0.7917 |
| **TensorFlow** T=8 | 0.8401 | 0.8240 | 0.7960 | 0.8242 | 0.7799 |

- **Chequeo de sanidad que hay que señalar:** las tres filas de α=0 de PyTorch dan
  exactamente `0.8122`. Con α=0 la pérdida se reduce a cross-entropy pura y la temperatura
  deja de existir, así que las tres TIENEN que coincidir. Coinciden.
- **PyTorch y TensorFlow prefieren regímenes OPUESTOS.** PyTorch mejora con temperatura
  alta y mucho peso de KL (T=8, α=0.9); TensorFlow prefiere temperatura baja y poco KL
  (T=2, α=0.3) y se degrada feo con α alto (0.78-0.79). No hay un T/α universal.
- Los valores fijos del script del profe (T=4, α=0.7) quedaron **cerca del óptimo en
  PyTorch y cerca del peor caso en TensorFlow**. Se mantuvieron igual a propósito: si el
  brazo B pudiera buscar dos hiperparámetros que el brazo A no tiene, ganaría por tener
  más búsqueda, no por destilar.

> **Salvedad de comparación TF↔PyTorch en la v8:** en la v7 los dos frameworks compartían
> TODO y sus resultados coincidían (0.8000 vs 0.7905). En la v8 comparten la partición y
> las métricas, pero **cada uno genera sus vistas aumentadas con su propia augmentation**
> (capas de Keras vs. transforms de torchvision) y usa los hiperparámetros que su propia
> CV eligió. Por eso vuelve a aparecer una brecha (0.87 vs 0.82) que **no** debe leerse
> como "TensorFlow es mejor": son datos de entrenamiento distintos.

---

## 4. Cómo correr todo

```bash
# 0) Verificar el dataset
python documentacion/verificar_dataset.py

# 1) Generar la partición canónica (UNA vez; los dos frameworks leen este archivo)
python documentacion/particion_datos.py
python documentacion/particion_datos.py --mostrar     # ver la que ya existe

# 2) Entrenar (venv tensorflow-ia para TF, frameworks-ia para PyTorch)
python TensorFlow/v7/07_scripts.py      # protocolo honesto  (~5 min)
python PyTorch/v7/07_scripts.py

# 2b) v8: destilación en cadena vs. directo. OJO: entrena 3 CNNs desde cero por
#     semilla, así que tarda (~20 min TF, ~40 min PyTorch en CPU).
#     Redirigir a archivo SIN pipes intermedios: un `| head` mata el proceso por
#     SIGPIPE a mitad de camino (pasó en la primera corrida).
python TensorFlow/v8/08_scripts.py > TensorFlow/v8/Resultado_8.txt 2>&1
python PyTorch/v8/08_scripts.py    > PyTorch/v8/Resultado_8.txt    2>&1

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

### v8 — `TensorFlow/v8/08_scripts.py` · destilación en cadena vs. directo

**Nuevo:** la pérdida de destilación con temperatura, el conjunto fijo de vistas, la
cadena de 5 eslabones y la comparación A/B contra el entrenamiento directo.

| Función | Qué hace |
|---|---|
| `construir_extractor()` | MobileNetV3 congelada + GAP → 576-d (igual que v6/v7). |
| `preparar_vistas(rutas, etiquetas, n_aug, semilla)` | Genera **1 vista limpia + n_aug aumentadas** por imagen. Aplica la augmentation **antes** de los dos resize (180 para la CNN, 224 para MobileNet), así el profesor y el estudiante ven la MISMA vista — requisito de la destilación. Devuelve las imágenes 180px en `uint8` y los embeddings ya pre-computados. |
| `baseline_cnn()` | La CNN de v1/v2/v3, pero **devolviendo logits** (sin softmax final). |
| `cabeza_v4()` | Réplica del `classifier` de MobileNetV3-Small: `Dense(1024, hard_swish) → Dropout(0.2) → Dense(3)`. Logits. |
| `cabeza_v7(params)` | La cabeza con los hiperparámetros que Optuna eligió en la v7. Logits. |
| `hacer_perdida_destilacion(T, alpha)` | **El corazón de la versión.** Devuelve `alpha·KL(T)·T² + (1−alpha)·CE`, lista para `compile()`. |
| `entrenar_eslabon(...)` | Entrena un eslabón; si `logits_prof is None` usa cross-entropy pura. |
| `logits_de(modelo, X)` · `probabilidades(modelo, X)` | Logits crudos y softmax aplicado a mano. |
| `brazo_directo(params, epochs, semilla)` | **BRAZO A**: la receta v7 entrenada de una. |
| `brazo_cadena(params, epochs_final, semilla)` | **BRAZO B**: los 5 eslabones encadenados. Devuelve el modelo final, el historial por eslabón y los logits de E4. |
| `barrido_T_alpha(params, epochs_final, semilla, logits_prof)` | Barre T ∈ {2,4,8} × α ∈ {0, .3, .5, .7, .9} **sobre CV**, reutilizando el profesor E4 ya entrenado. |

**Resultado: A directo 0.8698 ± 0.0394 · B cadena 0.8667 ± 0.0078.** Todas las
diferencias, ruido. Ver §3.4-3.6.

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

> **⚠ Este titular quedó en discusión.** La v7 lo matizó (sobre un test apartado la esquina
> 0↔2 vuelve a aparecer, así que el 22/22 era también un split favorable) y la **v8 lo
> contradice de frente**: en los dos frameworks, el eslabón con la CNN propia (E3) le gana
> al eslabón con MobileNetV3 congelada (E4). Ver §3.5. La salvedad es que E4/E5 usan el
> backbone **congelado**; con fine-tuning la conclusión podría volver a darse vuelta, y eso
> es hoy el experimento pendiente número 1.

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

### v8 — `PyTorch/v8/08_scripts.py` · destilación en cadena vs. directo

Espejo exacto del lado TF: misma partición, mismas métricas, misma cadena, mismos T y α.

| Elemento | Qué hace |
|---|---|
| `preparar_vistas(rutas, etiquetas, n_aug, semilla)` | 1 vista limpia + n_aug aumentadas por imagen; augmentation **antes** de los dos resize (180 para la CNN, 224 para MobileNet), así el profesor y el estudiante ven la MISMA vista. Las 180px se guardan en `uint8` (73 MB) y se normalizan por batch: en `float32` serían 292 MB y la máquina tenía poca RAM libre. |
| `class BaselineCNN(nn.Module)` | La CNN de v1/v2/v3 **sin tocar una línea**: sin padding, `32·43·43 = 59168` en el flatten. Los eslabones E1-E3 son literalmente los modelos de las versiones que representan. |
| `cabeza_v4()` | Réplica del `classifier` de MobileNetV3-Small: `Linear(576,1024) → Hardswish → Dropout(0.2) → Linear(1024,3)`. |
| `cabeza_v7(params)` · `optimizador_v7(modelo, params)` | La cabeza y el optimizador con los hiperparámetros que Optuna eligió en la v7. |
| `perdida_destilacion(logits_est, logits_prof, y, T, alpha)` | **El corazón de la versión.** `alpha · F.kl_div(log_softmax(est/T), softmax(prof/T), "batchmean") · T² + (1−alpha) · CE`. Calcada del `perdida_kl` del script del profe, de 50.257 tokens a 3 clases. |
| `normalizar180(x_uint8)` | `uint8 [0,255] → float [0,1]`. Equivalente del `ToTensor()` de v1-v3. |
| `entrenar_eslabon(...)` | Entrena un eslabón con los logits del profesor **pre-computados** (el profesor está congelado, así que no cambian entre épocas: por eso la cadena corre en minutos). |
| `logits_de(modelo, X, es_cnn)` · `probabilidades(...)` | Logits por lotes y softmax completo. |
| `brazo_directo(params, epochs, semilla)` | **BRAZO A**: la receta v7 entrenada de una, sobre las mismas vistas que la cadena. |
| `brazo_cadena(params, epochs_final, semilla)` | **BRAZO B**: los 5 eslabones. Devuelve el modelo final, el historial por eslabón y los logits de E4. |
| `barrido_T_alpha(...)` | Barre T × α **sobre CV**, reutilizando el profesor E4 ya entrenado. Excluye **todas** las vistas de las imágenes de validación, no solo la limpia: si una vista aumentada quedara en train y su vista limpia en validación, habría fuga (es la misma diapositiva). |

**Resultado: A directo 0.8190 ± 0.0215 · B cadena 0.8063 ± 0.0119.** Todas las
diferencias, ruido. Ver §3.4-3.6.

**Reglas del experimento (valen para los dos frameworks):**
- El estudiante se **RE-INICIALIZA** en cada eslabón (random en E1-E3, ImageNet en E4-E5).
  Si heredara los pesos del profesor sería *fine-tuning* con otro nombre, no destilación.
- Se destila sobre las **vistas aumentadas**, no sobre las limpias. Es la solución al
  **problema del profesor saturado**: E1 memoriza las vistas limpias (softmax casi
  one-hot, no enseña nada), pero nunca vio las rotadas, así que ahí sí duda — y el
  conocimiento oscuro solo existe donde el profesor duda.
- Las vistas se generan **una sola vez con semilla fija** (`SEED_VISTAS = 999`), igual
  para los dos brazos y para todas las semillas de entrenamiento. Se pierde diversidad de
  augmentation a cambio de que todo sea pre-computable y 100% reproducible.
- **No hay early stopping dentro de la cadena**: no hay un conjunto de validación que los
  dos brazos puedan compartir sin quitarle datos a alguno. El "converger" de la v3 se
  representa con un presupuesto de épocas mayor (E1=10, E2=30, E3=40). Es una
  simplificación consciente.

> **Verificación cruzada de la pérdida (vale la pena mencionarla en la defensa):** sobre
> los mismos logits fijos, la pérdida de destilación da `1.30930722` en PyTorch y
> `1.30930674` en TensorFlow. Coinciden hasta la 7ª cifra, así que la afirmación "las dos
> implementaciones son la misma pérdida" está **medida**, no asumida.

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
| **Logits vs. probabilidades (v8)** | La costumbre del proyecto era terminar la cabeza en `softmax`. **La destilación obliga a cambiarlo**: para aplicar temperatura hay que dividir los LOGITS por T, y una vez aplicado el softmax ya no se puede. Desde la v8 las cabezas devuelven logits y el softmax se aplica a mano. | Ya devolvía logits (`CrossEntropyLoss` incluye el softmax). **No hubo que cambiar nada.** |
| **Pasarle los logits del profesor a la pérdida (v8)** | `loss(y_true, y_pred)` solo recibe esos dos tensores. Hay que **contrabandear** los logits del profesor dentro de `y_true`: se arma un `y_true` de ancho `1 + n_clases` = `[etiqueta, logits_profesor]` y la pérdida lo desempaqueta. Es el idiom estándar de KD en Keras. | La pérdida es una función común: se le pasa lo que uno quiera. No hace falta ningún truco. |
| **KL divergence (v8)** | A mano: `reduce_mean(reduce_sum(p_prof · (log_p_prof − log_p_est)))`. | `F.kl_div(..., reduction="batchmean")` lo trae hecho. **Los dos dan el mismo número hasta la 7ª cifra.** |
| **Augmentation determinista (v8)** | Capas de Keras (`RandomRotation`, `RandomZoom`, `RandomBrightness`, `RandomContrast`) invocadas con `training=True`. | `transforms.RandomAffine` + `ColorJitter` de torchvision. **No producen las mismas vistas**, y por eso la v8 vuelve a mostrar una brecha TF↔PyTorch que la v7 había cerrado. |
| **Régimen óptimo de destilación (v8)** | Prefiere **T bajo y α bajo** (T=2, α=0.3). Se degrada con α alto. | Prefiere **T alto y α alto** (T=8, α=0.9). Régimen opuesto. |

---

## 9. Qué falta

**Inmediato — lo que la v8 dejó servido**
1. **Fine-tuning en dos fases (el experimento más urgente).** La v8 mostró que el mejor
   eslabón de la cadena es la **CNN propia (E3)**, no las features congeladas de ImageNet
   (E4/E5), en los dos frameworks. Eso pone en duda el titular "transfer learning > CNN
   propia" de la v4 — pero solo prueba que las features de ImageNet **sin adaptar** no
   ganan. Descongelar los últimos bloques del backbone con un lr bajo es lo que decide la
   discusión, y hoy no está medido.
2. **Escenario 2 del script del profe (SeqKD / pseudo-etiquetado).** La v8 implementó el
   Escenario 1 (mismo espacio de salida, KL sobre logits). El Escenario 2 no tiene análogo
   literal en clasificación, pero sí uno útil: que el profesor etiquete imágenes **sin
   etiquetar** y el estudiante las aprenda con cross-entropy. Aplicado a **fotos reales de
   teléfono**, es el ataque más barato al domain gap.
3. **Hiperparámetros que quedaron fuera del espacio de la v7:** `weight_decay`,
   `label_smoothing` (el hermano conceptual de la destilación), scheduler de lr.
4. **T7 — esqueleto Ionic** (Angular + OpenCV.js + TF.js) para la entrega final.

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
7. **Test de 63 imágenes.** La v8 terminó con TODAS sus comparaciones marcadas como
   "ruido". No es un defecto del experimento: es el tamaño del test. Para poder afirmar
   diferencias de 2-3 puntos hace falta más dataset, no más técnica. Es el argumento más
   fuerte a favor de la regla de la casa: **más datos primero, técnica después.**

**Tareas de Bato** (heredadas de `DetalleProyecto.md` §6)
- Prosa del análisis y plan de acción en `analisis_metricas.md` (bloques `COMPLETAR: Bato`).
- Renombrar la carpeta local `data/` → `dataset/` (hoy hay un symlink).
- Fotos reales con los scripts de cámara, para la primera medición del domain gap.
