# Estado del repositorio — Is-it-AI (detector de huella de IA en diapositivas)

*Resumen completo del estado actual, versión por versión, cómo funciona todo y qué falta.
Escrito al cierre de la **entrega 2**. Los números son de corridas reales (seed 123).*

---

## 1. Qué es el proyecto

Clasificador que, a partir de la **foto de una diapositiva**, estima su **densidad de
artefactos visuales de IA** en 3 niveles ordinales:

| Clase | Significado |
|-------|-------------|
| `0_sin_ia`      | Sin huella visible de IA (parece humano). |
| `1_rastro_ia`   | Huella parcial (IA con intervención humana). |
| `2_saturada_ia` | Predominantemente generada por IA. |

Decisión conceptual clave: **no mide procedencia ni proceso**, solo la **huella visual**,
que es lo único que una CNN puede aprender de una imagen. Implementación **dual y espejo**
en TensorFlow/Keras y PyTorch, con comentarios de *contraste de frameworks* donde difieren.

---

## 2. El arco por versiones (un cambio conceptual por versión)

Cada versión cambia **una sola cosa** respecto de la anterior, para que el A/B sea limpio.
Matrices: filas = real, columnas = predicho, orden `0 / 1 / 2`.

### TensorFlow / Keras

| Ver | Cambio conceptual | Resultado (val) | Matriz | Lectura |
|-----|-------------------|-----------------|--------|---------|
| v1 | CNN desde cero, sin nada (sobreajuste a propósito) | train 1.00 / val 0.97, loss 0.01 vs 0.10 | — | El overfitting se ve en la **loss**, no en la accuracy. |
| v2 | + Data augmentation domain-aware | train 0.77 / val 0.75 | `[[20,4,0],[4,13,1],[3,3,12]]` | Mató el overfitting pero quedó **sub-entrenado**. |
| v3 | + Early stopping (converger) | **88.3%** | `[[21,3,0],[4,14,0],[0,0,18]]` | Convergió; esquina 0↔2 **limpia**, error solo entre vecinos. |
| v4 | CNN → MobileNetV3-Small (transfer learning) | **80.0%** | `[[21,2,1],[8,10,0],[1,0,17]]` | Cortó en la época 60/60 (**sub-entrenada**); recall clase 1 flojo (10/18). |
| v5 | Dataset desbalanceado (100/50/25) sin/con pesos | 71.4% → **80.0%** | `[[16,2,0],[7,3,0],[1,0,6]]` → `[[15,2,1],[3,7,0],[1,0,6]]` | Sin pesos el recall de clase 1 **colapsa (3/10)**; con pesos **se recupera (7/10)**. Efecto de libro. |
| v6 | **Optuna** sobre la cabeza (backbone congelado, features pre-computadas) | *(se corre en la máquina del alumno)* | — | Hiperparámetros de la cabeza (dropout/LR/optimizador/tamaño) **buscados** con TPE+MedianPruner, no a ojo. Guarda `mejores_hiperparametros.json` + figuras. |

### PyTorch

| Ver | Cambio conceptual | Resultado (val) | Matriz | Lectura |
|-----|-------------------|-----------------|--------|---------|
| v1 | CNN desde cero, sin nada | train 0.996 / val 0.83, loss 0.03 vs 0.41 | — | Overfitting visible en la loss. |
| v2 | + Data augmentation | train 0.71 / val 0.72 | `[[12,6,0],[4,16,0],[1,6,15]]` | Regularizado pero sub-entrenado. |
| v3 | + Early stopping (converger) | **83.3%** | `[[14,3,1],[1,18,1],[3,1,18]]` | Tras converger **persisten 4 errores extremos 0↔2** → no era ruido. |
| v4 | CNN → MobileNetV3-Small | **90.0%** | `[[14,4,0],[1,18,1],[0,0,22]]` | Esquina 0↔2 **en cero**; clase 2 recall **22/22**. El titular del proyecto. |
| v5 | Dataset desbalanceado sin/con pesos | 82.9% → **85.7%** | `[[16,1,0],[3,7,0],[2,0,6]]` → `[[16,1,0],[2,8,0],[1,1,6]]` | Efecto **sutil** (las features preentrenadas ya separan bien) — también es un hallazgo. |
| v6 | **Optuna** sobre la cabeza (espejo del lado TF) | *(se corre en la máquina del alumno)* | — | Mismo espacio de búsqueda y lógica de Optuna; cambia solo cómo se arma/entrena la cabeza (bucle manual vs `fit`). |

**El titular (PyTorch, v3→v4):** las mismas imágenes que la CNN desde cero confundía en
los extremos, las features preentrenadas de ImageNet las separan. El límite era la
**capacidad del modelo, no las etiquetas**.

**El matiz honesto (TF, v3→v4):** TF v4 (80%) **no** superó a TF v3 (88.3%) en este split,
porque cortó sub-entrenada (techo de 60 épocas) y el split difiere entre frameworks. Está
documentado sin forzar la narrativa en `analisis_metricas.md`.

---

## 3. Cómo funciona todo (pipeline de punta a punta)

```
 OBTENCIÓN (separada del entrenamiento)
   presentaciones .pdf/.pptx
     └─ documentacion/deck_a_imagenes.py ──► 1 PNG por diapositiva
          └─ clasificación manual ──► dataset/{0_sin_ia, 1_rastro_ia, 2_saturada_ia}/
               └─ documentacion/verificar_dataset.py  (chequeo; corta si falta una clase)

 ENTRENAMIENTO (los scripts SOLO leen dataset/, nunca lo modifican)
   TensorFlow/vN/0N_scripts.py   y   PyTorch/vN/0N_scripts.py
     └─ guardan modelo (.keras/.pt), curvas (Figure_1) y matriz (Figure_2) + Resultado_N.txt

 CONSUMO
   modelos/v4/modelotf_v4.py / modelopt_v4.py   → clasifica imagenes_a_probar/ (vector completo)
   modelos/v4/camara_tf.py   / camara_pt.py     → webcam en vivo con OpenCV (overlay + teclas)

 DESPLIEGUE WEB (puente a la entrega final)
   TensorFlow/export_tfjs/exportar_tfjs.py  → web/model/ (TensorFlow.js)
     └─ web/index.html  → smoke test en navegador, 3 probabilidades, paridad < 0.02 vs Python
```

Decisiones de diseño que atraviesan todo:
- **Ruta única `dataset/`** en los 8 scripts de entrenamiento (guard si falta).
- **Normalización ImageNet dentro del modelo en TF** (el navegador/consumo recibe 0-255
  crudo) vs **en el transform en PyTorch** (la inferencia replica mean/std del checkpoint).
- **`print("Orden de clases", ...)`** en todo script que carga datos (gotcha alfabético
  resuelto por los prefijos `0_/1_/2_`).
- **Vector completo de probabilidades** en toda inferencia, nunca solo el argmax.
- Modelos y datos **fuera de git**; el `.gitignore` cubre los artefactos regenerables.

---

## 4. Estructura del repo

```
├── dataset/                     (fuera de git; 108/94/98 imágenes reales)
├── dataset_desbalanceado/       (fuera de git; subset 100/50/25 para la v5)
├── TensorFlow/  v1..v6/         entrenamiento espejo + requirements.txt (v6 = Optuna)
│   └── export_tfjs/             exportar_tfjs.py + requirements_tfjs.txt (venv aparte)
├── PyTorch/     v1..v6/         entrenamiento espejo + requirements.txt (v6 = Optuna)
├── modelos/
│   ├── v2/  modelotf_v2.py · modelopt_v2.py     (consumo v2)
│   ├── v4/  modelotf_v4.py · modelopt_v4.py     (consumo v4)
│   │         camara_tf.py · camara_pt.py         (cámara OpenCV)
│   └── v6/  modelotf_v6.py · modelopt_v6.py     (consumo v6, cabeza Optuna)
├── web/        index.html · imagen_prueba.jpg · paridad.txt   (smoke test TF.js)
├── imagenes_a_probar/           imágenes sueltas para el consumo
├── documentacion/
│   ├── deck_a_imagenes.py           obtención: PDF/PPTX → PNG
│   ├── verificar_dataset.py         chequeo del dataset antes de entrenar
│   ├── crear_datos_prueba.py        datos sintéticos para probar el flujo
│   ├── crear_subset_desbalanceado.py  subset 100/50/25 (v5)
│   ├── reporte_metricas.py          reporte por clase de los v4 sin re-entrenar
│   ├── analisis_metricas.md         análisis con números reales + COMPLETAR: Bato
│   ├── caso_comercial.md            uso comercial (req. 6)
│   └── avances_proyecto.md          histórico + nota de roadmap (TFLite → TF.js)
├── README.md                    propósito + pipeline de datos + cómo correr
└── DetalleProyecto.md           este documento (resumen completo del repo)
```

---

## 5. Requisitos de la entrega 2 — estado

| Req | Qué pedía | Estado |
|-----|-----------|--------|
| 1 | Scripts, no jupyters | ✅ (nunca se introdujeron notebooks) |
| 2 | Separar obtención del entrenamiento | ✅ `verificar_dataset.py` + README |
| 3 | Script de consumo independiente | ✅ espejo TF+PT en `modelos/v4/` |
| 4 | OpenCV para cámara | ✅ `camara_tf.py` / `camara_pt.py` |
| 5 | ≥2 modelos + uno desbalanceado | ✅ arco CNN→MobileNetV3 + v5 A/B con pesos |
| 6 | Uso comercial | ✅ `caso_comercial.md` |
| 7 | Métricas analizadas + plan | ✅ `analisis_metricas.md` (prosa final la escribe Bato) |
| Optuna | Búsqueda de hiperparámetros (mandatorio 2ª entrega) | ✅ **v6 espejo** `TensorFlow/v6/` + `PyTorch/v6/` (TPE+MedianPruner sobre la cabeza) |

---

## 6. Qué falta implementar

### Entrega final (después de la entrega 2)
- **T7 — Esqueleto Ionic** (Angular blank): una página cámara `getUserMedia` → botón
  Analizar → frame a canvas → **OpenCV.js** (resize/perspectiva) → tensor **TF.js** →
  modelo desde `assets/model/` (el export de T6) → 3 barras. `node_modules/` al gitignore.
  *El smoke test de `web/index.html` ya des-riesga la parte del modelo.*
- **T8 — Opcionales con señal del profe** (solo si sobra tiempo): fine-tuning en dos fases
  (descongelar las últimas capas del backbone); curvas ROC one-vs-rest.
  *(Optuna ya está hecho: es la v6, ver abajo.)*

### Hecho en esta iteración
- **v6 — Optuna (mandatorio 2ª entrega):** `TensorFlow/v6/06_scripts.py` y
  `PyTorch/v6/06_scripts.py` (espejo). Busca los hiperparámetros de la **cabeza** con
  TPE+MedianPruner maximizando `val_accuracy`, con el backbone congelado y las features
  **pre-computadas una sola vez** (embeddings 576-d) para que la búsqueda corra en minutos.
  Consumo espejo en `modelos/v6/`. `optuna` agregado a ambos `requirements.txt`. Correr los
  scripts en la máquina del alumno para llenar los `Resultado_6` y las figuras.
- **Carpeta `estudio/` (fuera de git):** `resumen_estudio.md` (métricas explicadas + análisis
  de la última versión + preguntas-trampa) y `slides.md` (Marp → `slides.html`/`slides.pdf`)
  para preparar la interrogación oral.

### Tareas de Bato (Code no las hace)
1. **Prosa del análisis y plan de acción** en `analisis_metricas.md` (interrogación oral
   individual — hay bloques `COMPLETAR: Bato` con preguntas guía).
2. **Validar con el profe** dos interpretaciones (§3.2 de la guía): el arco de 2 modelos y
   que "un modelo desbalanceado" = entrenado con clases desbalanceadas.
3. **Renombrar** la carpeta local `data/` → `dataset/` (hoy hay un symlink para las pruebas).
4. **Protocolo de etiquetado 1↔2** (pendiente histórico; los errores 0→1 lo mantienen vivo).
5. **Fotos reales** (teléfono → diapo) por los scripts de cámara: primera medición del
   *domain gap* (los números actuales son sobre renders limpios = techo optimista).
6. **Ampliar la clase 2** si se decide crecer el dataset (fija el techo de tamaño).

### Límites honestos (documentados en `analisis_metricas.md`)
- Validación chica (60 imgs v3/v4, ~35 v5) y sin estratificar → leer recalls en absolutos.
- Todo sobre renders limpios; el *domain gap* no está medido aún.
- Split distinto entre TF y PyTorch → comparar sus % lado a lado tiene un asterisco.

---

## 7. Cómo reproducir (comandos clave)

```bash
# 1) Verificar el dataset
python documentacion/verificar_dataset.py

# 2) Entrenar (ejemplo v4)
python TensorFlow/v4/04_scripts.py      # venv tensorflow-ia
python PyTorch/v4/04_scripts.py         # venv frameworks-ia

# 3) Reporte por clase de los v4 ya entrenados (sin re-entrenar)
python documentacion/reporte_metricas.py

# 4) v5 desbalanceado (genera el subset y entrena las 2 variantes)
python documentacion/crear_subset_desbalanceado.py
python TensorFlow/v5/05_scripts.py ; python PyTorch/v5/05_scripts.py

# 5) v6 Optuna (busca hiperparámetros de la cabeza; requiere 'optuna' instalado)
python TensorFlow/v6/06_scripts.py ; python PyTorch/v6/06_scripts.py

# 6) Consumo y cámara
python modelos/v4/modelotf_v4.py ; python modelos/v4/camara_pt.py
python modelos/v6/modelopt_v6.py        # consumo del modelo v6 (cabeza Optuna)

# 6) Export a TensorFlow.js + smoke test (venv aparte para tensorflowjs)
python TensorFlow/export_tfjs/exportar_tfjs.py
cd web && python -m http.server        # abrir http://localhost:8000/
```

---

## 8. Commits de la entrega 2

```
T0  unificar DATA_DIR a dataset/ + guard de existencia
T1  verificar_dataset.py + README pipeline
T2  consumo espejo v4 + cámara OpenCV + fix ruta consumo v2
T3  v5 clases desbalanceadas + corrección con pesos
T4  métricas por clase + esqueleto de análisis
T5  caso comercial
T6  export a TensorFlow.js + smoke test con paridad
```

Un commit por tarea; todos los entrenamientos corridos de verdad (números reales).
