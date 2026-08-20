# Guía para la presentación — qué hace cada función y qué contestar

*Para la muestra con el profe: que la app funcione en el teléfono y que se note que entendemos
qué hace. Las respuestas están escritas como para decirlas en voz alta, no para leerlas.*

---

# 1. La demo, en orden

1. Abrir la app (Vercel, HTTPS — la cámara no funciona sobre `http://192.168.x.x`).
2. **Precargar los modelos** desde la pantalla de modelos ANTES de empezar. La primera inferencia
   en WebGL es lentísima y no queremos que el profe vea la barra de carga.
3. Apuntar a una diapositiva en pantalla. Mostrar el HUD: **nitidez, FPS y la retícula** que salta
   a las esquinas de la pantalla cuando OpenCV la detecta.
4. Disparar. Mostrar que **el visor se congela en el frame que se analiza** — no sigue moviéndose
   mientras se mide otra cosa.
5. Mostrar el resultado con **las tres probabilidades**, no solo la clase ganadora.
6. **Mover el teléfono a propósito** y disparar: la app rechaza la foto por falta de nitidez. Eso
   demuestra que OpenCV no está de adorno.
7. Cambiar de modelo en el selector y repetir. Después **Comparar los 3** sobre la misma foto.

**La frase de apertura:** "A partir de la foto de una diapositiva estimamos cuánta huella de IA
tiene, en 3 niveles. Todo corre dentro del teléfono, no sale ninguna imagen. Hay tres modelos
cargados y se pueden comparar entre ellos."

---

# 2. El recorrido de una foto (esto es lo que hay que saber contar)

Cuando apretás el obturador pasa esto, en este orden. Cada paso es una función concreta:

```
  video (getUserMedia)
    │
    │  bucle()                 corre 60 veces por segundo, FUERA de la zona de Angular
    │   ├─ dibuja el frame al <canvas>
    │   ├─ opencv.nitidez()               cada 5 frames  -> el medidor del HUD
    │   └─ opencv.detectarPantallaRapido() cada 15 frames -> mueve la retícula
    │
    ▼  [ APRETÁS EL BOTÓN ]  analizar()
    │
    ├─ 1. opencv.analizarCalidad(lienzo)      ¿está nítida? ¿no está quemada?
    │                                          si no -> se rechaza y NO se molesta al modelo
    ├─ 2. opencv.prepararParaModelo(lienzo, 224)
    │        ├─ detectarPantalla()             busca el cuadrilátero del monitor
    │        ├─ enderezarPantalla()            warpPerspective -> cuadrado de 224
    │        └─ redimensionarCuadrado()        (si no encontró pantalla) resize INTER_AREA
    │
    ├─ 3. inferencia.predecir(canvas)         ...o predecirConTodos() si es "Comparar los 3"
    │        ├─ cargar(id)                     import dinámico + sesión cacheada
    │        ├─ preprocesar(canvas, orden)     píxeles 0-255 -> NHWC o NCHW
    │        ├─ [ el modelo ]                  TF.js sobre WebGL  |  ONNX sobre WASM
    │        └─ softmax(logits)                solo para B y C, que devuelven logits
    │
    └─ 4. abrirHoja()                          las 3 probabilidades + confianzaBaja()
```

**Lo que hay que poder decir sobre este dibujo:**

- **El modelo no ve la foto cruda.** Ve un cuadrado de 224×224 enderezado. OpenCV le devuelve la
  imagen al dominio en el que fue entrenado en vez de pedirle que adivine.
- **El filtro de calidad va antes que el modelo, a propósito.** Un modelo siempre responde: sobre
  una foto movida donde no se lee nada responde con confianza alta y se equivoca. Mejor no
  preguntarle.
- **La página de cámara no sabe qué modelo corre.** Llama a `predecir(canvas)` y recibe
  probabilidades. Adentro puede haber TensorFlow.js u ONNX; afuera se ve igual.

---

# 3. Índice de funciones

## 3.1 La app — `app/src/app/`

### `servicios/opencv.service.ts` — todo lo que hace OpenCV.js

| Función | Qué hace |
|---|---|
| `listo` / `listo$` | Si el WASM de OpenCV ya cargó (son ~9 MB, tarda). La app **funciona igual sin él**: cae a un resize normal. |
| `nitidez(canvas)` | Varianza del Laplaciano. El Laplaciano es la segunda derivada, o sea responde a los bordes: imagen nítida = bordes fuertes = varianza alta. Movida = varianza baja. |
| `analizarCalidad(canvas)` | Nitidez **+ brillo medio**. Devuelve si la foto es apta y por qué no. Umbrales: nitidez > **60**, brillo entre **35 y 245**. |
| `detectarPantalla(canvas)` | `cvtColor` → `GaussianBlur` → `Canny` → `findContours` → `approxPolyDP`; se queda con el cuadrilátero de mayor área. Devuelve 4 esquinas o `null`. |
| `detectarPantallaRapido(canvas, 320)` | Lo mismo pero achicando a 320 px primero. Es la que corre en el bucle en vivo, para mover la retícula sin comerse los FPS. |
| `ordenarEsquinas(pts)` | Ordena las 4 esquinas (arriba-izq, arriba-der, abajo-der, abajo-izq). Sin esto `warpPerspective` puede entregar la imagen rotada o espejada. |
| `enderezarPantalla(canvas, esquinas, lado)` | `warpPerspective`: estira el cuadrilátero a un cuadrado de 224. |
| `redimensionarCuadrado(canvas, lado)` | Resize con **`INTER_AREA`**. Es el fallback cuando no aparece pantalla. |
| `prepararParaModelo(canvas, lado)` | El pipeline completo. Devuelve el canvas listo **y** si logró enderezar (la UI lo muestra). |

> **Memoria:** cada `cv.Mat` se libera con `.delete()` dentro de un `finally`. El heap de WASM no lo
> maneja el garbage collector de JavaScript: un Mat de 1280×720 son 3.7 MB que quedan reservados
> para siempre. A 30 fps, un `delete()` olvidado mata la pestaña en segundos.

### `servicios/inferencia.service.ts` — la capa que unifica los dos motores

| Función | Qué hace |
|---|---|
| `cargarCatalogo()` | Lee `assets/modelos/catalogo.json` (lo genera el script de export con las métricas reales). Si no está, usa `CATALOGO_FALLBACK` para que la app no arranque en blanco. |
| `seleccionar(id)` · `estaCargado(id)` · `descargar(id)` | Cambiar de modelo activo, saber si ya está en memoria, liberarlo. |
| `cargar(id)` | **Import dinámico** de `@tensorflow/tfjs` o `onnxruntime-web` (así el bundle inicial no arrastra los dos motores) + crea la sesión y **la cachea**. Crear la sesión cuesta cientos de ms; ejecutarla, decenas. |
| `preprocesar(canvas, orden)` | Lee los píxeles del canvas y los ordena en **NHWC** (TensorFlow) o **NCHW** (PyTorch). **Nada más**: no divide por 255, no resta medias. |
| `softmax(logits)` | Solo para B y C. El modelo A ya devuelve probabilidades. |
| `predecir(canvas, id?)` | Devuelve `{clase, confianza, probabilidades[3], ms, modeloId}`. **Siempre el vector completo**, nunca solo el argmax. |
| `predecirConTodos(canvas)` | Los tres modelos sobre la misma imagen. Es el botón "Comparar los 3". |
| `consenso(predicciones)` | Cuántos de los tres coinciden. Si discrepan, la imagen es genuinamente ambigua. |

### `servicios/catalogo-modelos.ts` — el registro de modelos

Constantes (`CLASES`, `DESCRIPCION_CLASES`, `ID_MODELO_POR_DEFECTO`), las interfaces y el catálogo
de respaldo. **Para agregar un cuarto modelo: una entrada acá + el archivo en `assets/`. Nada más.**

> `CLASES` está en el orden exacto en que Keras e ImageFolder ordenaron las carpetas
> (alfabético: `0_`, `1_`, `2_`). Cambiar ese orden rompe **todas** las predicciones sin que nada falle.

### `paginas/camara/camara.page.ts` — la pantalla

| Función | Qué hace |
|---|---|
| `iniciar()` | Pide la cámara con `getUserMedia` (trasera si hay). Necesita HTTPS o localhost. |
| `ajustarLienzo()` | Recorte centrado del video al canvas: se recortan píxeles, no se reescala. |
| `bucle()` | El `requestAnimationFrame`, **fuera de la zona de Angular**. Dibuja el frame, calcula FPS con **suavizado exponencial** (α=0.85), llama a `nitidez()` cada **5** frames y a `detectarPantallaRapido()` cada **15**. Solo vuelve a entrar a la zona para publicar esos números. |
| `analizar(conTodos)` | Congela el visor, filtro de calidad, preprocesa, infiere fuera de la zona, abre la hoja de resultado. Vibración distinta para "analizando" y para "foto rechazada". |
| `confianzaBaja(p)` | `true` si la confianza está bajo **0.60** → la UI avisa que no se confíe. |
| `liberarCamara()` · `detener()` · `reintentar()` | Cerrar el stream (si no, la cámara queda prendida), manejar errores de permiso. |
| `tituloClase()` · `detalleClase()` · `nombreModelo()` | Textos de la UI. |

**Rendimiento — las 4 decisiones, por si preguntan:** bucle fuera de `NgZone` (a 60 fps, cada frame
dispararía un ciclo completo de detección de cambios), `ChangeDetectionStrategy.OnPush` + signals,
sesiones cacheadas con warmup, y `willReadFrequently: true` en el contexto 2D (sin esa bandera cada
`getImageData` fuerza una transferencia GPU→CPU).

---

## 3.2 Python — entrenamiento

### `TensorFlow/v9/09_scripts.py` — modelo A

| Función | Qué hace |
|---|---|
| `preparar_datos()` | Lee la partición, arma las vistas (1 limpia + 2 aumentadas por imagen). |
| `construir_base()` | MobileNetV3-Small de ImageNet, `include_top=False`, congelada. |
| `construir_cabeza(params, ...)` | La MLP con los hiperparámetros que **Optuna eligió en la v7**. |
| `modelo_completo(base, cabeza)` | Une backbone + cabeza en un modelo end-to-end. |
| `descongelar(base, n_capas)` | **Fase 2**: descongela las últimas 12 capas (277.200 parámetros) dejando las BatchNorm congeladas. |
| `optimizador(params, lr)` | El optimizador; en fase 2 con `lr = 1e-5`. |
| `evaluar(logits, Y, clases)` | Llama a `metricas.resumen_completo`. |

Al final guarda `modelotf_v9_finetune.keras` (**la mejor semilla de la fase 2**, con la capa Softmax
pegada) + `modelo_v9_meta.json` + las figuras.

### `PyTorch/v9/09_scripts.py` — modelo B (espejo)

`lotes()`, `a_tensor()`, `logits_de()`, `evaluar()`, `optimizador()`. Es más corto porque las
arquitecturas viven aparte:

### `PyTorch/v9/modelos_v9.py` — las arquitecturas (fuente única)

| Elemento | Qué hace |
|---|---|
| `Normalizador` | **La capa clave del despliegue.** Mete la normalización ImageNet **dentro** del modelo, para que el `.onnx` también coma 0-255 y el navegador no tenga que normalizar nada. |
| `CabezaV9` · `ModeloV9` · `ModeloC` | La cabeza y los dos modelos completos. |
| `bn_a_eval(modulo)` | Pone todas las BatchNorm en `eval()`. **Hay que llamarla después de cada `model.train()`**, porque `train()` se propaga a los hijos y las vuelve a activar. |
| `descongelar_desde(modelo, bloque)` | Fase 2 en PyTorch: descongela `features[11]` y `features[12]`. |
| `cargar_modelo_b()` · `cargar_modelo_c()` | Las usan el exportador a ONNX y el script de consumo. **Tener la arquitectura una sola vez** evita exportar un modelo distinto del que se entrenó. |

### `PyTorch/v9/09_modelo_c_desequilibrado.py` — modelo C

| Función | Qué hace |
|---|---|
| `submuestrear_desequilibrado(Y, proporcion, semilla)` | Arma el subconjunto **9.4 : 2.8 : 1** (100% de la clase 0, 30% de la 1, **10% de la 2**). |
| `pesos_de_clase(Y, n)` | `w_c = n_total / (n_clases · n_c)`. |
| `focal_loss(logits, objetivo, alpha, gamma=2)` | `-α(1-p)^γ·log(p)`. El `(1-p)^γ` desinfla los ejemplos fáciles. |
| `hacer_perdida(estrategia, peso_clase)` | Devuelve la pérdida de C0 (plana), C1 (class weights) o C2 (focal). |
| `lotes(..., pesos)` | Con `pesos` hace el **muestreo ponderado** (el equivalente del `WeightedRandomSampler`). |
| `entrenar(estrategia, ...)` | Entrena una variante. Se llama 3 estrategias × 3 semillas sobre **exactamente los mismos datos**. |

Al final elige la ganadora **por macro-F1** (fue C2) y guarda esa.

---

## 3.3 Python — módulos compartidos (`documentacion/`)

| Archivo | Funciones y qué hacen |
|---|---|
| `dominio.py` | `dominio_de(ruta)` decide si una imagen es **foto o render**: primero mira el **EXIF de cámara**, después el nombre de archivo, y si nada dispara asume render. `mapa_dominios()` lo aplica a todo el dataset. |
| `particion_v9.py` | `construir_particion()` arma test + 5 folds **estratificados** con la regla de la v9: **test y validación 100% fotos, los renders solo entrenan**. `huella_dataset()` es un SHA-256 del listado: si se agrega o borra una imagen, `cargar_particion()` avisa en vez de entrenar en silencio sobre una partición vieja. `rutas_y_etiquetas()` es lo que llaman los dos frameworks. |
| `aug_pantalla.py` | La simulación de foto de pantalla, un efecto por función: `perspectiva`, `moire`, `glare`, `vineta`, `temperatura_color`, `ruido_gauss`, `desenfoque`, `jpeg`. `simular_foto_pantalla()` las combina y `augmentar(img, dominio, rng)` decide cuánta fuerza aplicar según si la imagen es foto o render. |
| `imagenes.py` | `cargar_una()`, `construir_cache()`, `cargar_cache()`: decodifica las imágenes **una sola vez** y las cachea. `generar_vistas()` arma las vistas aumentadas con **semilla fija 999**, así TF y PyTorch reciben **el mismo array de píxeles byte a byte**. |
| `metricas.py` | Todas las métricas, en **numpy puro** (sin torch, sin tf, sin sklearn) para poder importarse desde los dos venvs: `matriz_confusion`, `reporte_por_clase`, `errores_extremos` (la esquina 0↔2), `qwk`, `mae_ordinal`, `curvas_roc_ovr`, `ece`, `resumen_completo`, `agregar_semillas`, y los `plot_*`. |

---

## 3.4 Python — exportación y consumo

| Archivo | Qué hace |
|---|---|
| `export/exportar_onnx.py` | `exportar()` hace `torch.onnx.export` (opset 17, `dynamic_axes` solo en batch). `verificar_paridad()` compara PyTorch contra ONNX Runtime sobre **12 fotos reales del test** y **aborta si la diferencia supera 1e-4**. Resultado: **3.15e-05** (B) y **6.56e-06** (C). |
| `TensorFlow/export_tfjs/exportar_tfjs.py` | Keras → TensorFlow.js. |
| `export/verificar_paridad.py` | Levanta el modelo TF.js **en Node** y lo compara contra el `.keras`: **1.55e-06**. |
| `export/generar_catalogo.py` | Vuelca las métricas de los JSON de entrenamiento al `catalogo.json` que lee la app. **Los números de la app no están escritos a mano**: si se reentrena, se actualizan solos. |
| `modelos/v9/consumir_v9.py` | El script de consumo independiente. `cargar(clave)`, `preparar(img, ejes)`, `modo_carpeta()` (clasifica una carpeta con uno o los tres modelos y marca dónde discrepan) y `modo_camara()` (webcam con OpenCV). **Es la referencia contra la que se verifica la app**: si el navegador y este script no coinciden, el problema está en el cliente. |

---

# 4. Los tres modelos, en criollo

**Los tres son de la v9.** Los tres resuelven lo mismo (3 clases, 224×224) pero cambian las tres
cosas que uno elige al desplegar: **framework, arquitectura y forma de entrenar**. Si cambiáramos
solo el framework serían tres botones que hacen lo mismo.

### A — MobileNetV3 en TensorFlow.js · el que viene por defecto
Es el más rápido, el más liviano (4.1 MB) y **el único que corre sobre WebGL**, o sea usando la GPU
del teléfono. Además es el que mejor mide: **0.9478** de accuracy y **0.9730 de recall en la clase
2** — la clase que el producto existe para detectar. Y es el mejor calibrado (ECE 0.0294), que es lo
que hace que el aviso de "confianza baja" signifique algo.

*Sale de `TensorFlow/v9/09_scripts.py`, fase 2 (fine-tuning), semilla 42.*

### B — el mismo modelo pero entrenado en PyTorch, exportado a ONNX
Está por tres razones: **(1)** hace que el requisito de los dos frameworks siga vivo dentro de la
app y no solo en el laboratorio; **(2)** corre sobre **WASM**, así que es el respaldo si WebGL falla
en algún teléfono; **(3)** —la mejor— **se equivoca distinto que A**. El error típico de A es saltar
de 0 a 2 (el error caro); el de B es confundir 2 con 1 (vecinos, barato). Dos modelos que se
equivocan igual no sirven de nada juntos; estos no, así que **cuando discrepan es porque la imagen
es realmente ambigua**.

*Sale de `PyTorch/v9/09_scripts.py`, fase 2, semilla 7.*

### C — EfficientNet-B0 entrenada con las clases desequilibradas
Es el "modelo desequilibrado" que pedía el enunciado, y lo hicimos como experimento: sacamos el 90%
de la clase 2 (la que hay que detectar) y entrenamos **tres estrategias sobre los mismos datos**:

| | macro F1 | recall clase 2 | ECE |
|---|---:|---:|---:|
| C0 · cross-entropy normal | 0.7820 | 0.6351 | **0.0561** |
| C1 · class weights | 0.8374 | 0.7342 | 0.0578 |
| **C2 · Focal Loss + muestreo ponderado** ← la desplegada | **0.8649** | **0.8198** | 0.1373 |

Además es de **otra familia de arquitectura**, así que cuando A y B discrepan aporta un voto que no
está emparentado. Es el más débil de los tres **a propósito**, y en el selector se ven sus métricas
reales al lado de las de A.

*Sale de `PyTorch/v9/09_modelo_c_desequilibrado.py`, variante C2, semilla 7.*

### El detalle de las semillas (por si lo nota)
El archivo que está en la app es **una semilla** (la mejor), pero el catálogo muestra la **media de
las 5**, así que no coinciden: el `.keras` de A rinde 0.9613 y la app dice 0.9478. **Se reporta la
media porque es lo honesto** — publicar el máximo de 5 corridas es justamente el sesgo que
detectamos en la v7.

---

# 5. Preguntas y respuestas

*Respuestas para decir, no para leer. Cada una termina con el dato con el que se cierra.*

## Sobre la app

**¿Qué hace la app exactamente?**
Sacás una foto de una diapositiva y te dice cuánta huella de IA tiene, en 3 niveles: sin rastro,
con rastro, o saturada. No dice si "la hizo una IA" — eso no se puede saber de una imagen. Dice
cuántos artefactos visuales de generación tiene, que es lo único que una CNN puede aprender mirando
píxeles.

**¿Dónde corre el modelo?**
Adentro del teléfono. No hay servidor, no sale ninguna imagen. Es por privacidad —analizamos el
trabajo de alguien—, por latencia y porque no hay que pagar ni mantener un servidor de inferencia.
El precio es que hay que cuidar el peso del bundle.

**¿Para qué usás OpenCV? ¿No podrías hacer un resize y listo?**
Hace tres cosas y ninguna es decorativa. Primero **mide la nitidez** con la varianza del Laplaciano
y rechaza la foto si está movida: el modelo siempre te contesta, incluso sobre una foto donde no se
lee nada, y te contesta con confianza alta. Segundo, **encuentra la pantalla y corrige la
perspectiva**, así el modelo recibe la diapositiva derecha y ocupando todo el cuadro, que es como
fue entrenado. Tercero, al bajar de 1280 a 224 usa **INTER_AREA** en vez de bilineal, porque la
bilineal produce aliasing sobre el texto, que es la señal más informativa que tiene una diapositiva.
*(Y si quiere ser malo: el aporte del recorte todavía no lo medimos, está en el plan de acción.)*

**¿Y si OpenCV no carga?**
La app no se rompe: cae a un resize con el propio canvas y sigue funcionando. El WASM son 9 MB y
tarda, así que no puede ser un requisito duro.

**¿Cómo hacés para que la interfaz no se trabe?**
El bucle de frames corre **fuera de la zona de Angular** — a 60 fps, cada frame dispararía un ciclo
completo de detección de cambios. Vuelvo a entrar solo cada 5 frames para publicar FPS y nitidez.
Además la nitidez se calcula cada 5 frames y la detección de pantalla cada 15, porque la escena no
cambia entre frames consecutivos y el Laplaciano no es gratis. Y las sesiones de los modelos quedan
cacheadas: crearlas cuesta cientos de milisegundos, ejecutarlas decenas.

**¿Por qué se congela el visor al disparar?**
Porque si el video sigue moviéndose mientras se analiza otra cosa, la interfaz te está mintiendo
sobre qué está mirando el modelo.

**¿Cómo agregarías un cuarto modelo?**
Una entrada en el catálogo y el archivo en `assets/`. Nada más. La página de cámara no sabe qué
modelo corre, le pide una predicción al servicio y recibe probabilidades.

## Sobre los modelos

**¿Por qué esos tres y no tres iguales?**
Porque un selector con tres modelos idénticos son tres botones que hacen lo mismo. Estos cambian
framework, arquitectura y régimen de entrenamiento, así que cada uno mueve un compromiso distinto:
A es el rápido y el más preciso, B es el respaldo y el que verifica la paridad entre frameworks,
C es el desequilibrado y el voto de otra arquitectura.

**¿Por qué MobileNetV3 y no VGG16 o ResNet?**
Porque el destino es un teléfono. MobileNetV3-Small son 2.5 M parámetros, diseñada con búsqueda de
arquitectura para móviles, con convoluciones depthwise separable y bloques squeeze-and-excite.
VGG16 son 138 M parámetros: sobredimensionada para 1337 imágenes y sin ninguna chance en un
navegador. El archivo desplegado pesa 4.1 MB.

**¿Qué es transfer learning y por qué lo usaste?**
Agarrar una red ya entrenada en ImageNet, que ya sabe detectar bordes, texturas y formas, y
reemplazarle solo la última parte para nuestras 3 clases. Con 1337 imágenes entrenar desde cero no
alcanza. **El matiz interesante:** en la v8 medimos que con el backbone **congelado**, una CNN chica
entrenada desde cero le empataba o le ganaba a MobileNetV3 — porque las features de ImageNet están
hechas para distinguir perros de aviones, no para detectar plantillas de generador. Recién con
fine-tuning el transfer learning gana claro.

**¿Qué es el fine-tuning en dos fases?**
Primero entrenás solo la cabeza con el backbone congelado. Después descongelás el último bloque
convolucional y seguís con un learning rate chiquito, 1e-5. Los dos pasos son necesarios: si
descongelás desde el principio, los gradientes enormes de una cabeza inicializada al azar te
destruyen las features de ImageNet antes de que sirvan. Nos subió de 0.9256 a **0.9478**, y la
mejora superó el umbral de ruido en los dos frameworks por separado.

**¿Por qué solo el último bloque?**
Porque las capas de abajo aprenden bordes y colores, que sirven igual para diapositivas. Las de
arriba aprenden conceptos de ImageNet, que es lo que hay que reemplazar. Y descongelar todo con
mil imágenes es sobreajuste garantizado.

**El gotcha de BatchNorm** *(si toca hablar de PyTorch vs Keras, esta es la buena)*
En PyTorch, poner `requires_grad = False` congela los **pesos** de la BatchNorm pero **no las
estadísticas**: el `running_mean` y el `running_var` se siguen actualizando en cada forward mientras
el módulo esté en modo train, y te corrompen en silencio las estadísticas de ImageNet. Hay que
llamar `.eval()` sobre cada BatchNorm, y hay que volver a llamarlo **después de cada
`model.train()`**, porque `train()` se propaga a todos los hijos. En Keras esto no pasa:
`trainable = False` ya las pone en modo inferencia. En la v9 tiene consecuencia real: sin eso el
fine-tuning con lr 1e-5 se desestabiliza, porque las estadísticas se mueven más rápido que los pesos.

## Sobre los datos

**¿De dónde salieron las imágenes?**
De presentaciones reales, convertidas a PNG con `deck_a_imagenes.py`, y **clasificadas a mano**.
Después le sumamos **fotos de pantalla tomadas con el teléfono**, que son el 78% del dataset. En
total 1337 imágenes bastante balanceadas: 434 / 435 / 468.

**¿Por qué el test son solo fotos?**
Porque la app come fotos, no PDFs. Hasta la v8 medíamos sobre renders limpios, que es un problema
más fácil que el que el producto resuelve. La regla de la v9 es explícita: **test y validación 100%
fotos, los renders solo entrenan**. El número que reportamos es el que un cliente puede exigir.

**Entonces, ¿mejoró el modelo respecto de la versión anterior?**
No son comparables, y es a propósito. La v9 no cambió cuánto medimos, cambió **qué** medimos.

**¿Qué es el domain gap y cuánto te dio?**
La diferencia entre el dominio en el que entrenás y el dominio real. Lo medimos y **nos dio al
revés de lo que esperábamos**: los modelos aciertan entre 8 y 11 puntos **más sobre fotos que sobre
renders**. La razón es que el 78% del entrenamiento son fotos y que la augmentation le aplica
simulación de pantalla a los renders con fuerza máxima, así que el modelo casi nunca ve un render
limpio. La consecuencia hay que decirla: **este modelo ya no es bueno clasificando PDFs**. Si
mañana quisiéramos aceptar archivos subidos, no alcanza con reusarlo.

**¿Qué augmentation usás?**
Simulación de foto de pantalla: perspectiva, moiré, glare, viñeta, temperatura de color, ruido,
desenfoque y compresión JPEG. **No usamos flip horizontal** a propósito: una diapositiva espejada
tiene el texto al revés, algo que no existe en el mundo real. La augmentation tiene que ser
realista *para el dominio*.

**¿Cómo distinguís una foto de un render en el dataset?**
`dominio_de()`: primero el EXIF de cámara, después el nombre del archivo, y si nada dispara asume
render. Es conservador a propósito: etiquetar una foto como render solo pierde una imagen de
entrenamiento, mientras que lo contrario metería un render en el test y contaminaría la métrica.
*(Anécdota buena: 326 fotos se llamaban "Copia de 2026...", el prefijo rompía el patrón de nombre
y sin el EXIF habrían entrado como renders, dejando la clase 0 sin ninguna foto en el test.)*

## Sobre desequilibrio y métricas

**¿Qué pasa cuando las clases están desequilibradas?**
Lo interesante es **qué** se rompe. En nuestro experimento el AUC casi no se movió entre las tres
estrategias (0.961 / 0.966 / 0.965) — o sea, el modelo **seguía sabiendo** distinguir la clase rara.
Lo que se movió es el recall: 0.635 → 0.820. **El desequilibrio no le arruinó la vista, le arruinó
dónde pone la frontera**, porque el argmax hereda el corte que le enseñó la pérdida, y una pérdida
plana sobre datos 9 a 1 pone el corte donde le conviene a la clase grande.

**¿Y la accuracy no te avisó?**
No, y no de la forma clásica. Como el **test no está desequilibrado**, la accuracy no tenía dónde
esconderse: bajó junto con el recall. Donde sí se ve la patología es en la **matriz de confusión**:
56 de 222 diapositivas saturadas terminaron clasificadas como "sin IA", empujadas hacia la clase
mayoritaria. Con Focal Loss esos 56 bajan a 16. **La lección es que un solo número no diagnostica un
desequilibrio: hay que mirar la matriz.**

**¿Qué es Focal Loss?**
Cross-entropy multiplicada por `(1-p)^γ`. Ese factor **desinfla los ejemplos fáciles** —los que el
modelo ya clasifica bien— así el gradiente se concentra en los difíciles. Con γ=0 es cross-entropy
normal; usamos γ=2, que es el del paper. Más un muestreo ponderado que rebalancea el batch.

**¿Y qué te costó?**
**La calibración.** El ECE se fue de 0.056 a 0.137, se multiplicó por 2.4. El modelo C acierta más
pero **miente sobre cuánto sabe**: cuando dice 90% no acierta el 90%. La causa es directa, el
muestreo ponderado lo hace entrenar sobre una distribución de clases inventada. Consecuencia
práctica: el aviso de confianza baja con umbral 0.60 es confiable en A y B pero no en C. Lo
arreglaríamos con temperature scaling, y está anotado como pendiente.

**¿Por qué no usaste SMOTE?**
Porque SMOTE interpola entre vecinos y sobre píxeles eso te inventa imágenes fantasma que no son
diapositivas: creás un dominio que el modelo nunca va a ver. Lo defendible sería aplicarlo sobre los
embeddings, no sobre los píxeles. Para imágenes lo estándar es augmentation + pesos o sampler, que
es lo que hicimos.

**¿Qué métricas mirás y por qué esas?**
Accuracy para el titular, **macro F1** para que la clase grande no tape a la chica, **recall de la
clase 2** porque es la que el producto tiene que detectar, **errores 0↔2** porque son el error caro,
**QWK y MAE ordinal** porque las clases son ordinales y confundir extremos no es lo mismo que
confundir vecinos, **AUC** para ver la calidad del ranking sin depender del umbral, y **ECE** para
saber si las probabilidades que le mostramos al usuario significan algo.

**¿Cuál es el peor error de tu modelo?**
Decirle "saturada de IA" a una diapositiva hecha por una persona. Y es **asimétrico**: pasa 22 veces
contra 9 del error inverso. Son el **2.1%** de las fotos. Para una herramienta que puede terminar
señalando el trabajo de un alumno, ese es el número que hay que poner sobre la mesa, no el 94.8% de
accuracy. Sabemos la causa: el modelo lee el **moiré** de la pantalla como artefacto de generación.
Lo mitigamos aplicando moiré simulado a las tres clases por igual, para que deje de correlacionar
con la etiqueta — bajó pero no desapareció. El plan es atacarlo **con datos**: fotografiar
diapositivas humanas en las pantallas que más moiré generan.

**¿Usaste Grad-CAM?**
No, y está declarado como deuda. Lo que contestaría es si el modelo mira la estética de la
diapositiva o un artefacto del pipeline de render — porque la clase 0 sale de decks viejos y la
clase 2 se generó con Gamma, y son pipelines distintos. Se hace tomando el gradiente de la clase
ganadora respecto de las activaciones de la última capa convolucional.

## Sobre ONNX y despliegue

**¿Qué es ONNX y para qué lo usaste?**
Es un formato abierto que guarda el modelo como un grafo serializado en Protocol Buffers: los nodos
son las operaciones, los edges los tensores y los initializers los pesos. Sirve para desacoplar el
entrenamiento del despliegue — entrenás en PyTorch y corrés en ONNX Runtime, CoreML, TensorRT o
WebAssembly sin reescribir nada. Acá es **lo que hace que PyTorch llegue al navegador**: sin ONNX,
el requisito de los dos frameworks se moría en el despliegue.

**¿Cómo exportás desde PyTorch?**
`torch.onnx.export()`, que funciona por **tracing**: ejecuta un forward con un input de ejemplo y
graba las operaciones por las que pasó. Por eso el `dummy_input` es obligatorio, y por eso un `if`
que dependa de los datos no se graba bien —queda congelado en la rama que tomó— y para eso está
`torch.jit.script()`. Desde TensorFlow es distinto: `tf2onnx`, que lee el grafo estático y no
necesita input de ejemplo.

**¿Cómo sabés que el modelo del navegador es el mismo que el de Python?**
Porque está medido. Exportar es silencioso: te genera el archivo aunque el grafo esté mal, y el
error aparecería recién en el teléfono. Comparamos PyTorch contra ONNX Runtime sobre **12 fotos
reales del test** y el script **aborta** si la diferencia pasa 1e-4: nos dio **3.15e-05** y
**6.56e-06**, con 12 de 12 clases coincidentes. Para el modelo A levantamos el TF.js en Node y lo
comparamos contra el `.keras`: **1.55e-06**.

**¿Qué es el opset?**
El conjunto de operadores que entiende esa versión de ONNX. Fijamos **17**: más alto te da más
operadores pero menos runtimes que lo soporten. Está puesto explícitamente y no por defecto, porque
si cambia entre versiones de torch te genera un archivo distinto sin avisar.

**¿Y `dynamic_axes`?**
Le dice a ONNX qué dimensiones pueden cambiar en ejecución. Dejamos **solo el batch** dinámico; alto,
ancho y canales quedan fijos en 224. Si la resolución fuera dinámica el runtime no puede optimizar
el grafo con formas conocidas, y además la app siempre manda 224.

**¿Qué es un Execution Provider?**
El backend de hardware sobre el que corre ONNX Runtime: CPU, CUDA, TensorRT, CoreML, OpenVINO. Se
pasan en orden de preferencia y usa el primero disponible. En el navegador nos toca **WASM**. El
modelo A no usa ONNX: va por TensorFlow.js sobre **WebGL**, o sea la GPU del teléfono, y por eso es
el default.

**¿Qué chequeás antes de dar por buena una exportación?**
`model.eval()` antes de exportar —si queda en train, el dropout y la BatchNorm se graban en modo
entrenamiento y desplegás otro modelo—, opset ≥17, `dynamic_axes` solo donde corresponde,
`onnx.checker`, y la comparación numérica contra el original sobre datos reales.

**¿Por qué el preprocesamiento va dentro del modelo?**
Porque es el bug más caro de encontrar. Si la normalización viviera en el `transform` de Python,
habría que reescribirla en TypeScript, y si difiere en un decimal el modelo **funciona pero predice
raro** sin que nada falle ni avise. Metiéndola adentro del grafo, el navegador manda píxeles 0-255
y no puede equivocarse. En Keras venía de fábrica; en PyTorch le agregamos una capa `Normalizador`
a propósito para replicarlo.

## Sobre método (Optuna, semillas)

**¿Qué es Optuna?**
Búsqueda automática de hiperparámetros donde cada trial es una hipótesis. Usa **TPE**, que es
búsqueda bayesiana: parte los trials en buenos y malos, ajusta una densidad para cada grupo y
propone donde históricamente salieron los buenos — en vez de probar al azar o exhaustivamente. Lo
usamos para los hiperparámetros de la cabeza: capas, units, dropout, optimizador y learning rate.

**¿Y el MedianPruner?**
Corta los trials malos antes de que terminen: si en un paso intermedio el trial está por debajo de
la mediana de los anteriores, se abandona. Nos podó 12 de 30 trials en TensorFlow.

**Te dio 93% con Optuna y después reportaste 80%. ¿Qué pasó?**
Sesgo de selección, y es lo mejor que aprendimos. Estábamos maximizando la accuracy sobre una
validación de 60 imágenes y después **reportando esa misma validación**. El ganador de 30 intentos
sobre un examen chico gana en parte por mérito y en parte por suerte. Lo arreglamos con test
apartado que Optuna nunca ve, validación cruzada de 5 folds como objetivo, split compartido entre
los dos frameworks y 5 semillas. El mismo modelo dio 0.79-0.80: **14 puntos menos**, en los dos
frameworks por separado.

**¿Por qué 5 semillas?**
Porque una sola corrida no es un resultado. Con los mismos hiperparámetros, cambiando solo la
semilla, TensorFlow nos dio entre **0.6667 y 0.8889**. Una corrida podía reportar 88.9% o 66.7% con
la misma cara de honestidad. Por eso la regla de la casa es que **una diferencia solo cuenta si
supera la suma de los desvíos**, y ese criterio se declara antes de correr, no después.

---

# 6. Si hace la pregunta incómoda

**"¿Está listo para producción?"**
No. El número que importa no es el 94.8%, es el **2.1% de falsos positivos graves**. Y el test son
207 fotos, sacadas por una sola persona, en una sesión, con pocas pantallas: con ese tamaño una
imagen vale medio punto de accuracy. Lo que falta está priorizado con criterio de medición.

**"Te doy esta matriz de confusión, analizala."**
El orden para no perderse:
1. Decir la convención en voz alta: **filas = real, columnas = predicho**, orden 0/1/2.
2. Mirar los soportes: ¿el test está balanceado?
3. Recall por fila, precision por columna. Si una fila colapsa, esa clase no se detecta.
4. **Buscar el patrón:** ¿los errores son entre vecinos o entre extremos? En un problema ordinal el
   error de vecinos es barato y el de extremos es caro.
5. ¿Hay una clase mayoritaria absorbiendo a las otras? Eso es desequilibrio.
6. Terminar en una **decisión**, no en un número: más datos del caso que falla, o pesos en la
   pérdida, o mover el umbral, o revisar el criterio de etiquetado.

**Cosas que NO hay que decir**
- No comparar la accuracy de la v9 con la de versiones anteriores: miden problemas distintos.
- No decir que el ensamble o el recorte de OpenCV "mejoran" nada: **no están medidos**.
- No vender el modelo C como bueno: es el más débil a propósito.
- No decir "transfer learning siempre gana": con el backbone congelado la CNN propia empataba.
- No decir que la app detecta si algo "lo hizo una IA": detecta **huella visual**.

---

# 7. Chuleta de números

```
DATASET      1337 imgs = 434 / 435 / 468       1037 fotos + 300 renders
PARTICIÓN    TEST 207 (100% fotos) · TEST_RENDER 61 · folds 830 · solo-train 239

MODELO A     TF/Keras · MobileNetV3-Small · TF.js · WebGL · 4.1 MB · NHWC · softmax
             acc 0.9478 · F1 0.9475 · recall2 0.9730 · ECE 0.0294     (fase 2, semilla 42)
MODELO B     PyTorch · MobileNetV3-Small · ONNX · WASM · 4.6 MB · NCHW · logits
             acc 0.9372 · F1 0.9374 · recall2 0.9378 · ECE 0.0375     (fase 2, semilla 7)
MODELO C     PyTorch · EfficientNet-B0 · ONNX · WASM · 16.0 MB · NCHW · logits
             acc 0.8647 · F1 0.8649 · recall2 0.8198 · ECE 0.1373     (C2 focal, semilla 7)
             desequilibrio 9.4 : 2.8 : 1  (100% / 30% / 10%)

FINE-TUNING  fase 2 = último bloque · lr 1e-5 · 4 épocas · BatchNorm congeladas
BRECHA       aciertan 8-11 puntos MÁS sobre fotos que sobre renders
PARIDAD      TF.js 1.55e-06 · ONNX B 3.15e-05 · ONNX C 6.56e-06 · 12/12 clases
ERROR CARO   0→2 = 2.1% de las fotos  (22 casos contra 9 del inverso)
APP          nitidez > 60 · brillo 35-245 · confianza baja < 0.60
             nitidez cada 5 frames · detección de pantalla cada 15 · FPS con EMA α=0.85
```
