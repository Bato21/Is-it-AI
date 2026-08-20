# v10 — La compuerta de rechazo

> **La pregunta que originó esta versión:** *«¿cómo evito que se analicen cosas que no son
> presentaciones? Y si meto imágenes de otras cosas, ¿cuántas deberían ser para que funcione
> bien y no sea desproporcionado?»*
>
> Este documento responde las dos: la primera con una decisión de arquitectura, la segunda
> con un razonamiento **y una medición**.

---

## 1. El problema: un softmax de 3 clases no puede abstenerse

Hasta la v9 el modelo tenía tres salidas y ninguna de ellas era «esto no es una diapositiva».
Eso no es un detalle de interfaz, es una imposibilidad matemática: un softmax reparte 1.0
entre sus opciones y no tiene ninguna donde poner la masa cuando la entrada no pertenece a
ninguna. Si el usuario apunta la cámara a su escritorio, a la cara de un compañero o a una
planilla de Excel, el modelo **tiene** que devolver un nivel de saturación de IA.

Y no lo devuelve dudando. Sobre entradas fuera de distribución el argmax suele salir con
confianza alta, porque la red nunca vio nada que la obligara a repartir masa fuera de sus tres
opciones. El resultado es el peor modo de falla posible para un producto: no falla «un poco»,
**responde con seguridad a una pregunta que nadie hizo**.

El test de la v9 no podía detectarlo porque el test de la v9 está hecho solo de diapositivas.
Medía muy bien un problema que en producción aparece menos veces que este.

## 2. La decisión: una cuarta clase, no un segundo modelo

Había tres caminos:

| opción | qué es | por qué no / por qué sí |
|---|---|---|
| **umbral de confianza** | rechazar si `max(softmax) < τ` | No requiere datos nuevos, y es lo que primero se le ocurre a cualquiera. Falla justo donde hay que confiar: una red sobre-confiada da 0.95 sobre una foto de un perro. Además τ sería un hiperparámetro más ajustado sobre un test chico. |
| **modelo detector aparte** | un clasificador binario «¿es una diapositiva?» delante | Correcto conceptualmente, caro en despliegue: **seis** artefactos en la app en vez de tres (uno por cada modelo del selector), seis exportaciones, seis verificaciones de paridad, y **dos pasadas por imagen** en el teléfono más lento. |
| **cuarta clase** ✅ | `3_no_diapositiva` en el mismo softmax | Un solo artefacto por modelo, una sola pasada, y la decisión sale del `argmax` sin umbral que elegir. |

**Se eligió la cuarta clase.** El costo de la decisión es que el eje ordinal deja de abarcar
todas las clases, y eso hay que manejarlo en las métricas en vez de ignorarlo (§4).

> `3_no_diapositiva` **no es un cuarto nivel de saturación**. No significa «todavía más
> generada por IA que la 2»: está *fuera* del eje. Esa distinción se propaga a todo — al
> cálculo de QWK, al color en la interfaz (gris sin croma, no el siguiente rojo de la rampa)
> y al veredicto que se muestra, que es la *ausencia* de veredicto.

## 3. Cuántas imágenes: el razonamiento y la medición

### 3.1 El razonamiento

Las tres clases existentes tienen 434 / 435 / 468 (promedio **446**). Se eligió **450**, y las
dos cotas que lo justifican son:

- **Cota superior — distorsión de la prior.** La clase de rechazo es la más *fácil* de las
  cuatro: separar «diapositiva» de «no diapositiva» es un margen visual enorme comparado con
  separar «rastro» de «saturada». Si fuera mucho más grande que las demás dominaría las
  métricas macro y la accuracy reportada subiría sin que el producto mejorara en lo que
  importa. Además obligaría a meter pesos de clase, rompiendo la comparabilidad con la v9
  (que entrena con cross-entropy plana).
- **Cota inferior — varianza intra-clase.** «Todo lo que no es una diapositiva» es un
  conjunto abierto: no es un concepto, son al menos cuatro familias visuales distintas. Con
  menos de ~100 por familia el modelo memoriza ejemplos en vez de aprender el complemento.

450 satisface las dos: es **un cuarto exacto** del dataset resultante (450 / 1787 = 25.2 %),
o sea el reparto neutro de un problema de cuatro clases, y deja ~112 por familia. Se
obtuvieron **449** (una fuente devolvió una imagen menos); la diferencia es irrelevante.

### 3.2 La composición importa más que el número

Si la clase de rechazo fueran solo fotos de perros y playas, el modelo aprendería la frontera
trivial *«¿hay una pantalla rectangular con texto?»* y seguiría analizando como diapositiva la
foto de un monitor con una planilla abierta — que es el caso que de verdad pasa. Por eso se
estratifica en dos mitades:

| familia | n | dominio | fuente | qué cubre |
|---|---:|---|---|---|
| escena | 160 | foto | SUN397 | habitaciones, oficinas, aulas, calles |
| objeto | 140 | foto | COCO 2014 | personas, objetos, escritorios, comida |
| interfaz | 100 | render | wave-ui + website-screenshots | **negativo difícil**: una pantalla que no es una diapositiva |
| documento | 49 | render | DocVQA + FUNSD | **negativo difícil**: texto sobre fondo blanco |
| **total** | **449** | 300 foto / 149 render | | |

Los 149 **negativos difíciles** son la mitad que decide si el producto sirve. Al declararse
como dominio `render`, la augmentation les aplica la simulación de foto de pantalla a fuerza
1.0: entrenan como *«captura de una pantalla mostrando algo que no es una diapositiva»*.

### 3.3 El guard que evita el atajo

Un error silencioso y fatal habría sido dejar que **toda** la clase de rechazo quedara marcada
como `render`. En ese caso, en el test las clases 0-2 serían fotos y la clase 3 renders, y
separar la clase 3 se volvería trivial: basta detectar la textura de render. El recall de
rechazo saldría ~1.00 en el laboratorio y **la app fallaría con la primera foto real de un
escritorio**.

Dos piezas lo impiden:

1. `documentacion/negativos_v10.py` escribe un **manifiesto de procedencia**
   (`dataset/3_no_diapositiva/_procedencia.json`) declarando el dominio de cada archivo, y
   `documentacion/dominio.py` lo respeta por encima de sus heurísticos de EXIF y nombre —
   evidencia declarada en la obtención, más fuerte que cualquier inferencia.
2. `documentacion/particion_v10.py` **aborta** si la clase de rechazo tiene menos de 100 fotos.

Resultado verificado en la partición: la compuerta se evalúa sobre **60 fotografías de cámara
reales**, en línea con las 65 / 68 / 74 de las otras tres clases.

### 3.4 La medición: la ablación

Un razonamiento no es una medición. `TensorFlow/v10/10_scripts.py` §5.7 entrena la misma
cabeza con el **25 %**, el **50 %** y el **100 %** de la clase de rechazo — mismas semillas,
mismo test, corriendo sobre embeddings pre-computados — y reporta:

- el **recall de la compuerta** (lo que se gana),
- el **recall de la clase 0** (lo que se puede canibalizar: una diapositiva humana sobria es
  texto negro sobre blanco, visualmente lo más parecido a un documento escaneado).

| imgs de rechazo | recall compuerta | fugas | rechazos indebidos | recall clase 0 | **0 → compuerta** | accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 90 | 0.9667 ± 0.018 | 2.0 | 0.0 | 0.9169 | **0.0** | 0.9341 |
| 180 | 0.9700 ± 0.031 | 1.8 | 0.0 | 0.8646 | **0.0** | 0.9318 |
| **359** | **0.9967 ± 0.007** | **0.2** | 0.2 | 0.8338 | **0.0** | 0.9438 |

*(Números en `TensorFlow/v10/resultados_v10.json` → `ablacion_clase_rechazo`; figura en
`TensorFlow/v10/Figure_ablacion_rechazo_v10.png`.)*

**La columna que resuelve la duda es `0 → compuerta`.** El recall de la clase 0 baja 8 puntos
al agrandar la clase de rechazo, y la lectura obvia sería *«la compuerta se está comiendo
diapositivas humanas sobrias»*. Es falsa: **cero** diapositivas de la clase 0 terminan
rechazadas, en los tres tamaños. Lo que se pierde se pierde *dentro* del eje ordinal (0→1,
0→2) — es la frontera vieja entre «sin IA» y «rastro de IA», no un efecto de la versión nueva.

Sin esa columna el informe habría concluido lo contrario, y habría recomendado achicar una
clase que no estaba causando el problema. Es exactamente por eso que la ablación reporta las
dos cosas y no solo el recall.

El criterio de lectura se declara antes de correr: si duplicar de ~225 a ~450 mueve el recall
**menos que la suma de los desvíos entre semillas**, la curva ya está plana y el informe no
puede afirmar que las 450 sean necesarias.

> **Dos lecturas del mismo resultado, y hay que dar las dos.** El criterio declarado mira el
> *recall* y dice «ruido». Pero el *conteo de fugas* baja de forma monótona en los tres puntos
> (2.0 → 1.8 → 0.2), y la fuga es lo que el producto realmente paga. Con 60 imágenes de test
> el recall es una métrica gruesa —cada imagen vale 1.7 puntos— así que «dentro del ruido» y
> «bajó 9 veces» no se contradicen: describen la misma tabla con distinta resolución.
>
> Quedarse solo con la lectura que conviene sería elegir el resultado. La conclusión honesta
> es: **450 no está demostrado como necesario, y tampoco sobra.** Se conserva porque el costo
> es cero y porque deja margen para negativos más difíciles.

## 4. Qué hubo que cambiar en las métricas

Con la cuarta clase fuera del eje, tres métricas del proyecto dejaban de tener sentido tal
como estaban:

| métrica | problema | solución |
|---|---|---|
| **QWK** y **MAE ordinal** | tratarían «no es una diapositiva» como un cuarto escalón, penalizando confundir 0 con 3 cuatro veces más que 0 con 1 | se calculan sobre el sub-bloque `cm[:3, :3]` y se leen como **condicionales**: «de lo que el modelo aceptó como diapositiva, qué tan bien lo ordenó» |
| **errores 0↔2** | `cm[n-1, 0]` apuntaba a «la última clase», que dejó de ser la 2 | índice explícito del eje, no «la última» |
| **recall clase 2** | ídem | índice explícito. Se calcula sobre la fila completa: una diapositiva saturada mandada a rechazo cuenta como fallo, que es lo correcto — el usuario no obtuvo su alerta y el motivo le da igual |

Y se agregaron cuatro métricas de compuerta, con los **dos errores contados por separado
porque no son simétricos**:

- `recall_rechazo` — de lo que no era diapositiva, cuánto atajó.
- `precision_rechazo` — de lo que rechazó, cuánto no era diapositiva.
- `fuga_no_diapositiva` — **el fallo que la v10 vino a cerrar**: analizó algo que no era.
- `rechazo_indebido` — el costo: diapositivas reales descartadas. Menos grave que la fuga (el
  usuario reencuadra y vuelve a disparar), pero es el que puede crecer si la clase de rechazo
  se agranda de más.

Todo esto está detrás de `n_ordinales`: sin ese argumento, `metricas.py` se comporta
**exactamente** como en la v9 y los resultados de v1–v9 siguen siendo reproducibles.

## 5. Resultados

### 5.1 La compuerta funciona, y el eje ordinal no se resiente

Modelo A (TensorFlow, fase 2), media de 5 semillas sobre el test de fotos:

| | v9 (3 clases) | **v10 (4 clases)** |
|---|---:|---:|
| accuracy | 0.9478 | **0.9558** |
| macro F1 | — | **0.9567** |
| recall clase 2 | 0.9730 | 0.9730 |
| errores 0↔2 | — | 6.0 |
| QWK (ordinal, condicional) | — | 0.8941 |
| ECE | — | 0.0255 |
| **recall de rechazo** | *(no existía)* | **0.9967 ± 0.0067** |
| **fugas** | *(60 de 60: el 100 %)* | **0.2 de 60 (0.3 %)** |
| **rechazos indebidos** | — | **0.2 de 207** |

> **El número que responde la pregunta original:** de las 60 fotos del test que no son
> diapositivas, la v9 habría analizado las 60 —no podía hacer otra cosa— y la v10 analiza
> **0.2 en promedio**. Y no lo paga en el eje: la accuracy *subió* respecto de la v9 pese a
> tener una clase más, y el recall de la clase 2 quedó idéntico.

Matriz acumulada (5 semillas, fase 2):

```
                    predicho ->
                    0     1     2     3
real  0_sin_ia    [294   10    21     0]     recall 0.905
      1_rastro_ia [  8  323     8     1]     recall 0.950
      2_saturada  [  9    1   360     0]     recall 0.973
      3_no_diapo  [  0    1     0   299]     recall 0.997
```

La columna 3 es la que importa para juzgar el costo de la compuerta: **una sola** diapositiva
real (de la clase 1) terminó rechazada en 1035 casos. La compuerta **no** se come diapositivas.

### 5.2 A y B no se equivocan igual, y eso es información

Los dos modelos ven exactamente las mismas vistas y la misma partición; lo único que difiere
es el framework y los hiperparámetros que Optuna eligió en la v7 de cada uno. Aun así se
comportan de forma distinta **en la compuerta**, y la diferencia es sistemática:

| | Modelo A (TensorFlow) | Modelo B (PyTorch) |
|---|---:|---:|
| accuracy | **0.9558** | 0.9390 |
| macro F1 | **0.9567** | 0.9410 |
| recall de rechazo | **0.9967** | 0.9700 |
| precisión de rechazo | 0.9967 | **1.0000** |
| fugas (de 60) | **0.2** | 1.8 |
| rechazos indebidos (de 207) | 0.2 | **0.0** |

**A es agresiva, B es conservadora.** B nunca rechaza una diapositiva real —precisión 1.000,
cero rechazos indebidos— y lo paga fugando 9 veces más. A ataja casi todo y a cambio descarta
una diapositiva cada tanto.

Para el producto la fuga es el error más caro (un veredicto inventado sobre el escritorio de
alguien) y el rechazo indebido el más barato (el usuario reencuadra), así que **A es el
default correcto**. Pero tener las dos posturas en el selector deja de ser cosmético: es una
elección real entre «no me molestes con rechazos» y «no me inventes respuestas».

Que los dos frameworks lleguen a una compuerta que funciona (0.97 y 0.997) con
hiperparámetros distintos es la validación cruzada del resultado: **la compuerta es una
propiedad de los datos, no del bucle de entrenamiento**.

Un detalle de la matriz de B que vale mirar: sus 7 fugas acumuladas caen todas en
`0_sin_ia` — son los **documentos escaneados**, la familia diseñada precisamente como el
negativo más difícil. La estratificación de §3.2 estaba apuntando al lugar correcto.

### 5.3 El modelo C: la compuerta sobrevive con la mitad de los negativos

El modelo C entrena con un dataset deliberadamente desequilibrado y **solo el 50 % de la clase
de rechazo** (ver §2 de su script). Sus tres estrategias sobre el mismo test:

| estrategia | macro F1 | recall c2 | **recall rechazo** | accuracy | fugas |
|---|---:|---:|---:|---:|---:|
| C0 sin corrección | 0.8418 | 0.6351 | 0.9722 | 0.8365 | 1.7 |
| C1 class weights | 0.8853 | 0.7658 | 0.9722 | 0.8814 | 1.7 |
| **C2 Focal + sampler** ✅ | **0.9024** | **0.8333** | **0.9778** | 0.8989 | 1.3 |

Dos lecturas:

1. **La corrección del desequilibrio funciona y la accuracy lo esconde.** El recall de la
   clase 2 sube de 0.635 a 0.833 entre C0 y C2 — 20 puntos sobre la clase que el producto
   necesita detectar — mientras la accuracy sube apenas 6. Es el argumento de siempre del
   proyecto: con desequilibrio, la accuracy es la métrica que miente.
2. **Tercera confirmación independiente del tamaño de la clase de rechazo.** C entrena con la
   mitad de los negativos y su compuerta rinde 0.978, prácticamente lo mismo que A y B con el
   100 %. Coincide con lo que dijo la ablación de §3.4, medido por otro camino y con otra
   arquitectura.

### 5.4 Verificación de punta a punta

| paso | resultado |
|---|---|
| PyTorch → ONNX (modelo B) | max\|dif\| **2.67e-05**, 12/12 clases coincidentes |
| PyTorch → ONNX (modelo C) | max\|dif\| **9.12e-06**, 12/12 clases coincidentes |
| Keras → TensorFlow.js (modelo A) | max\|dif\| **2.24e-08**, misma clase predicha |
| salida del modelo desplegado | `(None, 4)` — la cuarta clase llegó al navegador |
| consumo sobre imágenes del test nunca vistas | 3/3 no-diapositivas rechazadas al 100 %, 2/2 diapositivas con su nivel |

<!-- RESULTADOS_V10 -->

## 6. Qué cambió en la app

- **Cuarta clase en el catálogo.** `catalogo.json` ahora declara `clases` y `nOrdinales`, y el
  `InferenciaService` los lee de ahí en vez de tenerlos compilados: reentrenar con otra
  cantidad de clases actualiza la app sin recompilar.
- **Un veredicto distinto, no un cuarto nivel.** Cuando gana la compuerta, la hoja de
  resultado dice «Sin veredicto → No es una diapositiva» y explica por qué no informa nivel.
  Presentarla con la misma plantilla que un nivel sugeriría una escala de cuatro escalones que
  no existe — sería arreglar el fallo en la red y volver a introducirlo en la interfaz.
- **Color fuera de la rampa.** `--c3` es un gris azulado sin croma. Darle el siguiente color
  caliente de la escala (0 verde → 1 ámbar → 2 rojo) afirmaría visualmente que hay un cuarto
  nivel de huella de IA.
- **El umbral de confianza baja sigue al azar.** Era 0.60 fijo, calibrado sobre 3 clases
  (azar 33 %). Con 4 clases el azar es 25 %; dejarlo clavado habría convertido el aviso en
  ruido. Ahora es `1.8 × azar`, o sea 0.45 con cuatro clases.
- **`recall rech.` en el selector.** Es parte de lo que distingue a un modelo de otro tanto
  como la accuracy, así que se muestra al elegir.

## 7. Limitaciones honestas

- **El conjunto abierto sigue siendo abierto.** 449 imágenes de cuatro familias no cubren
  «todo lo que no es una diapositiva». Un negativo de una familia no representada (por
  ejemplo, un póster académico, que es casi una diapositiva gigante) puede pasar la compuerta.
- **Los negativos difíciles son sintéticamente «fotografiados».** Las capturas de interfaz y
  los documentos se convierten en fotos de pantalla vía `aug_pantalla.py`, no se fotografiaron
  de verdad. Es la misma concesión que la v9 ya hacía con sus 300 renders, y la misma
  respuesta: el test de la compuerta es 100 % fotografías reales, así que la métrica reportada
  no depende de la simulación.
- **La frontera clase 0 / documento es la más fina de todas.** Una diapositiva sobria de texto
  negro sobre blanco y un documento escaneado se parecen mucho. Por eso la ablación reporta el
  recall de la clase 0: es el número que avisaría si la compuerta se la está comiendo.
- **La ablación corre solo en TensorFlow.** Es una pregunta de diseño de datos, no de
  framework: la respuesta no puede depender de si el bucle está escrito en Keras o en PyTorch,
  y si dependiera el problema sería que el espejo está roto. La coincidencia entre el recall de
  compuerta de A y el de B es la validación cruzada, y sale gratis.

## 8. Reproducir

```bash
python documentacion/negativos_v10.py         # 449 imgs + manifiesto de procedencia
python documentacion/particion_v10.py         # partición de 4 clases, con guards
TensorFlow/.venv/Scripts/python TensorFlow/v10/10_scripts.py
PyTorch/.venv/Scripts/python    PyTorch/v10/10_scripts.py
PyTorch/.venv/Scripts/python    PyTorch/v10/10_modelo_c_desequilibrado.py
```
