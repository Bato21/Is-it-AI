# Is it AI? — app Ionic

Detector de huella de IA en diapositivas. Ionic 8 + Angular 18 (standalone), con inferencia
**en el dispositivo**: ninguna imagen sale del teléfono.

```
cámara → canvas → OpenCV.js (calidad + encuadre) → modelo (TF.js u ONNX) → veredicto
```

---

## 1. Puesta en marcha

```bash
cd app
npm install          # dependencias + .wasm de ONNX Runtime + OpenCV.js (todo a assets/)
npm start            # http://localhost:8100
```

`npm install` deja la app lista para funcionar **sin red**: copia los binarios de ONNX Runtime
y baja OpenCV.js a `assets/`. Las tipografías ya están self-hosteadas (101 kB, variables).

Para probar desde el teléfono en la misma red:

```bash
npm run ionic:serve  # expone el puerto 8100 en 0.0.0.0
```

> **La cámara necesita contexto seguro.** Los navegadores solo entregan `getUserMedia` sobre
> `https://` o sobre `localhost`. Al abrir la app desde el celular con `http://192.168.x.x:8100`
> la cámara **no** va a funcionar. Opciones: usar `ionic serve --ssl`, un túnel (`ngrok http 8100`),
> o probar en el navegador de escritorio con la webcam.

### Antes de que la app tenga modelos

Los tres modelos se generan desde el repositorio principal. Si `src/assets/modelos/` está
vacío, la app abre igual pero el selector los muestra como no disponibles.

```bash
# 1) Entrenar (desde la raíz del repo)
TensorFlow/.venv/Scripts/python  TensorFlow/v10/10_scripts.py
PyTorch/.venv/Scripts/python     PyTorch/v10/10_scripts.py
PyTorch/.venv/Scripts/python     PyTorch/v10/10_modelo_c_desequilibrado.py

# 2) Exportar
TensorFlow/export_tfjs/.venv/Scripts/python TensorFlow/export_tfjs/exportar_tfjs.py
PyTorch/.venv/Scripts/python export/exportar_onnx.py

# 3) Catálogo con las métricas reales
python export/generar_catalogo.py
```

---

## 2. Arquitectura

```
src/app/
├── servicios/
│   ├── catalogo-modelos.ts    Registro de los 3 modelos + tipos. Agregar un modelo = agregar
│   │                          una entrada acá y un archivo en assets/. Nada más.
│   ├── inferencia.service.ts  Capa de abstracción. Unifica TensorFlow.js y ONNX Runtime bajo
│   │                          una sola función `predecir(canvas)`. Carga perezosa y cacheada.
│   └── opencv.service.ts      Todo OpenCV.js: nitidez (Laplaciano), detección de la pantalla
│                              y corrección de perspectiva. Cada cv.Mat se libera.
├── componentes/
│   └── selector-modelos       El catálogo, como HOJA sobre el visor (no como ruta: navegar
│                              apagaría la cámara y haría perder el encuadre).
└── paginas/
    └── camara/                Toda la app: visor a pantalla completa, HUD, y las dos hojas.
```

### La pantalla

El visor ocupa el viewport completo y todo lo demás flota encima:

- **HUD superior** — modelo activo (toca para cambiarlo) y los medidores de nitidez y FPS.
- **Retícula viva** — cuatro escuadras sobre el video. Cambian de color con la nitidez
  (rojo = movida, ámbar = aceptable, verde = nítida) y **saltan a las esquinas de la pantalla
  real** cuando OpenCV la detecta, mostrando de antemano qué recorte va a recibir el modelo.
- **HUD inferior** — pista contextual de una línea, obturador de 76 px y los dos accesorios.
- **Hojas** — resultado y selector suben desde abajo con `ion-modal` y breakpoints (arrastre,
  foco atrapado y cierre por deslizamiento nativos), sin desmontar la cámara.

Al disparar, el visor **se congela** en el frame que se analiza y lo recorre un barrido. Sin
eso, el video sigue moviéndose mientras se mide otra cosa y la interfaz miente sobre qué está
mirando el modelo. Hay pulso háptico distinto para "analizando" y para "foto rechazada".

**El punto de la arquitectura:** la página de cámara no sabe qué modelo corre. Le pide una
predicción a `InferenciaService` y recibe probabilidades. Cambiar, agregar o mejorar un modelo
no toca una línea de la aplicación — es el patrón que separa un producto de un experimento.

---

## 3. Los tres modelos

| | Modelo A | Modelo B | Modelo C |
|---|---|---|---|
| Framework | TensorFlow / Keras | PyTorch | PyTorch |
| Arquitectura | MobileNetV3-Small | MobileNetV3-Small | EfficientNet-B0 |
| Formato | TensorFlow.js (GraphModel) | ONNX | ONNX |
| Motor | WebGL (GPU) | WebAssembly | WebAssembly |
| Ejes | NHWC | NCHW | NCHW |
| Salida | softmax | logits | logits |
| Entrenamiento | balanceado, fine-tuning 2 fases | ídem A | **desequilibrado 10:3:1** + Focal Loss |

Los tres comen **píxeles 0-255 crudos**: la normalización ImageNet viaja dentro del grafo
(en Keras por `include_preprocessing=True`, en PyTorch por la capa `Normalizador`). El cliente
no normaliza nada, y por lo tanto no puede equivocarse al hacerlo.

**Por qué tres:** ver la explicación en la propia pantalla de modelos y el encabezado de
`catalogo-modelos.ts`. En resumen: A es el rápido, B verifica la paridad entre frameworks y
sirve de respaldo en WASM, y C aporta un voto de otra familia de arquitectura. El botón
**Comparar los 3** los corre sobre la misma foto y muestra si hay consenso.

---

## 4. Lo que hace OpenCV.js (y por qué)

1. **Filtro de calidad — varianza del Laplaciano.** Antes de molestar al modelo se mide la
   nitidez. Un modelo siempre responde, incluso sobre una foto movida donde no se lee nada:
   responde con confianza alta y se equivoca. Si la varianza está bajo el umbral, la app
   **rechaza la foto** y pide otra. También se controla el brillo medio, porque una foto
   quemada por el reflejo puede pasar el filtro de nitidez y no tener información.

2. **Detección de pantalla y corrección de perspectiva.** Canny + `findContours` +
   `approxPolyDP` buscan el cuadrilátero más grande; si aparece, `warpPerspective` lo estira a
   un cuadrado de 224 px. Eso devuelve la foto al dominio en el que el modelo fue entrenado
   (la diapositiva ocupando todo el cuadro, sin inclinación) en vez de pedirle que adivine.

3. **Redimensionado con `INTER_AREA`.** Al reducir de 1280 a 224 px, la interpolación bilineal
   produce aliasing sobre el texto — la señal más informativa de una diapositiva. `INTER_AREA`
   promedia el área completa, igual que el remuestreo con antialias del pipeline de
   entrenamiento.

> **Memoria WASM:** cada `cv.Mat` se libera con `.delete()` en un bloque `finally`.
> El heap de WebAssembly no lo administra el recolector de basura de JavaScript: un `Mat` de
> 1280×720 RGBA son 3.7 MB que quedan reservados para siempre. A 30 fps, olvidar un `delete()`
> mata la pestaña en segundos.

---

## 5. Rendimiento

- **El bucle de frames corre fuera de la zona de Angular** (`NgZone.runOutsideAngular`). Con
  60 fps dentro de la zona, cada frame dispararía un ciclo completo de detección de cambios.
  Se vuelve a entrar solo cada 5 frames, para publicar FPS y nitidez.
- **`ChangeDetectionStrategy.OnPush`** en la página y el selector: Angular solo re-renderiza cuando
  una señal cambia.
- **La nitidez se calcula cada 5 frames**, no en los 60: la escena no cambia entre frames
  consecutivos y el Laplaciano no es gratis.
- **Las sesiones de los modelos se cachean.** Crear la sesión cuesta cientos de milisegundos;
  ejecutarla, decenas. Recrearla en cada foto es el error de rendimiento clásico.
- **`willReadFrequently: true`** en el contexto 2D: el canvas se lee constantemente con
  `getImageData`/`cv.imread`, y sin esa bandera cada lectura fuerza una transferencia GPU→CPU.

---

## 6. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `Failed to fetch ort-wasm*.wasm` | Faltan los binarios: `npm run copiar:ort` |
| `SharedArrayBuffer is not defined` | Hilos de WASM sin cabeceras COOP/COEP. Ya se fuerza `numThreads = 1`. |
| La cámara no abre en el celular | No es contexto seguro: usar HTTPS o un túnel (ver §1). |
| `cv is not defined` | OpenCV.js todavía no cargó. Se sirve desde `assets/opencv.js` (lo baja `npm install`); si falta, la app cae al CDN. Forzar con `npm run traer:opencv`. |
| El modelo carga pero predice raro | Casi siempre es el preprocesamiento. Verificar en Netron el orden de ejes (NHWC vs NCHW) y que el modelo espere 0-255. |
| La primera predicción tarda mucho | Es la descarga + creación de sesión. Usar **Precargar** en la pantalla de modelos antes de la demo. |
