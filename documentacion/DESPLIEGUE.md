# Despliegue de la app en Vercel

Guía para publicar `app/` (Ionic 8 + Angular 18) en Vercel, con la URL accesible desde
cualquier teléfono.

## Desplegado

**https://app-eta-one-19.vercel.app** — producción, sirviendo la **v10**.

| | |
|---|---|
| proyecto | `app` (cuenta `vjrodrig2004-3707s-projects`) |
| deployment | `dpl_7a2tpscfJ34Rd2eiYXpH37xbNKDm` · target `production` · `READY` |
| desplegado con | `cd app && vercel --prod` (CLI, sin integración de GitHub) |

Verificado sobre la URL publicada:

- `catalogo.json` declara `version: v10`, las **4 clases** y `nOrdinales: 3`.
- Los tres modelos responden 200 con su peso real: `tfjs_v10/model.json` (195 KB),
  `modelo_b_v10.onnx` (4.6 MB), `modelo_c_v10.onnx` (16.0 MB).
- `opencv.js` (10.3 MB) y `ort-wasm-simd-threaded.wasm` (11.2 MB) también responden 200, o
  sea que el `postinstall` corrió en el build de Vercel.
- Los assets de la v9 ya no están.
- Los modelos salen con `Cache-Control: public, max-age=31536000, immutable`.

> **Nota sobre el deploy por CLI.** Se publicó con `vercel --prod` desde `app/` y no con la
> integración de GitHub. La consecuencia práctica es que **un push a `main` NO redespliega**:
> hay que volver a correr el comando. Para que cada push publique solo hace falta conectar el
> repo desde el panel de Vercel (sección 1 de esta guía) con Root Directory = `app`.

> **Detalle conocido, menor.** Una ruta inexistente bajo `/assets/` devuelve `index.html` con
> **200** en vez de un 404 — la reescritura de SPA la atrapa aunque el patrón excluya
> `assets/`. No afecta a la app (el catálogo solo apunta a archivos que existen), pero si
> alguna vez falta un modelo, el error en consola va a ser un fallo de parseo raro en lugar de
> un 404 claro. Vale tenerlo presente al diagnosticar.

---

**Por qué Vercel y no la red local:** la cámara del navegador (`getUserMedia`) solo funciona
en *contexto seguro* — HTTPS o `localhost`. Servir desde `http://192.168.x.x:8100` carga la
app pero **la cámara no arranca**. Vercel da HTTPS automático y resuelve eso sin
certificados autofirmados ni reglas de firewall.

---

## 0. Lo único que hace falta antes de empezar

El repositorio es **`Bato21/Is-it-AI`**, así que la instalación de la app de GitHub la tiene
que hacer alguien con permisos de admin sobre ese repo.

1. Entrar a **https://github.com/apps/vercel**
2. **Configure** → elegir la cuenta **`Bato21`**
3. Dar acceso a **`Is-it-AI`** (o a "All repositories")

Sin este paso, Vercel responde:

```
To link a GitHub repository, you need to install the GitHub integration first.
```

> Si no se tiene admin sobre `Bato21`, la alternativa es hacer un fork del repo a la cuenta
> propia, instalar ahí la app de GitHub y conectar ese fork. El despliegue funciona igual;
> lo que se pierde es que los push a `Bato21/Is-it-AI` disparen deploys automáticos.

---

## 1. Configuración del proyecto

**Lo único que hay que configurar a mano es el Root Directory.** Todo lo demás lo declara
`app/vercel.json` y Vercel lo toma de ahí, sin tocar el panel.

| Ajuste | Valor | De dónde sale |
|---|---|---|
| **Root Directory** | **`app`** | **hay que ponerlo a mano** |
| Framework Preset | Other | `vercel.json` (`"framework": null`) |
| Build Command | `npm run build` | `vercel.json` |
| Output Directory | `www` | `vercel.json` |
| Install Command | *(dejar el default)* | — |
| Node.js Version | 22.x | `package.json` → `engines` |
| Production Branch | `main` | default |

> **No pisar el Install Command.** El `postinstall` del `package.json` es el que trae dos
> dependencias que **no están en git** (ver §4). Si se reemplaza el install por algo que no
> ejecute los hooks (por ejemplo `npm ci --ignore-scripts`), la app despliega **sin OpenCV.js
> y sin el runtime de ONNX**: se ve bien, pero sin filtro de calidad, sin corrección de
> perspectiva y con los modelos B y C rotos.

---

## 2. Camino A — desde el panel de Vercel (recomendado)

1. https://vercel.com/new
2. **Import Git Repository** → `Bato21/Is-it-AI`
3. En la pantalla de configuración:
   - **Root Directory** → *Edit* → seleccionar **`app`**
   - El resto se deja como está (lo define `vercel.json`)
4. **Deploy**

El primer build tarda ~2–3 minutos (instala 908 paquetes y baja 10 MB de OpenCV.js).

A partir de ahí, cada push a `main` despliega solo.

---

## 3. Camino B — desde la CLI

Requiere Node 22+ instalado.

```bash
npm i -g vercel

cd app          # importante: desde app/, no desde la raíz del repo
vercel login    # abre el navegador
vercel --prod
```

La CLI pregunta el scope y el nombre del proyecto. Como se corre desde `app/`, el root
directory queda implícito y lee `app/vercel.json`.

---

## 4. Qué pasa durante el build (y por qué importa)

```
npm install
  └─ postinstall
       ├─ scripts/copiar-ort.mjs      -> src/assets/ort/          (11.3 MB)
       └─ scripts/descargar-opencv.mjs -> src/assets/opencv.js     (10.3 MB)

npm run build  (ng build)
  └─ www/    47 MB · 46 archivos
```

**Estos dos archivos NO están en git, a propósito:**

- `assets/ort/*.wasm` — es una copia del paquete npm `onnxruntime-web`. Versionarlo sería
  guardar en git una dependencia, y quedaría desincronizado al subir la versión del paquete.
- `assets/opencv.js` — son 10 MB de un binario de terceros.

Los genera el `postinstall` en cada build. Ya se verificó clonando el repo limpio y corriendo
el mismo `npm install && npm run build`: los 46 archivos salen completos.

**Si `docs.opencv.org` está caído durante el build**, `descargar-opencv.mjs` no rompe el
deploy: avisa por consola y la app cae al CDN en tiempo de ejecución (`index.html` tiene el
respaldo). Peor experiencia, pero funciona.

---

## 5. Verificación después del deploy

Abrir la URL que da Vercel y comprobar, en orden:

| # | Qué | Cómo se ve si está bien |
|---|---|---|
| 1 | Portada | Fondo oscuro, título "Is it AI?" en Bricolage Grotesque, retícula cian animada al centro |
| 2 | Tipografías | Si se ve una monoespaciada genérica del sistema, no cargaron las fuentes |
| 3 | Cámara | *Encender cámara* → el navegador pide permiso → se ve el video a pantalla completa |
| 4 | OpenCV | El medidor **NITIDEZ** deja de estar en 0 y la retícula cambia de color al mover el teléfono |
| 5 | Modelo A | Disparar sobre una diapositiva → veredicto con las cuatro barras |
| 6 | **Compuerta (v10)** | Disparar sobre algo que NO sea una diapositiva (un escritorio, una pared, una web abierta) → *Sin veredicto · No es una diapositiva*, en gris. Si en cambio devuelve un nivel de IA, el modelo desplegado no es el de la v10 |
| 7 | Modelos B y C | Chip del modelo (arriba a la izquierda) → *Precargar* en B y en C → sin errores |
| 8 | Los tres | Botón **los 3** → tres filas con clase, confianza y milisegundos |

Comprobación rápida por URL (reemplazar `<URL>`):

```bash
curl -sI <URL>/assets/opencv.js                | head -1   # 200
curl -sI <URL>/assets/ort/ort-wasm-simd-threaded.wasm | head -1   # 200
curl -s  <URL>/assets/modelos/catalogo.json    | head -5   # JSON con los 3 modelos
```

Si los tres dan 200, el despliegue está completo.

---

## 6. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| Build falla: `Could not find /vercel/path0/package.json` | Root Directory sin configurar | Ponerlo en `app` |
| Deploy OK pero pantalla en blanco | Output Directory mal | Tiene que ser `www` (lo dice `vercel.json`) |
| La app carga pero **NITIDEZ** siempre 0 | `assets/opencv.js` da 404 | El install se corrió sin scripts. Revisar que el Install Command sea el default |
| Modelos B/C fallan con `Failed to fetch ort-wasm*.wasm` | Falta `assets/ort/` | Mismo motivo que arriba |
| La cámara no abre en el teléfono | Se está entrando por HTTP | Usar la URL `https://` de Vercel |
| Rutas profundas dan 404 | Falta la reescritura de SPA | Ya está en `vercel.json`; verificar que Vercel lo esté leyendo (Root Directory = `app`) |
| La primera foto tarda mucho | Descarga inicial de ~15 MB | Normal la primera vez; después queda en caché (`Cache-Control: immutable`) |

Para ver por qué falló un build: panel de Vercel → *Deployments* → el deploy → *Building*.

---

## 7. Decisión a tomar: pública o privada

Los proyectos nuevos de Vercel salen con **Vercel Authentication desactivada**, o sea que
**cualquiera con la URL entra**.

- **Pública** — conviene para mostrársela al profe sin que tenga que iniciar sesión.
  Contrapartida: los tres modelos quedan descargables por cualquiera.
- **Privada** — *Project Settings → Deployment Protection → Vercel Authentication*. Solo
  entran cuentas del equipo de Vercel.

Las fotos de los usuarios **no salen del dispositivo** en ninguno de los dos casos: la
inferencia corre entera en el navegador. Lo que queda expuesto son los modelos, no los datos.

---

## 8. Datos del despliegue

| | |
|---|---|
| Repositorio | `Bato21/Is-it-AI`, rama `main` |
| Commit preparado | `77f73da` |
| Root Directory | `app` |
| Peso del build | 47 MB · 46 archivos |
| Bundle inicial | 456 kB en disco · **128 kB transferidos** |
| Descarga de la primera foto | ~15 MB (OpenCV 10 MB + modelo A 4.1 MB + chunk de TF.js) |
| Modelo C (opcional) | +16 MB, solo si se selecciona |
| Plan | Hobby alcanza: 100 GB/mes de ancho de banda |
