/**
 * copiar-ort.mjs — Copia los binarios WebAssembly de onnxruntime-web a src/assets/ort/.
 *
 * POR QUÉ HACE FALTA ESTE PASO:
 *   onnxruntime-web es JavaScript que orquesta un runtime compilado a WebAssembly. Ese
 *   runtime son archivos .wasm sueltos que NO los empaqueta el bundler: se cargan en
 *   tiempo de ejecución con un fetch. Por defecto los busca en un CDN de jsdelivr.
 *
 *   Depender del CDN tiene dos problemas concretos para este proyecto:
 *     - La app deja de funcionar sin internet. La demo del examen es exactamente el momento
 *       en que uno no quiere depender de la red de la sala.
 *     - La versión del .wasm del CDN puede no coincidir con la del paquete npm instalado, y
 *       un desajuste entre el JS y el WASM produce errores incomprensibles.
 *
 *   Copiándolos a assets/ y apuntando ort.env.wasm.wasmPaths = 'assets/ort/' (lo hace
 *   InferenciaService), el runtime se sirve desde el mismo origen que la app: offline y con
 *   la versión exacta que instaló npm.
 *
 * POR QUÉ UNA LISTA EXPLÍCITA Y NO "todos los .wasm y .mjs":
 *   Copiar el dist entero son 50 MB, y sobra casi todo:
 *
 *     - Los `ort.*.mjs` (ort.all, ort.webgpu, ort.webgl, ort.node…) son puntos de entrada
 *       para el BUNDLER. Angular resuelve el import de 'onnxruntime-web' y los empaqueta él
 *       mismo; nunca se piden por fetch desde assets/. Eran 18 MB muertos.
 *     - `ort-wasm-simd-threaded.jsep.wasm` (21.6 MB) es el build JSEP, que solo se usa con
 *       los execution providers WebGPU/WebNN. InferenciaService pide explícitamente
 *       executionProviders: ['wasm'], así que ese archivo no se descarga nunca.
 *
 *   Quedan 11.3 MB en vez de 50. En un despliegue web eso es la diferencia entre una app
 *   que pesa 86 MB y una de 47 MB.
 *
 *   Si alguna vez se habilita el provider 'webgpu', hay que volver a agregar los dos
 *   archivos .jsep a esta lista o la sesión va a fallar con un 404 del .wasm.
 *
 * Se corre solo con `npm install` (hook postinstall) o a mano con `npm run copiar:ort`.
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const AQUI = dirname(fileURLToPath(import.meta.url));
const RAIZ = join(AQUI, '..');
const ORIGEN = join(RAIZ, 'node_modules', 'onnxruntime-web', 'dist');
const DESTINO = join(RAIZ, 'src', 'assets', 'ort');

/** Lo único que onnxruntime-web pide por fetch en tiempo de ejecución con el provider 'wasm'. */
const NECESARIOS = [
  'ort-wasm-simd-threaded.wasm',   // el runtime compilado
  'ort-wasm-simd-threaded.mjs',    // el "glue" que lo instancia; sin él el .wasm no arranca
];

if (!existsSync(ORIGEN)) {
  console.error(`[copiar-ort] No existe ${ORIGEN}. ¿Corriste npm install?`);
  process.exit(1);
}

mkdirSync(DESTINO, { recursive: true });

// Limpieza de copias viejas: si esta lista se achicó, los archivos de antes seguirían en
// assets/ y se desplegarían igual, que es justo lo que se está tratando de evitar.
if (existsSync(DESTINO)) {
  for (const viejo of readdirSync(DESTINO)) {
    if (!NECESARIOS.includes(viejo)) rmSync(join(DESTINO, viejo), { force: true });
  }
}

let bytes = 0;
for (const archivo of NECESARIOS) {
  const origen = join(ORIGEN, archivo);
  if (!existsSync(origen)) {
    console.error(`[copiar-ort] FALTA ${archivo} en onnxruntime-web/dist.`);
    console.error('[copiar-ort] ¿Cambió el naming del paquete? Revisá la lista NECESARIOS.');
    process.exit(1);
  }
  const destino = join(DESTINO, archivo);
  copyFileSync(origen, destino);
  bytes += statSync(destino).size;
}

console.log(`[copiar-ort] ${NECESARIOS.length} archivos -> src/assets/ort/ ` +
            `(${(bytes / 1e6).toFixed(1)} MB)`);
