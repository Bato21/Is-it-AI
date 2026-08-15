/**
 * copiar-ort.mjs — Copia los binarios WebAssembly de onnxruntime-web a src/assets/ort/.
 *
 * POR QUÉ HACE FALTA ESTE PASO:
 *   onnxruntime-web es JavaScript que orquesta un runtime compilado a WebAssembly. Ese
 *   runtime son archivos .wasm/.mjs sueltos que NO los empaqueta el bundler: se cargan en
 *   tiempo de ejecución con un fetch. Por defecto los busca en un CDN de jsdelivr.
 *
 *   Depender del CDN tiene dos problemas concretos para este proyecto:
 *     - La app deja de funcionar sin internet. La demo del examen es exactamente el momento
 *       en que uno no quiere depender de la red de la sala.
 *     - La versión del .wasm del CDN puede no coincidir con la del paquete npm instalado, y
 *       un desajuste de versiones entre el JS y el WASM produce errores incomprensibles.
 *
 *   Copiándolos a assets/ y apuntando ort.env.wasm.wasmPaths = 'assets/ort/' (lo hace
 *   InferenciaService), el runtime se sirve desde el mismo origen que la app: offline y con
 *   la versión exacta que instaló npm.
 *
 * Se corre solo con `npm install` (hook postinstall) o a mano con `npm run copiar:ort`.
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const AQUI = dirname(fileURLToPath(import.meta.url));
const RAIZ = join(AQUI, '..');
const ORIGEN = join(RAIZ, 'node_modules', 'onnxruntime-web', 'dist');
const DESTINO = join(RAIZ, 'src', 'assets', 'ort');

if (!existsSync(ORIGEN)) {
  console.error(`[copiar-ort] No existe ${ORIGEN}. ¿Corriste npm install?`);
  process.exit(1);
}

mkdirSync(DESTINO, { recursive: true });

// Se copian los .wasm y los .mjs que los acompañan. Los .mjs son los "glue" que cargan cada
// variante del runtime (simd, threaded); sin ellos el .wasm no se puede instanciar.
const archivos = readdirSync(ORIGEN).filter(f => f.endsWith('.wasm') || f.endsWith('.mjs'));

if (archivos.length === 0) {
  console.error('[copiar-ort] No se encontró ningún .wasm en onnxruntime-web/dist.');
  process.exit(1);
}

let bytes = 0;
for (const archivo of archivos) {
  const destino = join(DESTINO, archivo);
  copyFileSync(join(ORIGEN, archivo), destino);
  bytes += statSync(destino).size;
}

console.log(`[copiar-ort] ${archivos.length} archivos -> src/assets/ort/ ` +
            `(${(bytes / 1e6).toFixed(1)} MB)`);
