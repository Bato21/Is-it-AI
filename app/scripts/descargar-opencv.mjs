/**
 * descargar-opencv.mjs — Trae OpenCV.js a src/assets/ para que la app no dependa del CDN.
 *
 * POR QUÉ:
 *   El build de OpenCV.js pesa ~10 MB. Servido desde docs.opencv.org, en una conexión
 *   mediocre tarda decenas de segundos, y durante ese tiempo la app funciona pero SIN el
 *   filtro de nitidez ni la corrección de perspectiva — o sea, sin las dos cosas que
 *   distinguen al producto. En una prueba headless con red normal no llegó a cargar en 40
 *   segundos.
 *
 *   Y para la demo del examen es peor: es exactamente el momento en que uno no quiere
 *   depender del wifi de la sala.
 *
 *   Sirviéndolo desde el mismo origen que la app, carga en menos de un segundo y funciona
 *   sin red. El archivo NO se versiona (está en .gitignore): es una dependencia externa,
 *   igual que los .wasm de ONNX Runtime, y se baja en cada `npm install`.
 *
 * Se corre solo con `npm install` (hook postinstall) o a mano con `npm run traer:opencv`.
 */
import { createWriteStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const AQUI = dirname(fileURLToPath(import.meta.url));
const DESTINO_DIR = join(AQUI, '..', 'src', 'assets');
const DESTINO = join(DESTINO_DIR, 'opencv.js');
const URL_ORIGEN = 'https://docs.opencv.org/4.9.0/opencv.js';
const MINIMO_BYTES = 5_000_000;   // guard: un HTML de error pesa unos pocos kB

if (existsSync(DESTINO) && statSync(DESTINO).size > MINIMO_BYTES) {
  console.log(`[opencv] ya está (${(statSync(DESTINO).size / 1e6).toFixed(1)} MB), no se baja`);
  process.exit(0);
}

mkdirSync(DESTINO_DIR, { recursive: true });
console.log(`[opencv] bajando ${URL_ORIGEN} …`);

try {
  const resp = await fetch(URL_ORIGEN);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  await pipeline(Readable.fromWeb(resp.body), createWriteStream(DESTINO));

  const tam = statSync(DESTINO).size;
  if (tam < MINIMO_BYTES) {
    throw new Error(`el archivo bajado pesa ${tam} bytes: no es OpenCV.js`);
  }
  console.log(`[opencv] guardado en src/assets/opencv.js (${(tam / 1e6).toFixed(1)} MB)`);
} catch (err) {
  // No se rompe el install: index.html cae al CDN si el archivo local no está. Se avisa,
  // porque el que prepare la demo tiene que saber que va a depender de la red.
  console.warn(`[opencv] NO se pudo bajar (${err.message}).`);
  console.warn('[opencv] La app va a usar el CDN. Para la demo offline, corré:');
  console.warn('[opencv]   npm run traer:opencv');
  process.exit(0);
}
