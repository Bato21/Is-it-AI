/**
 * paridad_tfjs.mjs — Verifica que el modelo convertido a TensorFlow.js dé lo MISMO que el
 * .keras original.
 *
 * POR QUÉ HACE FALTA:
 *   La conversión a TensorFlow.js pasa por tres etapas (Keras -> SavedModel -> Grappler ->
 *   GraphModel) y ninguna avisa si algo salió mal: todas producen un archivo. Encima este
 *   proyecto DESACTIVA el pase 'remap' de Grappler para esquivar la op _FusedHardSwish que
 *   tf.js no implementa — o sea que el grafo convertido no es idéntico al que TensorFlow
 *   optimizaría por defecto.
 *
 *   El modo de falla es cruel: el modelo carga bien, predice, y las probabilidades están
 *   mal. En el teléfono eso se ve como "la app clasifica cualquier cosa" y no hay ningún
 *   error que buscar. Este script convierte ese fantasma en un número.
 *
 * CÓMO EVITA CONFUNDIR EL MODELO CON LA DECODIFICACIÓN:
 *   No usa una imagen. Usa un PATRÓN SINTÉTICO calculado con la misma fórmula entera en
 *   JavaScript y en Python:
 *
 *       x[i][j][c] = (i * 7 + j * 13 + c * 29) % 256
 *
 *   Si se usara un JPEG, las diferencias entre el decodificador de Pillow y el del navegador
 *   se mezclarían con las del modelo y el número no diría nada. Con un patrón generado, la
 *   entrada es idéntica bit a bit de los dos lados y lo único que puede diferir es el modelo.
 *
 * Vive en app/scripts/ y no en export/ porque Node resuelve los imports ESM desde el
 * directorio del ARCHIVO: acá encuentra @tensorflow/tfjs en app/node_modules.
 *
 * Uso (lo invoca export/verificar_paridad.py, que además levanta el servidor):
 *     node app/scripts/paridad-tfjs.mjs http://localhost:8123/model.json
 */
import * as tf from '@tensorflow/tfjs';

const LADO = 224;

const url = process.argv[2];
if (!url) {
  console.error('Uso: node paridad_tfjs.mjs <url-del-model.json>');
  process.exit(1);
}

// Backend CPU: es el más comparable numéricamente con la referencia de Python (mismo
// float32 en CPU). WebGL introduce diferencias propias de la precisión de la GPU, que son
// aceptables en producción pero ensuciarían esta medición.
await tf.setBackend('cpu');
await tf.ready();

const modelo = await tf.loadGraphModel(url);

// El MISMO patrón que genera paridad_tfjs.py. Entrada 0-255 cruda: la normalización
// ImageNet viaja dentro del grafo.
const datos = new Float32Array(LADO * LADO * 3);
let k = 0;
for (let i = 0; i < LADO; i++) {
  for (let j = 0; j < LADO; j++) {
    for (let c = 0; c < 3; c++) {
      datos[k++] = (i * 7 + j * 13 + c * 29) % 256;
    }
  }
}

const entrada = tf.tensor(datos, [1, LADO, LADO, 3], 'float32');
const salida = modelo.predict(entrada);
const probs = await (Array.isArray(salida) ? salida[0] : salida).data();

console.log(JSON.stringify({
  backend: tf.getBackend(),
  probabilidades: Array.from(probs),
}));
