/**
 * CATÁLOGO DE MODELOS
 * ===================
 * El registro de los tres modelos que la app puede cargar, y la razón por la que hay tres.
 *
 * Este archivo es, en la práctica, la pieza de arquitectura que el cierre de la asignatura
 * describe como el valor real del proyecto: "modelos intercambiables (...) el producto no
 * cambia cuando el modelo mejora. Se reemplaza el archivo y el sistema actualiza su
 * comportamiento sin modificar una línea de código de la aplicación".
 *
 * Concretamente: para agregar un cuarto modelo se agrega una entrada a CATALOGO y se copia
 * su archivo a src/assets/modelos/. Nada más. Ni la página de cámara, ni el servicio de
 * inferencia, ni el pipeline de OpenCV se enteran — lo único que consumen es esta interfaz.
 *
 * POR QUÉ ESTOS TRES Y NO OTROS
 * -----------------------------
 * Los tres resuelven el MISMO problema (3 clases, 224x224) pero difieren en las tres cosas
 * que un ingeniero elige al desplegar: framework, arquitectura y régimen de entrenamiento.
 * Así el selector no es cosmético — cada opción cambia un compromiso real:
 *
 *   A · TensorFlow.js   MobileNetV3-Small, datos balanceados.
 *       El más liviano y el más rápido en el navegador. Es el DEFAULT del producto.
 *       Además es el único que corre sobre WebGL, o sea con GPU del teléfono.
 *
 *   B · ONNX Runtime    MobileNetV3-Small, datos balanceados, entrenado en PyTorch.
 *       Misma arquitectura que A pero otro framework y otro camino de export. Sirve para
 *       dos cosas concretas: verificar en vivo que la paridad entre frameworks se sostiene
 *       fuera del laboratorio, y actuar de respaldo si el backend WebGL falla en un
 *       dispositivo (ONNX corre sobre WASM, que anda en cualquier lado).
 *
 *   C · ONNX Runtime    EfficientNet-B0, entrenado bajo DESEQUILIBRIO 10:3:1 con Focal Loss.
 *       Arquitectura de otra familia y régimen de entrenamiento distinto. Es el "modelo
 *       desequilibrado" que pide el enunciado, y en el producto cumple una función real:
 *       cuando A y B discrepan, un tercer voto de una arquitectura no emparentada rompe el
 *       empate mejor que repetir la misma familia. Pesa ~3.5x más y corre más lento: ese es
 *       el precio, y el usuario lo ve declarado antes de elegirlo.
 *
 * Las métricas de cada entrada NO están inventadas ni redondeadas a mano: salen de los JSON
 * que generan los scripts de entrenamiento (TensorFlow/v9/resultados_v9.json,
 * PyTorch/v9/resultados_v9.json y PyTorch/v9/resultados_modelo_c_v9.json) y las copia
 * export/generar_catalogo.py al generar assets/modelos/catalogo.json. Si se reentrena un
 * modelo, sus números en la app se actualizan solos.
 */

/** Las 3 clases, EN EL ORDEN EXACTO en que las ordenaron Keras e ImageFolder al entrenar.
 *  Cambiar este orden rompe silenciosamente todas las predicciones. */
export const CLASES = ['0_sin_ia', '1_rastro_ia', '2_saturada_ia'] as const;

/** Nombres legibles y explicación de cada clase, para la UI. */
export const DESCRIPCION_CLASES: Record<string, { titulo: string; detalle: string; color: string }> = {
  '0_sin_ia': {
    titulo: 'Sin rastro de IA',
    detalle: 'No se detectan artefactos de generación automática. Diapositiva humana.',
    color: 'var(--clase-0)',
  },
  '1_rastro_ia': {
    titulo: 'Rastro de IA',
    detalle: 'Hay señales de asistencia por IA: plantillas genéricas, imágenes sintéticas puntuales.',
    color: 'var(--clase-1)',
  },
  '2_saturada_ia': {
    titulo: 'Saturada de IA',
    detalle: 'Artefactos de generación densos y sistemáticos en toda la diapositiva.',
    color: 'var(--clase-2)',
  },
};

/** Motor de inferencia que consume el modelo. */
export type Backend = 'tfjs' | 'onnx';

/** Orden de los ejes que el modelo espera. Es la diferencia Keras (NHWC) vs PyTorch (NCHW). */
export type OrdenEjes = 'NHWC' | 'NCHW';

/** Qué devuelve la última capa: si son logits, el cliente aplica softmax. */
export type TipoSalida = 'softmax' | 'logits';

export interface MetricasModelo {
  /** Accuracy sobre el test de FOTOS (la accuracy comercial de la v9). */
  accuracy: number;
  macroF1: number;
  /** Recall de 2_saturada_ia: la métrica de negocio (no dejar pasar lo que hay que detectar). */
  recallClase2: number;
  /** Errores 0<->2 acumulados: el error caro del proyecto. */
  errores02: number;
}

export interface DefinicionModelo {
  id: string;
  nombre: string;
  /** Etiqueta corta para el chip de la barra superior. */
  etiqueta: string;
  framework: string;
  arquitectura: string;
  backend: Backend;
  /** Ruta relativa dentro de assets/. */
  ruta: string;
  ordenEjes: OrdenEjes;
  salida: TipoSalida;
  /** Lado del cuadrado que espera el modelo. Los tres usan 224. */
  tamano: number;
  /** Tamaño aproximado del archivo, para avisar antes de descargar en datos móviles. */
  pesoMB: number;
  /** Una línea: qué es. */
  descripcion: string;
  /** Una línea: por qué está en el catálogo (el criterio de ingeniería). */
  porQue: string;
  metricas: MetricasModelo | null;
}

/**
 * Catálogo por defecto (fallback).
 *
 * En ejecución, ModelosService intenta primero leer assets/modelos/catalogo.json, que es el
 * archivo que genera el script de exportación con las métricas REALES del último
 * entrenamiento. Esta constante se usa solo si ese archivo no está — por ejemplo cuando
 * alguien clona el repo y levanta la app antes de entrenar nada. Tener el fallback evita
 * que la app arranque en blanco y deje al usuario sin saber qué falta.
 */
export const CATALOGO_FALLBACK: DefinicionModelo[] = [
  {
    id: 'a-tfjs',
    nombre: 'Modelo A — MobileNetV3 (TensorFlow.js)',
    etiqueta: 'A · TF.js',
    framework: 'TensorFlow / Keras',
    arquitectura: 'MobileNetV3-Small',
    backend: 'tfjs',
    ruta: 'assets/modelos/tfjs_v9/model.json',
    ordenEjes: 'NHWC',
    salida: 'softmax',
    tamano: 224,
    pesoMB: 4,
    descripcion: 'Fine-tuning en dos fases sobre fotos de pantalla. El más rápido.',
    porQue: 'Default del producto: menor latencia y único que aprovecha WebGL.',
    metricas: null,
  },
  {
    id: 'b-onnx',
    nombre: 'Modelo B — MobileNetV3 (PyTorch → ONNX)',
    etiqueta: 'B · ONNX',
    framework: 'PyTorch',
    arquitectura: 'MobileNetV3-Small',
    backend: 'onnx',
    ruta: 'assets/modelos/modelo_b_v9.onnx',
    ordenEjes: 'NCHW',
    salida: 'logits',
    tamano: 224,
    pesoMB: 5,
    descripcion: 'El espejo de A entrenado en PyTorch, con las mismas vistas y partición.',
    porQue: 'Verifica la paridad entre frameworks en producción y sirve de respaldo en WASM.',
    metricas: null,
  },
  {
    id: 'c-onnx',
    nombre: 'Modelo C — EfficientNet-B0 desequilibrado (ONNX)',
    etiqueta: 'C · Efficient',
    framework: 'PyTorch',
    arquitectura: 'EfficientNet-B0',
    backend: 'onnx',
    ruta: 'assets/modelos/modelo_c_v9.onnx',
    ordenEjes: 'NCHW',
    salida: 'logits',
    tamano: 224,
    pesoMB: 16,
    descripcion: 'Otra familia de arquitectura, entrenada con desequilibrio 10:3:1 y Focal Loss.',
    porQue: 'Tercer voto independiente: rompe empates cuando A y B discrepan.',
    metricas: null,
  },
];

/** El modelo que se activa al abrir la app por primera vez. */
export const ID_MODELO_POR_DEFECTO = 'a-tfjs';
