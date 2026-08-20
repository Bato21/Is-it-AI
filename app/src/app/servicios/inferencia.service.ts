/**
 * InferenciaService
 * =================
 * La capa de abstracción entre la app y los modelos. Es el archivo que hace que el producto
 * no dependa del modelo: la página de cámara pide `predecir(canvas)` y no sabe —ni le
 * importa— si atrás corre TensorFlow.js sobre WebGL o ONNX Runtime sobre WebAssembly.
 *
 * Qué resuelve, concretamente:
 *
 *   1. DOS MOTORES, UNA INTERFAZ. TensorFlow.js y ONNX Runtime Web tienen APIs distintas,
 *      esperan los tensores en órdenes de ejes distintos (NHWC vs NCHW) y devuelven cosas
 *      distintas (probabilidades vs logits). Todo eso se normaliza acá adentro y afuera se
 *      ve una sola función que devuelve siempre probabilidades.
 *
 *   2. CARGA PEREZOSA Y CACHEADA. Los modelos pesan entre 6 y 21 MB. Se descargan la primera
 *      vez que se los selecciona, no al abrir la app, y quedan cacheados en memoria: crear
 *      la sesión de ONNX o el GraphModel de tf.js es lo caro (cientos de ms), no ejecutarlo.
 *      Volver a crear la sesión en cada foto es EL error de rendimiento clásico de este
 *      despliegue.
 *
 *   3. IMPORTS DINÁMICOS. `import('@tensorflow/tfjs')` y `import('onnxruntime-web')` se
 *      hacen dentro de la función, no arriba del archivo. Así los dos motores quedan en
 *      chunks separados y el bundle inicial de la app no arrastra ~2 MB de JavaScript de un
 *      motor que el usuario quizá nunca use.
 *
 * SOBRE EL PREPROCESAMIENTO (y por qué acá casi no hay):
 *   Los tres modelos llevan su normalización ADENTRO del grafo — en TensorFlow porque
 *   MobileNetV3 viene con include_preprocessing=True, y en PyTorch porque los scripts de la
 *   v9 le agregaron una capa `Normalizador` explícitamente para esto. Consecuencia: acá solo
 *   hay que leer los píxeles 0-255 del canvas y ordenarlos en el eje que el modelo espera.
 *   Ninguna media, ninguna desviación estándar, ningún /255 escrito en TypeScript.
 *
 *   Eso no es comodidad: es eliminar de raíz la clase de bug más cara del despliegue, que es
 *   que el preprocesamiento del cliente y el del entrenamiento difieran en un decimal y el
 *   modelo "funcione pero prediga raro" sin que nada falle ni avise.
 */
import { Injectable } from '@angular/core';

import {
  CATALOGO_FALLBACK, CLASES, DefinicionModelo, ID_MODELO_POR_DEFECTO, N_ORDINALES,
} from './catalogo-modelos';

/** Resultado de una predicción. */
export interface Prediccion {
  /** Índice de la clase ganadora. */
  indice: number;
  /** Nombre de la clase ganadora (ej. '2_saturada_ia'). */
  clase: string;
  /** Probabilidad de la clase ganadora, 0-1. */
  confianza: number;
  /** Vector completo de probabilidades, en el orden de CLASES. */
  probabilidades: number[];
  /** Milisegundos que tardó la inferencia (sin contar el preprocesamiento). */
  ms: number;
  /** Id del modelo que la produjo. */
  modeloId: string;
  /**
   * true cuando la clase ganadora es la COMPUERTA de rechazo: la foto no es una diapositiva.
   *
   * Se resuelve acá y no en la vista para que la regla ("¿el índice cae fuera del eje
   * ordinal?") viva junto al lugar que conoce cuántas clases ordinales declaró el modelo.
   * Quien consume la predicción solo pregunta el booleano.
   */
  esRechazo: boolean;
}

/** Una sesión ya cargada, con su motor concreto adentro. */
interface SesionCargada {
  definicion: DefinicionModelo;
  tipo: 'tfjs' | 'onnx';
  modelo: any;
  /** Nombres de entrada/salida (solo ONNX los necesita). */
  nombreEntrada?: string;
  nombreSalida?: string;
}

@Injectable({ providedIn: 'root' })
export class InferenciaService {
  private catalogo: DefinicionModelo[] = CATALOGO_FALLBACK;
  private catalogoCargado = false;
  /** Clases y tamaño del eje ordinal. Se sobrescriben con lo que declare catalogo.json, que
   *  los copia de los resultados del entrenamiento: reentrenar con otra cantidad de clases no
   *  necesita recompilar la app. Las constantes importadas son el fallback. */
  private clases: readonly string[] = CLASES;
  private nOrdinales = N_ORDINALES;
  private sesiones = new Map<string, SesionCargada>();
  private cargasEnCurso = new Map<string, Promise<SesionCargada>>();
  private activoId = ID_MODELO_POR_DEFECTO;
  private ortConfigurado = false;

  // -------------------------------------------------------------------------------------
  // Catálogo
  // -------------------------------------------------------------------------------------

  /**
   * Lee assets/modelos/catalogo.json, que genera el script de exportación con las métricas
   * reales del último entrenamiento. Si no existe, se queda con el catálogo compilado.
   *
   * Que las métricas vengan de un JSON y no del código es lo que permite reentrenar un
   * modelo y que la app muestre sus números nuevos sin recompilar nada.
   */
  async cargarCatalogo(): Promise<DefinicionModelo[]> {
    if (this.catalogoCargado) return this.catalogo;
    try {
      const resp = await fetch('assets/modelos/catalogo.json', { cache: 'no-cache' });
      if (resp.ok) {
        const datos = await resp.json();
        if (Array.isArray(datos?.modelos) && datos.modelos.length) {
          this.catalogo = datos.modelos as DefinicionModelo[];
        }
        if (Array.isArray(datos?.clases) && datos.clases.length) {
          this.clases = datos.clases as string[];
          // Si el catálogo no declara nOrdinales, se asume que TODAS las clases son
          // ordinales: es el comportamiento de la v9 y lo correcto ante un catálogo viejo.
          // Suponer lo contrario haría que la app tratara la última clase como compuerta sin
          // que nadie lo haya dicho.
          this.nOrdinales = typeof datos.nOrdinales === 'number'
            ? datos.nOrdinales
            : this.clases.length;
        }
      }
    } catch {
      // Sin catálogo.json se sigue con el fallback: la app tiene que abrir igual.
    }
    this.catalogoCargado = true;
    if (!this.catalogo.some(m => m.id === this.activoId)) {
      this.activoId = this.catalogo[0].id;
    }
    return this.catalogo;
  }

  get modelos(): DefinicionModelo[] {
    return this.catalogo;
  }

  /** Las clases del modelo cargado, en el orden en que salen del softmax. */
  get nombresClases(): readonly string[] {
    return this.clases;
  }

  /** Cuántas clases forman el eje ordinal. Las de índice >= son compuertas de rechazo. */
  get cantidadOrdinales(): number {
    return this.nOrdinales;
  }

  get activo(): DefinicionModelo {
    return this.catalogo.find(m => m.id === this.activoId) ?? this.catalogo[0];
  }

  seleccionar(id: string): void {
    if (this.catalogo.some(m => m.id === id)) this.activoId = id;
  }

  estaCargado(id: string): boolean {
    return this.sesiones.has(id);
  }

  // -------------------------------------------------------------------------------------
  // Carga de modelos
  // -------------------------------------------------------------------------------------

  /**
   * Carga un modelo (o devuelve el ya cargado).
   *
   * El mapa `cargasEnCurso` evita la carrera obvia: si el usuario toca el botón dos veces
   * antes de que termine la descarga de 21 MB, sin esto se dispararían dos descargas y se
   * crearían dos sesiones, duplicando memoria. Con esto, la segunda llamada espera la misma
   * promesa que la primera.
   */
  async cargar(id: string): Promise<SesionCargada> {
    const cacheada = this.sesiones.get(id);
    if (cacheada) return cacheada;

    const enCurso = this.cargasEnCurso.get(id);
    if (enCurso) return enCurso;

    const definicion = this.catalogo.find(m => m.id === id);
    if (!definicion) throw new Error(`Modelo desconocido: ${id}`);

    const promesa = (definicion.backend === 'tfjs'
      ? this.cargarTfjs(definicion)
      : this.cargarOnnx(definicion))
      .then(sesion => {
        this.sesiones.set(id, sesion);
        this.cargasEnCurso.delete(id);
        return sesion;
      })
      .catch(err => {
        this.cargasEnCurso.delete(id);
        throw err;
      });

    this.cargasEnCurso.set(id, promesa);
    return promesa;
  }

  private async cargarTfjs(definicion: DefinicionModelo): Promise<SesionCargada> {
    const tf = await import('@tensorflow/tfjs');

    // WebGL usa la GPU del teléfono y es varias veces más rápido que la CPU. Si el
    // dispositivo no lo soporta (o el driver lo bloquea, cosa habitual en Android viejo),
    // se cae a CPU en vez de romper: un modelo lento es infinitamente mejor que una app que
    // no arranca.
    try {
      await tf.setBackend('webgl');
      await tf.ready();
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
    }

    const modelo = await tf.loadGraphModel(definicion.ruta);
    return { definicion, tipo: 'tfjs', modelo };
  }

  private async cargarOnnx(definicion: DefinicionModelo): Promise<SesionCargada> {
    const ort = await import('onnxruntime-web');

    if (!this.ortConfigurado) {
      // Los .wasm del runtime se sirven desde assets/ort/ (los copia `npm run copiar:ort`).
      // Sin esta ruta, onnxruntime-web los busca en un CDN y la app deja de funcionar
      // offline — que es justamente lo que no puede pasar en la demo del examen.
      ort.env.wasm.wasmPaths = 'assets/ort/';
      // Un solo hilo y sin SIMD multihilo: los hilos de WASM necesitan SharedArrayBuffer,
      // que exige cabeceras COOP/COEP que el `ng serve` de desarrollo no manda. Con 1 hilo
      // anda en todos lados; subirlo es una optimización posterior, no un requisito.
      ort.env.wasm.numThreads = 1;
      this.ortConfigurado = true;
    }

    const modelo = await ort.InferenceSession.create(definicion.ruta, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    return {
      definicion,
      tipo: 'onnx',
      modelo,
      nombreEntrada: modelo.inputNames[0],
      nombreSalida: modelo.outputNames[0],
    };
  }

  // -------------------------------------------------------------------------------------
  // Preprocesamiento
  // -------------------------------------------------------------------------------------

  /**
   * Canvas cuadrado -> Float32Array 0-255 en el orden de ejes que pida el modelo.
   *
   * Se asume que el canvas YA viene del tamaño correcto: el redimensionado y la corrección
   * de perspectiva los hace OpenCvService.prepararParaModelo(), que usa INTER_AREA (el
   * remuestreo correcto para reducir mucho). Hacerlo allá y no acá es deliberado: así los
   * tres modelos comparten exactamente el mismo preprocesamiento y las diferencias entre sus
   * predicciones son atribuibles al modelo.
   *
   * getImageData devuelve RGBA intercalado (4 bytes por píxel). El canal alfa se DESCARTA:
   * los modelos esperan 3 canales y el alfa de un canvas opaco es siempre 255.
   */
  private preprocesar(canvas: HTMLCanvasElement, orden: 'NHWC' | 'NCHW'): Float32Array {
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixeles = width * height;
    const salida = new Float32Array(pixeles * 3);

    if (orden === 'NHWC') {
      // Keras: (alto, ancho, canal). Los canales de un píxel quedan contiguos, igual que en
      // el buffer del canvas, así que es una copia salteando el alfa.
      for (let i = 0, j = 0; i < pixeles; i++) {
        salida[j++] = data[i * 4];
        salida[j++] = data[i * 4 + 1];
        salida[j++] = data[i * 4 + 2];
      }
    } else {
      // PyTorch: (canal, alto, ancho). Los tres canales quedan en planos separados, así que
      // hay que escribir en tres posiciones distantes por píxel.
      for (let i = 0; i < pixeles; i++) {
        salida[i] = data[i * 4];
        salida[pixeles + i] = data[i * 4 + 1];
        salida[pixeles * 2 + i] = data[i * 4 + 2];
      }
    }
    return salida;
  }

  /** Softmax numéricamente estable (se resta el máximo antes de exponenciar). */
  private softmax(logits: Float32Array | number[]): number[] {
    const max = Math.max(...logits);
    const exp = Array.from(logits, v => Math.exp(v - max));
    const suma = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / suma);
  }

  // -------------------------------------------------------------------------------------
  // Inferencia
  // -------------------------------------------------------------------------------------

  /**
   * Predice con el modelo indicado (o el activo). Devuelve SIEMPRE probabilidades.
   *
   * IMPORTANTE PARA QUIEN LA LLAMA: esta función es async y hace trabajo pesado. La página
   * de cámara la invoca dentro de NgZone.runOutsideAngular y con .then() en vez de await
   * dentro del frame loop, para que el video no se trabe mientras el modelo evalúa. Ver el
   * comentario correspondiente en camara.page.ts.
   */
  async predecir(canvas: HTMLCanvasElement, id?: string): Promise<Prediccion> {
    const modeloId = id ?? this.activoId;
    const sesion = await this.cargar(modeloId);
    const { definicion } = sesion;

    const datos = this.preprocesar(canvas, definicion.ordenEjes);
    const lado = definicion.tamano;
    const forma = definicion.ordenEjes === 'NHWC'
      ? [1, lado, lado, 3]
      : [1, 3, lado, lado];

    const t0 = performance.now();
    let crudo: Float32Array | number[];

    if (sesion.tipo === 'tfjs') {
      const tf = await import('@tensorflow/tfjs');
      // tf.tidy libera los tensores intermedios apenas termina el bloque. Sin él, cada
      // predicción deja tensores en la memoria de WebGL y la app degrada foto a foto —
      // el equivalente en tf.js del problema de los cv.Mat sin delete().
      const salida = tf.tidy(() => {
        const entrada = tf.tensor(datos, forma as [number, number, number, number], 'float32');
        const r = sesion.modelo.predict(entrada);
        return (Array.isArray(r) ? r[0] : r) as any;
      });
      crudo = Array.from(await salida.data() as Float32Array);
      salida.dispose();
    } else {
      const ort = await import('onnxruntime-web');
      const tensor = new ort.Tensor('float32', datos, forma);
      const resultado = await sesion.modelo.run({ [sesion.nombreEntrada!]: tensor });
      crudo = resultado[sesion.nombreSalida!].data as Float32Array;
    }

    const ms = performance.now() - t0;
    const probabilidades = definicion.salida === 'softmax'
      ? Array.from(crudo)
      : this.softmax(crudo);

    let indice = 0;
    for (let i = 1; i < probabilidades.length; i++) {
      if (probabilidades[i] > probabilidades[indice]) indice = i;
    }

    return {
      indice,
      clase: this.clases[indice] ?? `clase_${indice}`,
      confianza: probabilidades[indice],
      probabilidades,
      ms: Math.round(ms),
      modeloId,
      esRechazo: indice >= this.nOrdinales,
    };
  }

  /**
   * Corre LOS TRES modelos sobre la misma imagen y devuelve las tres predicciones.
   *
   * Esta es la función que justifica tener tres modelos y no uno. Dos usos concretos:
   *
   *   - CONFIANZA REAL. Que tres redes distintas —dos frameworks, dos arquitecturas, dos
   *     regímenes de entrenamiento— coincidan es evidencia mucho más fuerte que un solo
   *     softmax alto. Una red sobre-confiada dice 0.99 y se equivoca; tres redes que no
   *     comparten ni pesos ni datos de entrenamiento rara vez se equivocan igual.
   *   - DIAGNÓSTICO. Cuando discrepan, la imagen es genuinamente ambigua. Eso es información
   *     para el usuario ("revisá esta a mano"), no un error que haya que esconder.
   *
   * Se ejecutan en SERIE y no con Promise.all a propósito: los tres compiten por la misma
   * CPU/GPU del teléfono, así que en paralelo no terminarían antes y además el pico de
   * memoria sería la suma de los tres. En serie el pico es el del más grande.
   */
  async predecirConTodos(canvas: HTMLCanvasElement): Promise<Prediccion[]> {
    const salidas: Prediccion[] = [];
    for (const modelo of this.catalogo) {
      salidas.push(await this.predecir(canvas, modelo.id));
    }
    return salidas;
  }

  /** ¿Coinciden todas las predicciones? Devuelve el índice consensuado o null si no hay. */
  consenso(predicciones: Prediccion[]): { indice: number | null; acuerdo: number } {
    if (!predicciones.length) return { indice: null, acuerdo: 0 };
    const votos = new Map<number, number>();
    for (const p of predicciones) votos.set(p.indice, (votos.get(p.indice) ?? 0) + 1);
    let ganador = predicciones[0].indice;
    for (const [indice, n] of votos) {
      if (n > (votos.get(ganador) ?? 0)) ganador = indice;
    }
    const acuerdo = (votos.get(ganador) ?? 0) / predicciones.length;
    return { indice: acuerdo > 0.5 ? ganador : null, acuerdo };
  }

  /** Libera un modelo de memoria. Útil en teléfonos con poca RAM. */
  descargar(id: string): void {
    const sesion = this.sesiones.get(id);
    if (!sesion) return;
    try {
      if (sesion.tipo === 'tfjs') sesion.modelo.dispose?.();
      else sesion.modelo.release?.();
    } catch {
      // Si el motor no expone liberación explícita, se deja al GC.
    }
    this.sesiones.delete(id);
  }
}
