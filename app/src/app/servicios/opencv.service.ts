/**
 * OpenCvService
 * =============
 * Todo lo que la app hace con OpenCV.js, en un solo lugar.
 *
 * Tres responsabilidades:
 *   1. CARGA        — envolver la inicialización asíncrona del módulo WASM.
 *   2. CALIDAD      — medir la nitidez con la varianza del Laplaciano y decidir si la foto
 *                     sirve para pedirle una predicción al modelo (SUGERENCIAS_V9.md §5.2).
 *   3. ENCUADRE     — detectar el rectángulo de la pantalla dentro de la foto y corregir la
 *                     perspectiva, para entregarle al modelo la diapositiva y no la mesa.
 *
 * LA REGLA INNEGOCIABLE DE ESTE ARCHIVO: cada cv.Mat que se crea, se borra.
 * ---------------------------------------------------------------------------
 * OpenCV.js corre sobre WebAssembly y su memoria vive en un heap propio que el recolector
 * de basura de JavaScript NO administra. Un cv.Mat que se sale de alcance en JS deja su
 * buffer reservado en el heap WASM para siempre. A resolución de cámara (1280x720 RGBA son
 * 3.7 MB por Mat) y con un frame loop a 30 fps, olvidar UN delete() agota el heap en
 * segundos y el navegador móvil mata la pestaña sin mensaje útil.
 *
 * Por eso TODA función de este archivo que crea Mats usa try/finally, y el finally libera
 * incluso si el código de adentro tiró una excepción. Es verboso a propósito: es el bug más
 * caro y más difícil de diagnosticar de todo el despliegue.
 */
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

declare var cv: any;

/** Resultado del análisis de calidad de un frame. */
export interface Calidad {
  /** Varianza del Laplaciano. Más alto = más nítido. */
  nitidez: number;
  /** Brillo medio 0-255: detecta fotos quemadas por el glare o demasiado oscuras. */
  brillo: number;
  /** true si la foto es apta para pedir una predicción. */
  apta: boolean;
  /** Motivo del rechazo, vacío si es apta. */
  motivo: string;
}

/** Esquinas detectadas de la pantalla, en orden: sup-izq, sup-der, inf-der, inf-izq. */
export type Esquinas = [number, number][];

/**
 * Umbrales de calidad.
 *
 * Son EMPÍRICOS y ese es el punto: la varianza del Laplaciano no tiene unidades absolutas,
 * depende del contenido y de la resolución. Estos valores se fijaron mirando las fotos del
 * dataset v9 a 1280x720 — el mismo tipo de foto que la app va a recibir. Si se cambia la
 * resolución de captura hay que recalibrarlos, porque la varianza escala con el detalle.
 *
 * Una diapositiva es texto sobre fondo plano: tiene MUCHÍSIMO borde cuando está enfocada y
 * casi ninguno cuando está movida. Eso hace que el Laplaciano separe muy bien acá, mejor
 * que en una escena natural.
 */
export const UMBRAL_NITIDEZ = 60;      // por debajo: movida o desenfocada
export const UMBRAL_BRILLO_MIN = 35;   // por debajo: demasiado oscura
export const UMBRAL_BRILLO_MAX = 245;  // por encima: quemada por el reflejo

@Injectable({ providedIn: 'root' })
export class OpenCvService {
  private _listo$ = new BehaviorSubject<boolean>(false);

  /** Emite true cuando el runtime WASM terminó de inicializar. */
  get listo$(): Observable<boolean> {
    return this._listo$.asObservable();
  }

  get listo(): boolean {
    return this._listo$.value;
  }

  get cv(): any {
    return typeof cv !== 'undefined' ? cv : null;
  }

  constructor() {
    // Caso recarga en caliente: el script ya terminó antes de que Angular creara el servicio,
    // así que el evento 'opencv-ready' ya se disparó y nunca va a volver. Se consulta el flag.
    if (typeof cv !== 'undefined' && (window as any)['_opencvReady']) {
      this._listo$.next(true);
      return;
    }
    window.addEventListener('opencv-ready', () => this._listo$.next(true), { once: true });
  }

  // ---------------------------------------------------------------------------------------
  // 1. Calidad de imagen
  // ---------------------------------------------------------------------------------------

  /**
   * Varianza del Laplaciano = medida de nitidez.
   *
   * El Laplaciano es la segunda derivada espacial: responde fuerte donde la intensidad
   * cambia de golpe, o sea en los bordes. Una imagen nítida tiene bordes marcados y por lo
   * tanto una distribución de respuestas ANCHA (varianza alta). Una imagen movida los tiene
   * suavizados, las respuestas se concentran cerca de cero y la varianza se desploma.
   *
   * Se mide la varianza y no la media porque la media del Laplaciano es ~0 en cualquier
   * imagen (los bordes aportan valores positivos y negativos que se cancelan). Lo que
   * distingue nítido de borroso es la DISPERSIÓN, no el promedio.
   */
  nitidez(canvas: HTMLCanvasElement): number {
    const src = cv.imread(canvas);
    const gris = new cv.Mat();
    const lap = new cv.Mat();
    const media = new cv.Mat();
    const desvio = new cv.Mat();
    try {
      cv.cvtColor(src, gris, cv.COLOR_RGBA2GRAY, 0);
      cv.Laplacian(gris, lap, cv.CV_64F, 1, 1, 0, cv.BORDER_DEFAULT);
      cv.meanStdDev(lap, media, desvio);
      return Math.round(desvio.doubleAt(0, 0) ** 2);
    } finally {
      src.delete(); gris.delete(); lap.delete(); media.delete(); desvio.delete();
    }
  }

  /**
   * Nitidez + brillo en UNA sola pasada, con el veredicto de si la foto sirve.
   *
   * Se calculan juntos a propósito: leer el canvas y convertirlo a gris es lo caro
   * (cv.imread copia el buffer completo del canvas al heap WASM), y hacerlo dos veces para
   * dos métricas duplicaría el costo del frame loop sin necesidad.
   *
   * El brillo entra porque el Laplaciano tiene un punto ciego conocido: una foto QUEMADA por
   * el reflejo de una luz puede tener bordes duros en la zona no quemada y pasar el umbral
   * de nitidez, aunque la mitad de la diapositiva sea un rectángulo blanco sin información.
   */
  analizarCalidad(canvas: HTMLCanvasElement): Calidad {
    const src = cv.imread(canvas);
    const gris = new cv.Mat();
    const lap = new cv.Mat();
    const media = new cv.Mat();
    const desvio = new cv.Mat();
    try {
      cv.cvtColor(src, gris, cv.COLOR_RGBA2GRAY, 0);
      const brillo = cv.mean(gris)[0];

      cv.Laplacian(gris, lap, cv.CV_64F, 1, 1, 0, cv.BORDER_DEFAULT);
      cv.meanStdDev(lap, media, desvio);
      const nitidez = Math.round(desvio.doubleAt(0, 0) ** 2);

      let motivo = '';
      if (nitidez < UMBRAL_NITIDEZ) {
        motivo = 'La foto está movida o desenfocada. Apoyá el teléfono y volvé a intentar.';
      } else if (brillo < UMBRAL_BRILLO_MIN) {
        motivo = 'Muy oscura. Subí el brillo de la pantalla que estás fotografiando.';
      } else if (brillo > UMBRAL_BRILLO_MAX) {
        motivo = 'Quemada por el reflejo. Cambiá el ángulo para esquivar la luz.';
      }

      return { nitidez, brillo: Math.round(brillo), apta: motivo === '', motivo };
    } finally {
      src.delete(); gris.delete(); lap.delete(); media.delete(); desvio.delete();
    }
  }

  // ---------------------------------------------------------------------------------------
  // 2. Detección de la pantalla y corrección de perspectiva
  // ---------------------------------------------------------------------------------------

  /**
   * Busca el cuadrilátero más grande de la imagen: la pantalla que muestra la diapositiva.
   *
   * Devuelve sus 4 esquinas ordenadas, o null si no encontró nada convincente.
   *
   * Cómo funciona, paso a paso y por qué cada paso:
   *   1. Escala de grises y desenfoque gaussiano. El desenfoque NO es cosmético: Canny
   *      responde al ruido igual que a los bordes reales, y una foto de pantalla trae
   *      moiré, que es ruido de alta frecuencia por todos lados. Suavizar primero es lo que
   *      hace que Canny encuentre el marco del monitor en vez de mil bordes del moiré.
   *   2. Canny: detector de bordes con histéresis (dos umbrales) — marca borde fuerte y
   *      extiende por los débiles conectados, así los bordes salen continuos.
   *   3. dilate: cierra los cortes que deja Canny en las esquinas del marco. Sin esto el
   *      contorno del monitor sale partido en cuatro segmentos y no como un polígono.
   *   4. findContours + approxPolyDP: se queda con los contornos que, aproximados, tienen
   *      exactamente 4 vértices y son convexos. Un rectángulo visto en perspectiva sigue
   *      siendo un cuadrilátero convexo; una silueta cualquiera, no.
   *   5. Filtro de área: se descartan los cuadriláteros que ocupan menos del 15% del frame.
   *      Sin ese filtro, el detector se queda con cualquier ícono rectangular de la propia
   *      diapositiva en vez de con la pantalla.
   */
  detectarPantalla(canvas: HTMLCanvasElement): Esquinas | null {
    const src = cv.imread(canvas);
    const gris = new cv.Mat();
    const suave = new cv.Mat();
    const bordes = new cv.Mat();
    const contornos = new cv.MatVector();
    const jerarquia = new cv.Mat();
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    try {
      cv.cvtColor(src, gris, cv.COLOR_RGBA2GRAY, 0);
      cv.GaussianBlur(gris, suave, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
      cv.Canny(suave, bordes, 50, 150, 3, false);
      cv.dilate(bordes, bordes, kernel, new cv.Point(-1, -1), 1,
                cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
      cv.findContours(bordes, contornos, jerarquia, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      const areaMinima = canvas.width * canvas.height * 0.15;
      let mejor: Esquinas | null = null;
      let mejorArea = areaMinima;

      for (let i = 0; i < contornos.size(); i++) {
        const c = contornos.get(i);
        const aprox = new cv.Mat();
        try {
          // epsilon = 2% del perímetro: tolerancia con la que approxPolyDP colapsa vértices
          // casi colineales. Muy chico deja 20 vértices por el ruido del borde; muy grande
          // convierte cualquier blob en triángulo.
          const perimetro = cv.arcLength(c, true);
          cv.approxPolyDP(c, aprox, 0.02 * perimetro, true);
          if (aprox.rows !== 4 || !cv.isContourConvex(aprox)) continue;

          const area = Math.abs(cv.contourArea(aprox));
          if (area <= mejorArea) continue;

          const pts: Esquinas = [];
          for (let j = 0; j < 4; j++) {
            pts.push([aprox.intAt(j, 0), aprox.intAt(j, 1)]);
          }
          mejor = this.ordenarEsquinas(pts);
          mejorArea = area;
        } finally {
          aprox.delete();
          c.delete();
        }
      }
      return mejor;
    } finally {
      src.delete(); gris.delete(); suave.delete(); bordes.delete();
      contornos.delete(); jerarquia.delete(); kernel.delete();
    }
  }

  /**
   * Ordena 4 puntos como sup-izq, sup-der, inf-der, inf-izq.
   *
   * findContours los devuelve en el orden en que recorrió el contorno, que puede arrancar en
   * cualquier vértice y girar en cualquier sentido. warpPerspective necesita la
   * correspondencia exacta punto-a-punto: si el orden no coincide con el del rectángulo
   * destino, la imagen sale rotada 90° o espejada.
   *
   * El truco clásico: la suma x+y es mínima en la esquina superior izquierda y máxima en la
   * inferior derecha; la resta y-x separa las otras dos. Funciona para cualquier rotación
   * moderada, que es el caso de una foto tomada a mano.
   */
  private ordenarEsquinas(pts: Esquinas): Esquinas {
    const suma = pts.map(([x, y]) => x + y);
    const resta = pts.map(([x, y]) => y - x);
    const supIzq = pts[suma.indexOf(Math.min(...suma))];
    const infDer = pts[suma.indexOf(Math.max(...suma))];
    const supDer = pts[resta.indexOf(Math.min(...resta))];
    const infIzq = pts[resta.indexOf(Math.max(...resta))];
    return [supIzq, supDer, infDer, infIzq];
  }

  /**
   * Corrige la perspectiva: toma las 4 esquinas y las estira a un cuadrado de `lado` px.
   *
   * Este paso es el que más acerca la entrada de producción a la de entrenamiento. El modelo
   * v9 se entrenó con la diapositiva ocupando todo el cuadro (los renders lo hacían por
   * construcción, y las fotos se aumentaron con perspectivas de ±6%). Si en producción llega
   * una foto donde la pantalla ocupa el 40% del cuadro y está inclinada 25°, el modelo ve
   * algo que nunca vio. Recortar y enderezar lo devuelve al dominio de entrenamiento.
   *
   * Se estira a CUADRADO (y no se respeta el 16:9) a propósito: es exactamente lo que hizo
   * el pipeline de entrenamiento al redimensionar a 224x224 aplastando el aspecto
   * (documentacion/imagenes.py). La deformación tiene que ser la MISMA de los dos lados.
   */
  enderezarPantalla(canvas: HTMLCanvasElement, esquinas: Esquinas, lado: number): HTMLCanvasElement {
    const src = cv.imread(canvas);
    const destino = new cv.Mat();
    const origenPts = cv.matFromArray(4, 1, cv.CV_32FC2, esquinas.flat());
    const destinoPts = cv.matFromArray(4, 1, cv.CV_32FC2,
      [0, 0, lado, 0, lado, lado, 0, lado]);
    const M = cv.getPerspectiveTransform(origenPts, destinoPts);
    try {
      cv.warpPerspective(src, destino, M, new cv.Size(lado, lado),
                         cv.INTER_LINEAR, cv.BORDER_CONSTANT,
                         new cv.Scalar(0, 0, 0, 255));
      const salida = document.createElement('canvas');
      salida.width = lado;
      salida.height = lado;
      cv.imshow(salida, destino);
      return salida;
    } finally {
      src.delete(); destino.delete(); origenPts.delete(); destinoPts.delete(); M.delete();
    }
  }

  /**
   * Redimensiona a un cuadrado con INTER_AREA. Es el camino cuando NO se detectó pantalla.
   *
   * INTER_AREA y no INTER_LINEAR: al reducir mucho (1280 -> 224 es 5.7x), la interpolación
   * bilineal muestrea unos pocos píxeles y produce aliasing sobre el texto — que es la señal
   * más informativa de una diapositiva. INTER_AREA promedia el área completa de cada píxel
   * de destino, que es el equivalente al remuestreo con antialias que usó el pipeline de
   * entrenamiento (Pillow BILINEAR aplica el filtro escalado al factor de reducción).
   * Elegir mal acá introduce una diferencia sistemática entre entrenamiento y producción.
   */
  redimensionarCuadrado(canvas: HTMLCanvasElement, lado: number): HTMLCanvasElement {
    const src = cv.imread(canvas);
    const destino = new cv.Mat();
    try {
      cv.resize(src, destino, new cv.Size(lado, lado), 0, 0, cv.INTER_AREA);
      const salida = document.createElement('canvas');
      salida.width = lado;
      salida.height = lado;
      cv.imshow(salida, destino);
      return salida;
    } finally {
      src.delete(); destino.delete();
    }
  }

  /**
   * Pipeline completo de preprocesamiento: foto cruda -> cuadrado de `lado` px listo para el
   * modelo. Devuelve también si hubo corrección de perspectiva, para poder mostrarlo en la UI.
   */
  prepararParaModelo(canvas: HTMLCanvasElement, lado: number)
    : { canvas: HTMLCanvasElement; enderezada: boolean } {
    const esquinas = this.detectarPantalla(canvas);
    if (esquinas) {
      return { canvas: this.enderezarPantalla(canvas, esquinas, lado), enderezada: true };
    }
    return { canvas: this.redimensionarCuadrado(canvas, lado), enderezada: false };
  }
}
