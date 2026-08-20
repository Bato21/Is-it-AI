/**
 * CamaraPage — la pantalla principal. Es, en la práctica, toda la app.
 *
 * EL CAMBIO DE FONDO RESPECTO DE LA VERSIÓN ANTERIOR
 * --------------------------------------------------
 * Antes esto era una lista de tarjetas de Ionic con un canvas adentro y el resto del
 * contenido creciendo hacia abajo. Funcionaba en un escritorio y era mala en un teléfono,
 * por tres razones concretas:
 *
 *   1. El visor competía por espacio con el chrome. En una pantalla de 6", entre la barra
 *      de título, el borde de la tarjeta y el padding, el video quedaba con menos de la
 *      mitad del alto — justo lo único que el usuario necesita ver para encuadrar.
 *   2. Elegir modelo NAVEGABA a otra ruta. Eso apaga la cámara, la vuelve a pedir al
 *      volver, y en el camino se pierde el encuadre que costó conseguir.
 *   3. El resultado aparecía debajo del pliegue: había que hacer scroll para leer el
 *      veredicto, sosteniendo el teléfono apuntado a una pantalla.
 *
 * Ahora el visor ocupa el viewport completo y todo lo demás flota encima: los datos en un
 * HUD sobre las áreas seguras, y tanto el resultado como el selector de modelos en hojas
 * que suben desde abajo SIN desmontar la cámara.
 *
 * LAS TRES DECISIONES TÉCNICAS QUE SOSTIENEN LA PANTALLA (siguen valiendo)
 * -----------------------------------------------------------------------
 * 1. EL BUCLE DE FRAMES CORRE FUERA DE ANGULAR. requestAnimationFrame dispara ~60 veces
 *    por segundo; dentro de la zona, cada frame gatillaría un ciclo de detección de
 *    cambios completo. Se lanza con NgZone.runOutsideAngular() y solo se reentra cada N
 *    frames para publicar las lecturas.
 * 2. LA INFERENCIA NUNCA BLOQUEA EL VIDEO. Predecir cuesta entre 30 ms y 400 ms según el
 *    modelo. Se dispara desde el obturador, no desde el bucle, y el resultado entra por
 *    promesa.
 * 3. LA CALIDAD SE MIDE ANTES DE PREDECIR Y PUEDE RECHAZAR LA FOTO. Un modelo siempre
 *    responde, incluso sobre una foto movida donde no se lee nada: responde con confianza
 *    alta y se equivoca.
 *
 * LO QUE AGREGA ESTA VERSIÓN
 * --------------------------
 * 4. RETÍCULA VIVA. Cada N frames se corre una detección de pantalla BARATA (sobre una
 *    copia de 320 px) y las esquinas encontradas se dibujan encima del video. El usuario
 *    ve, antes de disparar, exactamente qué recorte va a recibir el modelo. Deja de ser
 *    un proceso invisible que hay que explicar y pasa a ser algo que se mira.
 * 5. CONGELADO + BARRIDO DURANTE EL ANÁLISIS. Al disparar, el bucle deja de dibujar
 *    frames nuevos: la imagen queda congelada en el frame que se está analizando, con un
 *    barrido encima. Sin eso, el video sigue moviéndose mientras se analiza otra cosa y la
 *    interfaz miente sobre qué está mirando el modelo.
 * 6. HÁPTICA. Un pulso corto al disparar, uno doble al rechazar por calidad. En un
 *    teléfono sostenido con las dos manos y la vista en la pantalla que se fotografía, la
 *    confirmación táctil llega antes que la visual.
 */
import { CommonModule } from '@angular/common';
import {
  ChangeDetectionStrategy, ChangeDetectorRef, Component, computed, ElementRef,
  NgZone, OnDestroy, OnInit, signal, ViewChild,
} from '@angular/core';
import { Subscription } from 'rxjs';

import { IonContent, IonModal } from '@ionic/angular/standalone';

import { SelectorModelosComponent } from '../../componentes/selector-modelos.component';
import {
  CLASES, DESCRIPCION_CLASES, DefinicionModelo, NOMBRE_CORTO,
} from '../../servicios/catalogo-modelos';
import { InferenciaService, Prediccion } from '../../servicios/inferencia.service';
import { Calidad, Esquinas, OpenCvService, UMBRAL_NITIDEZ } from '../../servicios/opencv.service';

type EstadoApp = 'inicial' | 'pidiendo' | 'activa' | 'error';

/** Cada cuántos frames se refrescan las lecturas del HUD (nitidez + FPS). */
const FRAMES_LECTURA = 5;

/** Cada cuántos frames se busca la pantalla. Más caro que la nitidez aunque sea sobre una
 *  copia reducida, y el encuadre cambia despacio: 15 frames (~4 veces por segundo) es más
 *  que suficiente para que la retícula se sienta viva. */
const FRAMES_DETECCION = 15;

/** Suavizado exponencial del FPS. Alto = más estable, más lento en reaccionar. */
const ALFA_FPS = 0.85;

@Component({
  selector: 'app-camara',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, IonContent, IonModal, SelectorModelosComponent],
  templateUrl: './camara.page.html',
  styleUrls: ['./camara.page.scss'],
})
export class CamaraPage implements OnInit, OnDestroy {
  @ViewChild('video') videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('lienzo') lienzoRef!: ElementRef<HTMLCanvasElement>;

  readonly CLASES = CLASES;
  readonly UMBRAL_NITIDEZ = UMBRAL_NITIDEZ;

  /**
   * Los índices del EJE ORDINAL, para la lista de niveles de la portada.
   *
   * La portada anuncia "el modelo estima cuánta huella de IA tiene, en N niveles" y los
   * lista. La compuerta de rechazo NO es un nivel —es la ausencia de veredicto— así que no va
   * en esa lista: incluirla prometería una escala de cuatro escalones que no existe.
   *
   * Es un campo y no un getter porque la plantilla lo recorre con @for: un getter devolvería
   * un array nuevo en cada ciclo de detección de cambios y forzaría a Angular a re-diferenciar
   * la lista sesenta veces por segundo mientras corre el bucle de la cámara. Se llena una vez
   * en ngOnInit, cuando ya se sabe cuántas clases declaró el catálogo.
   */
  indicesOrdinales: number[] = [0, 1, 2];

  // --- Cámara ---
  estado = signal<EstadoApp>('inicial');
  errorMsg = signal('');
  fps = signal(0);
  nitidez = signal(0);
  opencvListo = signal(false);
  /** Esquinas de la pantalla detectada, en coordenadas del canvas. */
  esquinas = signal<Esquinas | null>(null);

  // --- Modelo ---
  modeloActivo = signal<DefinicionModelo | null>(null);
  cargandoModelo = signal(false);
  selectorAbierto = signal(false);

  // --- Análisis ---
  analizando = signal(false);
  prediccion = signal<Prediccion | null>(null);
  predicciones = signal<Prediccion[]>([]);
  calidadRechazada = signal<Calidad | null>(null);
  fueEnderezada = signal(false);
  muestra = signal('');
  hojaAbierta = signal(false);
  destello = signal(false);

  // --- Derivados ---
  activa = computed(() => this.estado() === 'activa');
  pidiendo = computed(() => this.estado() === 'pidiendo');
  hayError = computed(() => this.estado() === 'error');

  /** Estado de la nitidez en tres niveles. Manda el color de la retícula y del HUD. */
  nivelNitidez = computed<'mala' | 'media' | 'buena'>(() => {
    const n = this.nitidez();
    if (n === 0) return 'media';
    if (n < UMBRAL_NITIDEZ) return 'mala';
    return n < 160 ? 'media' : 'buena';
  });

  etiquetaNitidez = computed(() => {
    const n = this.nitidez();
    if (n === 0) return '—';
    if (n < UMBRAL_NITIDEZ) return 'MOVIDA';
    if (n < 160) return 'ACEPTABLE';
    if (n < 420) return 'NITIDA';
    return 'OPTIMA';
  });

  /** true cuando la retícula está enganchada a una pantalla detectada. */
  enganchada = computed(() => this.esquinas() !== null);

  /** Se puede disparar solo si la escena está en condiciones. Deshabilitar el obturador
   *  en vez de dejar disparar y rechazar después evita una interacción frustrante. */
  puedeDisparar = computed(() =>
    this.activa() && !this.analizando() && this.nivelNitidez() !== 'mala');

  consenso = computed(() => this.inferencia.consenso(this.predicciones()));
  hayComparacion = computed(() => this.predicciones().length > 1);

  /**
   * ¿Los modelos discrepan sobre si la imagen es siquiera una diapositiva?
   *
   * No es lo mismo que una discrepancia cualquiera, y meterlas en la misma bolsa haría que la
   * app dijera algo falso. Cuando A dice "rastro de IA" y B dice "saturada", discrepan en el
   * GRADO: la diapositiva es ambigua y el mensaje "revisala a mano" es correcto.
   *
   * Cuando uno rechaza y otro informa un nivel, discrepan en si hay algo que medir. Decirle al
   * usuario "la diapositiva es ambigua" en ese caso sería afirmar que hay una diapositiva,
   * que es justamente lo que está en duda. La acción correcta tampoco es la misma: no es
   * mirarla con más cuidado, es reencuadrar.
   */
  discrepanEnTipo = computed(() => {
    const ps = this.predicciones();
    return ps.length > 1 && ps.some(p => p.esRechazo) && ps.some(p => !p.esRechazo);
  });

  /** viewBox del SVG de la retícula: sigue el tamaño interno del canvas para que las
   *  coordenadas de OpenCV se puedan dibujar tal cual, sin convertir a píxeles de CSS. */
  viewBox = signal('0 0 640 480');

  /** Los 4 puntos de la retícula. Si no hay pantalla detectada, un rectángulo centrado
   *  que sugiere el encuadre buscado. */
  puntos = computed<Esquinas>(() => {
    const detectadas = this.esquinas();
    if (detectadas) return detectadas;
    const [, , w, h] = this.viewBox().split(' ').map(Number);
    const mx = w * 0.1;
    const my = h * 0.12;
    return [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]];
  });

  polilinea = computed(() => this.puntos().map(p => p.join(',')).join(' '));

  /**
   * Las 4 escuadras de esquina, como paths SVG.
   *
   * Se dibujan como segmentos que arrancan en cada vértice y avanzan hacia sus dos
   * vecinos. Al calcularlas desde los puntos reales (y no como un rectángulo fijo), las
   * escuadras se inclinan con la perspectiva de la pantalla detectada: la retícula deja de
   * ser un adorno y pasa a mostrar la geometría que OpenCV encontró.
   */
  escuadras = computed<string[]>(() => {
    const p = this.puntos();
    const t = 0.18;   // fracción del lado que ocupa cada brazo
    const lerp = (a: number[], b: number[]) =>
      [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    return p.map((esquina, i) => {
      const previo = p[(i + 3) % 4];
      const siguiente = p[(i + 1) % 4];
      const a = lerp(esquina, previo);
      const b = lerp(esquina, siguiente);
      return `M ${a[0]} ${a[1]} L ${esquina[0]} ${esquina[1]} L ${b[0]} ${b[1]}`;
    });
  });

  private stream: MediaStream | null = null;
  private rafId: number | null = null;
  private ultimoTiempo = 0;
  private fpsSuavizado = 0;
  private contador = 0;
  /** Mientras es true, el bucle deja de volcar frames nuevos: la imagen queda congelada. */
  private congelado = false;
  private subCv?: Subscription;

  constructor(
    private opencv: OpenCvService,
    private inferencia: InferenciaService,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
  ) {}

  async ngOnInit(): Promise<void> {
    this.subCv = this.opencv.listo$.subscribe(listo => {
      this.opencvListo.set(listo);
      this.cdr.markForCheck();
    });
    await this.inferencia.cargarCatalogo();
    this.modeloActivo.set(this.inferencia.activo);
    this.indicesOrdinales = Array.from(
      { length: this.inferencia.cantidadOrdinales }, (_, i) => i);
    this.cdr.markForCheck();
  }

  ngOnDestroy(): void {
    this.liberarCamara();
    this.subCv?.unsubscribe();
  }

  // ---------------------------------------------------------------------------------------
  // Cámara
  // ---------------------------------------------------------------------------------------

  async iniciar(): Promise<void> {
    this.errorMsg.set('');
    this.estado.set('pidiendo');
    this.cdr.markForCheck();

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },   // trasera: es con la que se fotografía
          width: { ideal: 1280, min: 320 },
          height: { ideal: 720, min: 240 },
        },
      });
      await this.arrancarCaptura();
    } catch (err: any) {
      this.estado.set('error');
      this.errorMsg.set(this.mensajeError(err));
      this.cdr.markForCheck();
    }
  }

  private async arrancarCaptura(): Promise<void> {
    if (!this.stream) return;
    const video = this.videoRef.nativeElement;
    const lienzo = this.lienzoRef.nativeElement;

    video.srcObject = this.stream;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = e => reject(e);
      setTimeout(() => reject(new Error('Timeout esperando el video')), 5000);
    });
    await video.play();

    this.ajustarLienzo();

    this.ultimoTiempo = performance.now();
    this.fpsSuavizado = 0;
    this.contador = 0;
    this.congelado = false;
    this.estado.set('activa');
    this.cdr.markForCheck();

    // Precarga mientras el usuario encuadra: cuando apriete el obturador, el modelo ya
    // está en memoria y la primera predicción no se come los segundos de descarga.
    this.precargarModelo();
    window.addEventListener('resize', this.alRotar);
    window.addEventListener('orientationchange', this.alRotar);
    this.zone.runOutsideAngular(() => this.bucle());
  }

  detener(): void {
    this.liberarCamara();
    this.zone.run(() => {
      this.estado.set('inicial');
      this.fps.set(0);
      this.nitidez.set(0);
      this.esquinas.set(null);
      this.cerrarHoja();
      this.cdr.markForCheck();
    });
  }

  reintentar(): void {
    this.estado.set('inicial');
    this.errorMsg.set('');
    this.cdr.markForCheck();
  }

  /**
   * Ajusta el canvas al RECORTE del video que realmente se ve en pantalla.
   *
   * Esto arregla un problema que era a la vez visual y de correctitud. La cámara entrega
   * un cuadro apaisado (1280x720) y el teléfono es vertical (390x844). Con el canvas del
   * tamaño del video y `object-fit: cover`, el navegador recortaba más de la mitad del
   * ancho para llenar la pantalla, y eso traía dos consecuencias:
   *
   *   - La retícula se dibuja en coordenadas del canvas. Como el canvas era mucho más ancho
   *     que lo mostrado, sus esquinas caían fuera de la pantalla y no se veían.
   *   - Más grave: OpenCV medía y recortaba sobre el canvas COMPLETO, o sea que el modelo
   *     recibía franjas laterales que el usuario nunca vio. El encuadre que la persona
   *     compone no era el que se analizaba.
   *
   * La solución es que el canvas SEA el recorte: se calcula la porción centrada del video
   * con la relación de aspecto de la pantalla y se usa como tamaño interno. A partir de ahí
   * las coordenadas del canvas y las de la pantalla son la misma cosa.
   *
   * El recorte se hace a RESOLUCIÓN NATIVA (se recortan píxeles, no se reescala). No es un
   * detalle: el umbral de nitidez (UMBRAL_NITIDEZ) es empírico y la varianza del Laplaciano
   * depende de la densidad de píxeles. Reescalar cambiaría la escala de la medición y
   * dejaría el umbral calibrado para otra cosa.
   */
  private ajustarLienzo(): void {
    const video = this.videoRef?.nativeElement;
    const lienzo = this.lienzoRef?.nativeElement;
    if (!video || !lienzo || !video.videoWidth) return;

    const anchoCss = lienzo.clientWidth || window.innerWidth;
    const altoCss = lienzo.clientHeight || window.innerHeight;
    const relacion = altoCss / anchoCss;          // alto / ancho de lo que se muestra

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    let ancho: number;
    let alto: number;
    if (vw / vh > 1 / relacion) {
      alto = vh;
      ancho = Math.round(vh / relacion);          // el video sobra de ancho: se recorta
    } else {
      ancho = vw;
      alto = Math.round(vw * relacion);           // sobra de alto
    }

    lienzo.width = Math.max(2, Math.min(ancho, vw));
    lienzo.height = Math.max(2, Math.min(alto, vh));
    this.viewBox.set(`0 0 ${lienzo.width} ${lienzo.height}`);
  }

  /** Al rotar el teléfono cambia la relación de aspecto y el recorte deja de servir. */
  private alRotar = (): void => {
    // Se espera un frame: en el evento, el layout todavía tiene las medidas viejas.
    setTimeout(() => {
      this.ajustarLienzo();
      this.cdr.markForCheck();
    }, 120);
  };

  /** El bucle. Corre FUERA de la zona de Angular. Solo trabajo barato acá adentro. */
  private bucle(): void {
    if (this.estado() !== 'activa') return;

    const video = this.videoRef.nativeElement;
    const lienzo = this.lienzoRef.nativeElement;

    if (video.readyState < 2) {
      this.rafId = requestAnimationFrame(() => this.bucle());
      return;
    }

    const ahora = performance.now();
    const delta = ahora - this.ultimoTiempo;
    this.ultimoTiempo = ahora;

    if (!this.congelado) {
      // willReadFrequently mantiene el canvas en memoria del sistema en vez de en la GPU.
      // Sin la bandera, cada getImageData/cv.imread fuerza una transferencia GPU->CPU que
      // en móvil cuesta más que todo el resto del frame.
      const ctx = lienzo.getContext('2d', { willReadFrequently: true })!;
      // Recorte centrado del video al tamaño del canvas (ver ajustarLienzo). Origen y
      // destino tienen el mismo tamaño: se recortan píxeles, no se reescala.
      const sx = Math.max(0, (video.videoWidth - lienzo.width) / 2);
      const sy = Math.max(0, (video.videoHeight - lienzo.height) / 2);
      ctx.drawImage(video, sx, sy, lienzo.width, lienzo.height,
                    0, 0, lienzo.width, lienzo.height);
    }

    if (delta > 0) {
      const inst = 1000 / delta;
      this.fpsSuavizado = this.fpsSuavizado === 0
        ? inst
        : ALFA_FPS * this.fpsSuavizado + (1 - ALFA_FPS) * inst;
    }

    this.contador++;

    // Las lecturas se publican SIEMPRE, aunque OpenCV todavía no haya cargado. Antes este
    // bloque entero estaba detrás de opencvListo() y el FPS se quedaba en 0 hasta que
    // llegara el WASM: el HUD parecía roto mientras la cámara funcionaba perfecto.
    if (!this.congelado && this.contador % FRAMES_LECTURA === 0) {
      let nueva = this.nitidez();
      if (this.opencvListo()) {
        try {
          nueva = this.opencv.nitidez(lienzo);
        } catch { /* el canvas puede estar en transición de tamaño */ }
      }
      this.zone.run(() => {
        this.fps.set(Math.round(this.fpsSuavizado));
        this.nitidez.set(nueva);
        this.cdr.markForCheck();
      });
    }

    if (!this.congelado && this.opencvListo() && this.contador % FRAMES_DETECCION === 0) {
      let encontradas: Esquinas | null = null;
      try {
        encontradas = this.opencv.detectarPantallaRapido(lienzo);
      } catch { /* ídem */ }
      this.zone.run(() => {
        this.esquinas.set(encontradas);
        this.cdr.markForCheck();
      });
    }

    this.rafId = requestAnimationFrame(() => this.bucle());
  }

  private liberarCamara(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    window.removeEventListener('resize', this.alRotar);
    window.removeEventListener('orientationchange', this.alRotar);
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
  }

  // ---------------------------------------------------------------------------------------
  // Análisis
  // ---------------------------------------------------------------------------------------

  private async precargarModelo(): Promise<void> {
    const activo = this.inferencia.activo;
    if (this.inferencia.estaCargado(activo.id)) return;
    this.cargandoModelo.set(true);
    this.cdr.markForCheck();
    try {
      await this.inferencia.cargar(activo.id);
    } catch { /* se reintenta al analizar, y ahí sí se avisa */ }
    this.cargandoModelo.set(false);
    this.cdr.markForCheck();
  }

  /** Pulso háptico. `navigator.vibrate` no existe en iOS Safari: el `?.` lo cubre. */
  private vibrar(patron: number | number[]): void {
    try {
      navigator.vibrate?.(patron);
    } catch { /* algunos navegadores lo exponen pero lo bloquean sin gesto previo */ }
  }

  /** Obturador. */
  async analizar(conTodos = false): Promise<void> {
    if (!this.activa() || this.analizando()) return;

    this.vibrar(12);
    this.destello.set(true);
    setTimeout(() => {
      this.destello.set(false);
      this.cdr.markForCheck();
    }, 180);

    this.analizando.set(true);
    this.congelado = true;          // el visor queda en el frame que se analiza
    this.prediccion.set(null);
    this.predicciones.set([]);
    this.calidadRechazada.set(null);
    this.cdr.markForCheck();

    const lienzo = this.lienzoRef.nativeElement;

    try {
      // --- 1. Filtro de calidad ---
      if (this.opencvListo()) {
        const calidad = this.opencv.analizarCalidad(lienzo);
        this.nitidez.set(calidad.nitidez);
        if (!calidad.apta) {
          this.vibrar([18, 60, 18]);      // patrón distinto: se siente como un "no"
          this.calidadRechazada.set(calidad);
          this.abrirHoja();
          return;
        }
      }

      // --- 2. Preprocesamiento con OpenCV ---
      const lado = this.inferencia.activo.tamano;
      let entrada: HTMLCanvasElement;
      if (this.opencvListo()) {
        const prep = this.opencv.prepararParaModelo(lienzo, lado);
        entrada = prep.canvas;
        this.fueEnderezada.set(prep.enderezada);
      } else {
        // Sin OpenCV la app NO se rompe: cae a un redimensionado con el propio canvas.
        entrada = document.createElement('canvas');
        entrada.width = entrada.height = lado;
        const ctx = entrada.getContext('2d')!;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(lienzo, 0, 0, lado, lado);
        this.fueEnderezada.set(false);
      }
      this.muestra.set(entrada.toDataURL('image/jpeg', 0.85));

      // --- 3. Inferencia, fuera de la zona ---
      if (conTodos) {
        const todas = await this.zone.runOutsideAngular(
          () => this.inferencia.predecirConTodos(entrada));
        this.predicciones.set(todas);
        this.prediccion.set(
          todas.find(p => p.modeloId === this.inferencia.activo.id) ?? todas[0]);
      } else {
        const p = await this.zone.runOutsideAngular(() => this.inferencia.predecir(entrada));
        this.prediccion.set(p);
      }
      this.vibrar(8);
      this.abrirHoja();
    } catch (err: any) {
      this.errorMsg.set(`No se pudo analizar: ${err?.message ?? err}`);
      this.abrirHoja();
    } finally {
      this.analizando.set(false);
      this.cdr.markForCheck();
    }
  }

  // ---------------------------------------------------------------------------------------
  // Hojas
  // ---------------------------------------------------------------------------------------

  abrirHoja(): void {
    this.hojaAbierta.set(true);
    this.cdr.markForCheck();
  }

  /** Cerrar la hoja descongela el visor: se vuelve a encuadrar de inmediato. */
  cerrarHoja(): void {
    this.hojaAbierta.set(false);
    this.congelado = false;
    this.prediccion.set(null);
    this.predicciones.set([]);
    this.calidadRechazada.set(null);
    this.errorMsg.set('');
    this.cdr.markForCheck();
  }

  abrirSelector(): void {
    this.selectorAbierto.set(true);
    this.cdr.markForCheck();
  }

  cerrarSelector(): void {
    this.selectorAbierto.set(false);
    this.modeloActivo.set(this.inferencia.activo);
    this.cdr.markForCheck();
    if (this.activa()) this.precargarModelo();
  }

  // ---------------------------------------------------------------------------------------
  // Presentación
  // ---------------------------------------------------------------------------------------

  /** Nombre de la clase en el índice dado, según lo que declaró el modelo cargado. */
  private nombreClase(indice: number): string {
    return this.inferencia.nombresClases[indice] ?? CLASES[indice] ?? '';
  }

  tituloClase(indice: number): string {
    const nombre = this.nombreClase(indice);
    return DESCRIPCION_CLASES[nombre]?.titulo ?? nombre;
  }

  detalleClase(indice: number): string {
    return DESCRIPCION_CLASES[this.nombreClase(indice)]?.detalle ?? '';
  }

  /** Nombre corto para las barras del espectro, donde no entra el título completo. */
  cortoClase(indice: number): string {
    const nombre = this.nombreClase(indice);
    return NOMBRE_CORTO[nombre] ?? nombre.toUpperCase();
  }

  nombreModelo(id: string): string {
    return this.inferencia.modelos.find(m => m.id === id)?.etiqueta ?? id;
  }

  /**
   * Aviso de confianza baja.
   *
   * El umbral es 1.8 veces el azar, y el azar depende de cuántas clases haya: con 3 clases
   * era 0.33 y el umbral 0.60; con las 4 de la v10 el azar baja a 0.25 y el umbral a 0.45.
   * Dejarlo clavado en 0.60 al agregar una clase habría convertido el aviso en ruido — una
   * predicción de 0.55 sobre 4 clases es más del doble del azar y no tiene nada de dudosa.
   *
   * El factor 1.8 se sostiene porque los modelos A y B tienen un ECE cercano a 0.03, o sea
   * que su confianza declarada es fiel. OJO con el modelo C: la Focal Loss y el muestreo
   * ponderado degradan su calibración, así que para C el umbral es menos confiable.
   */
  confianzaBaja(p: Prediccion | null): boolean {
    if (!p) return false;
    const azar = 1 / Math.max(2, p.probabilidades.length);
    return p.confianza < azar * 1.8;
  }

  /** El azar para el modelo actual, en porcentaje. Se muestra junto al aviso de confianza
   *  baja: sin ese número, "confianza 40%" no le dice nada al usuario. */
  azarPct(p: Prediccion | null): number {
    return Math.round(100 / Math.max(2, p?.probabilidades.length ?? 4));
  }

  private mensajeError(err: any): string {
    switch (err?.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Permiso de cámara denegado. Habilitalo en los ajustes del navegador.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No se encontró ninguna cámara en este dispositivo.';
      case 'NotReadableError':
      case 'TrackStartError':
        return 'La cámara está siendo usada por otra aplicación.';
      case 'OverconstrainedError':
        return 'La resolución solicitada no es compatible con esta cámara.';
      default:
        return `Error de cámara: ${err?.message ?? String(err)}`;
    }
  }
}
