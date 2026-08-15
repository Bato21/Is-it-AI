/**
 * CamaraPage — la pantalla principal del producto.
 *
 * Flujo completo, de la luz al veredicto:
 *
 *   cámara -> <video> -> canvas -> [filtro de calidad OpenCV] -> [recorte de pantalla
 *   OpenCV] -> InferenciaService -> probabilidades -> UI
 *
 * Las tres decisiones de ingeniería que sostienen esta pantalla:
 *
 * 1. EL BUCLE DE FRAMES CORRE FUERA DE ANGULAR.
 *    requestAnimationFrame dispara ~60 veces por segundo. Si ese callback viviera dentro de
 *    la zona de Angular, cada frame gatillaría un ciclo de detección de cambios completo:
 *    60 renders por segundo de una UI que cambia dos números. La app se arrastra y el video
 *    tironea. Por eso el bucle se lanza con NgZone.runOutsideAngular() y solo se vuelve a
 *    entrar (zone.run) cada N frames, para publicar FPS y nitidez. Es la recomendación
 *    explícita de SUGERENCIAS_V9.md §5.3.
 *
 * 2. LA INFERENCIA NUNCA BLOQUEA EL VIDEO.
 *    Predecir tarda entre 30 ms (MobileNet en WebGL) y 400 ms (EfficientNet en WASM). Si se
 *    hiciera con await dentro del bucle, el video se congelaría ese tiempo. Acá la
 *    predicción se dispara desde el botón —no desde el bucle— y el bucle sigue corriendo
 *    mientras tanto; el resultado entra por .then(). El usuario ve el video fluido y el
 *    resultado aparecer cuando está.
 *
 * 3. LA CALIDAD SE MIDE ANTES DE PREDECIR, Y PUEDE RECHAZAR LA FOTO.
 *    Un modelo siempre devuelve una respuesta, incluso sobre una foto movida donde no se lee
 *    nada: devuelve una respuesta con confianza alta y equivocada. El filtro de nitidez es lo
 *    que convierte "el modelo se equivocó" en "esta foto no sirve, sacá otra" — que para el
 *    usuario es un producto que funciona en vez de uno que miente.
 */
import { CommonModule } from '@angular/common';
import {
  ChangeDetectionStrategy, ChangeDetectorRef, Component, computed, ElementRef,
  NgZone, OnDestroy, OnInit, signal, ViewChild,
} from '@angular/core';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';

import {
  IonButton, IonButtons, IonCard, IonCardContent, IonCardHeader, IonCardSubtitle,
  IonCardTitle, IonChip, IonContent, IonHeader, IonIcon, IonLabel, IonNote,
  IonProgressBar, IonSpinner, IonTitle, IonToolbar,
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import {
  alertCircleOutline, camera, cameraReverseOutline, checkmarkCircle, cropOutline,
  eyeOutline, layersOutline, refreshOutline, scanOutline, speedometerOutline,
  swapHorizontalOutline, videocam, videocamOff,
} from 'ionicons/icons';

import { CLASES, DESCRIPCION_CLASES, DefinicionModelo } from '../../servicios/catalogo-modelos';
import { InferenciaService, Prediccion } from '../../servicios/inferencia.service';
import { Calidad, OpenCvService, UMBRAL_NITIDEZ } from '../../servicios/opencv.service';

type EstadoApp = 'inicial' | 'pidiendo' | 'activa' | 'error';

/** Cada cuántos frames se recalcula la nitidez y se refresca la UI.
 *  Calcular el Laplaciano en los 60 frames sería tirar cómputo: la nitidez de la escena no
 *  cambia entre frames consecutivos, y el usuario no percibe la diferencia. Es el mismo
 *  criterio de "solo el cómputo necesario" del cierre de la asignatura. */
const CADA_N_FRAMES = 5;

/** Coeficiente del suavizado exponencial del FPS. Alto = más estable, más lento en reaccionar. */
const ALFA_FPS = 0.85;

@Component({
  selector: 'app-camara',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    IonHeader, IonToolbar, IonTitle, IonContent, IonButtons, IonButton, IonIcon,
    IonCard, IonCardHeader, IonCardTitle, IonCardSubtitle, IonCardContent,
    IonChip, IonLabel, IonNote, IonSpinner, IonProgressBar,
  ],
  templateUrl: './camara.page.html',
  styleUrls: ['./camara.page.scss'],
})
export class CamaraPage implements OnInit, OnDestroy {
  @ViewChild('video') videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('lienzo') lienzoRef!: ElementRef<HTMLCanvasElement>;

  readonly CLASES = CLASES;
  readonly DESCRIPCION_CLASES = DESCRIPCION_CLASES;
  readonly UMBRAL_NITIDEZ = UMBRAL_NITIDEZ;

  // --- Estado de la cámara ---
  estado = signal<EstadoApp>('inicial');
  errorMsg = signal('');
  fps = signal(0);
  nitidez = signal(0);
  opencvListo = signal(false);

  // --- Estado del modelo ---
  modeloActivo = signal<DefinicionModelo | null>(null);
  cargandoModelo = signal(false);

  // --- Estado de la predicción ---
  analizando = signal(false);
  prediccion = signal<Prediccion | null>(null);
  predicciones = signal<Prediccion[]>([]);   // modo comparar los 3
  calidadRechazada = signal<Calidad | null>(null);
  fueEnderezada = signal(false);
  miniatura = signal<string>('');

  activa = computed(() => this.estado() === 'activa');
  pidiendo = computed(() => this.estado() === 'pidiendo');
  hayError = computed(() => this.estado() === 'error');

  /** Etiqueta cualitativa de la nitidez: el número crudo no le dice nada al usuario. */
  etiquetaNitidez = computed(() => {
    const n = this.nitidez();
    if (n === 0) return '—';
    if (n < UMBRAL_NITIDEZ) return 'Movida';
    if (n < 150) return 'Aceptable';
    if (n < 400) return 'Nítida';
    return 'Muy nítida';
  });

  colorNitidez = computed(() => {
    const n = this.nitidez();
    if (n < UMBRAL_NITIDEZ) return 'danger';
    if (n < 150) return 'warning';
    return 'success';
  });

  colorFps = computed(() => (this.fps() < 15 ? 'danger' : this.fps() < 25 ? 'warning' : 'success'));

  /** Consenso del modo comparar: cuántos de los 3 coinciden. */
  consenso = computed(() => this.inferencia.consenso(this.predicciones()));

  private stream: MediaStream | null = null;
  private rafId: number | null = null;
  private ultimoTiempo = 0;
  private fpsSuavizado = 0;
  private contadorFrames = 0;
  private subCv?: Subscription;

  constructor(
    private opencv: OpenCvService,
    private inferencia: InferenciaService,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
    private router: Router,
  ) {
    addIcons({
      camera, videocam, videocamOff, refreshOutline, alertCircleOutline, checkmarkCircle,
      speedometerOutline, eyeOutline, scanOutline, cropOutline, layersOutline,
      swapHorizontalOutline, cameraReverseOutline,
    });
  }

  async ngOnInit(): Promise<void> {
    this.subCv = this.opencv.listo$.subscribe(listo => {
      this.opencvListo.set(listo);
      this.cdr.markForCheck();
    });

    await this.inferencia.cargarCatalogo();
    this.modeloActivo.set(this.inferencia.activo);
    this.cdr.markForCheck();
  }

  ionViewWillEnter(): void {
    // Al volver del selector, el modelo activo pudo cambiar.
    this.modeloActivo.set(this.inferencia.activo);
    this.cdr.markForCheck();
  }

  ngOnDestroy(): void {
    this.liberarCamara();
    this.subCv?.unsubscribe();
  }

  // -------------------------------------------------------------------------------------
  // Cámara
  // -------------------------------------------------------------------------------------

  async iniciar(): Promise<void> {
    this.errorMsg.set('');
    this.estado.set('pidiendo');
    this.cdr.markForCheck();

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          // 'environment' = cámara trasera: es con la que se fotografía una pantalla.
          facingMode: { ideal: 'environment' },
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

    const ajustes = this.stream.getVideoTracks()[0].getSettings();
    lienzo.width = ajustes.width ?? video.videoWidth ?? 640;
    lienzo.height = ajustes.height ?? video.videoHeight ?? 480;

    this.ultimoTiempo = performance.now();
    this.fpsSuavizado = 0;
    this.contadorFrames = 0;
    this.estado.set('activa');
    this.cdr.markForCheck();

    // Precarga del modelo activo mientras el usuario encuadra: cuando apriete el botón, el
    // modelo ya está en memoria y la primera predicción no se come los 2 s de descarga.
    this.precargarModelo();

    this.zone.runOutsideAngular(() => this.bucleFrames());
  }

  detener(): void {
    this.liberarCamara();
    const lienzo = this.lienzoRef?.nativeElement;
    if (lienzo) {
      lienzo.getContext('2d')?.clearRect(0, 0, lienzo.width, lienzo.height);
    }
    this.zone.run(() => {
      this.estado.set('inicial');
      this.fps.set(0);
      this.nitidez.set(0);
      this.cdr.markForCheck();
    });
  }

  reintentar(): void {
    this.estado.set('inicial');
    this.errorMsg.set('');
    this.cdr.markForCheck();
  }

  /**
   * El bucle. Corre FUERA de la zona de Angular (ver decisión 1 del encabezado).
   *
   * Todo lo que hay acá adentro se ejecuta ~60 veces por segundo, así que solo puede haber
   * trabajo barato: dibujar el frame y contar. Lo caro (Laplaciano) va cada N frames, y
   * tocar señales de Angular —que es lo que dispara el render— también.
   */
  private bucleFrames(): void {
    if (this.estado() !== 'activa') return;

    const video = this.videoRef.nativeElement;
    const lienzo = this.lienzoRef.nativeElement;

    if (video.readyState < 2) {
      this.rafId = requestAnimationFrame(() => this.bucleFrames());
      return;
    }

    const ahora = performance.now();
    const delta = ahora - this.ultimoTiempo;
    this.ultimoTiempo = ahora;

    // willReadFrequently le avisa al navegador que este canvas se va a leer con
    // getImageData/cv.imread muy seguido, y lo mantiene en memoria del sistema en vez de en
    // la GPU. Sin esta bandera, cada lectura fuerza una transferencia GPU->CPU que en móvil
    // cuesta más que todo el resto del frame.
    const ctx = lienzo.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(video, 0, 0, lienzo.width, lienzo.height);

    if (delta > 0) {
      // Suavizado exponencial: el FPS instantáneo salta demasiado como para leerlo.
      const inst = 1000 / delta;
      this.fpsSuavizado = this.fpsSuavizado === 0
        ? inst
        : ALFA_FPS * this.fpsSuavizado + (1 - ALFA_FPS) * inst;
    }

    this.contadorFrames++;

    if (this.contadorFrames % CADA_N_FRAMES === 0) {
      let nuevaNitidez = this.nitidez();
      if (this.opencvListo()) {
        try {
          nuevaNitidez = this.opencv.nitidez(lienzo);
        } catch {
          // OpenCV puede fallar puntualmente si el canvas está en transición de tamaño.
        }
      }
      // Reentrada a la zona SOLO acá: 12 veces por segundo en vez de 60.
      this.zone.run(() => {
        this.fps.set(Math.round(this.fpsSuavizado));
        this.nitidez.set(nuevaNitidez);
        this.cdr.markForCheck();
      });
    }

    this.rafId = requestAnimationFrame(() => this.bucleFrames());
  }

  private liberarCamara(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
  }

  // -------------------------------------------------------------------------------------
  // Análisis
  // -------------------------------------------------------------------------------------

  private async precargarModelo(): Promise<void> {
    const activo = this.inferencia.activo;
    if (this.inferencia.estaCargado(activo.id)) return;
    this.cargandoModelo.set(true);
    this.cdr.markForCheck();
    try {
      await this.inferencia.cargar(activo.id);
    } catch {
      // Si falla la precarga, se reintenta al analizar y ahí sí se le avisa al usuario.
    }
    this.cargandoModelo.set(false);
    this.cdr.markForCheck();
  }

  /** Botón principal: saca la foto, la valida, la preprocesa y la clasifica. */
  async analizar(conTodos = false): Promise<void> {
    if (!this.activa() || this.analizando()) return;

    this.analizando.set(true);
    this.prediccion.set(null);
    this.predicciones.set([]);
    this.calidadRechazada.set(null);
    this.cdr.markForCheck();

    const lienzo = this.lienzoRef.nativeElement;

    try {
      // --- Paso 1: filtro de calidad (ver decisión 3 del encabezado) ---
      if (this.opencvListo()) {
        const calidad = this.opencv.analizarCalidad(lienzo);
        this.nitidez.set(calidad.nitidez);
        if (!calidad.apta) {
          this.calidadRechazada.set(calidad);
          this.analizando.set(false);
          this.cdr.markForCheck();
          return;   // se rechaza la foto: NO se molesta al modelo
        }
      }

      // --- Paso 2: preprocesamiento con OpenCV ---
      const lado = this.inferencia.activo.tamano;
      let entrada: HTMLCanvasElement;
      if (this.opencvListo()) {
        const prep = this.opencv.prepararParaModelo(lienzo, lado);
        entrada = prep.canvas;
        this.fueEnderezada.set(prep.enderezada);
      } else {
        // Sin OpenCV (todavía cargando o sin red) la app NO se rompe: cae a un
        // redimensionado con el propio canvas. Peor calidad de remuestreo, mismo contrato.
        entrada = document.createElement('canvas');
        entrada.width = entrada.height = lado;
        const ctx = entrada.getContext('2d')!;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(lienzo, 0, 0, lado, lado);
        this.fueEnderezada.set(false);
      }
      this.miniatura.set(entrada.toDataURL('image/jpeg', 0.8));

      // --- Paso 3: inferencia (fuera de la zona: ver decisión 2) ---
      if (conTodos) {
        const todas = await this.zone.runOutsideAngular(
          () => this.inferencia.predecirConTodos(entrada));
        this.predicciones.set(todas);
        this.prediccion.set(todas.find(p => p.modeloId === this.inferencia.activo.id) ?? todas[0]);
      } else {
        const p = await this.zone.runOutsideAngular(() => this.inferencia.predecir(entrada));
        this.prediccion.set(p);
      }
    } catch (err: any) {
      this.errorMsg.set(`No se pudo analizar: ${err?.message ?? err}`);
    } finally {
      this.analizando.set(false);
      this.cdr.markForCheck();
    }
  }

  irAModelos(): void {
    this.router.navigate(['/modelos']);
  }

  // -------------------------------------------------------------------------------------
  // Presentación
  // -------------------------------------------------------------------------------------

  tituloClase(indice: number): string {
    return DESCRIPCION_CLASES[CLASES[indice]]?.titulo ?? CLASES[indice];
  }

  detalleClase(indice: number): string {
    return DESCRIPCION_CLASES[CLASES[indice]]?.detalle ?? '';
  }

  colorClase(indice: number): string {
    return DESCRIPCION_CLASES[CLASES[indice]]?.color ?? 'var(--ion-color-primary)';
  }

  nombreModelo(id: string): string {
    return this.inferencia.modelos.find(m => m.id === id)?.etiqueta ?? id;
  }

  /**
   * Aviso de confianza baja.
   *
   * El umbral 0.60 no es arbitrario: los modelos de la v9 tienen un ECE cercano a 0.03, o
   * sea que su confianza declarada es razonablemente fiel. Por debajo de 0.60 sobre 3 clases
   * (el azar es 0.33) el modelo está genuinamente dudando, y decirlo es más útil que mostrar
   * un veredicto tajante.
   */
  confianzaBaja(p: Prediccion | null): boolean {
    return !!p && p.confianza < 0.6;
  }

  private mensajeError(err: any): string {
    switch (err?.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Permiso de cámara denegado. Habilitalo en Configuración → Privacidad → Cámara.';
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
