/**
 * ModelosPage — el selector de modelos.
 *
 * Por qué esta pantalla existe y no es un simple desplegable escondido en ajustes:
 *
 *   Elegir un modelo es una decisión de INGENIERÍA, no una preferencia estética. Cada uno de
 *   los tres tiene un compromiso distinto entre velocidad, peso y comportamiento frente a
 *   las clases. Para que la elección sea informada hay que mostrar los números — y los
 *   números REALES, los que salieron del test de fotos de la v9, no adjetivos.
 *
 *   Además es la demostración concreta de la arquitectura que el producto defiende: los
 *   modelos son piezas intercambiables. Cambiar el que responde es un toque acá, sin
 *   reinstalar la app ni tocar código.
 *
 * De dónde salen las métricas: de assets/modelos/catalogo.json, que genera
 * export/exportar_modelos.py leyendo los resultados_*.json de los entrenamientos. Si un
 * modelo se reentrena y mejora, esta pantalla muestra sus números nuevos sin recompilar.
 */
import { CommonModule } from '@angular/common';
import {
  ChangeDetectionStrategy, ChangeDetectorRef, Component, OnInit, signal,
} from '@angular/core';
import { Router } from '@angular/router';

import {
  IonBackButton, IonBadge, IonButton, IonButtons, IonCard, IonCardContent, IonCardHeader,
  IonCardSubtitle, IonCardTitle, IonChip, IonContent, IonHeader, IonIcon, IonLabel,
  IonNote, IonSpinner, IonTitle, IonToolbar,
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import {
  checkmarkCircle, cloudDownloadOutline, flashOutline, informationCircleOutline,
  layersOutline, scaleOutline, trashOutline,
} from 'ionicons/icons';

import { DefinicionModelo } from '../../servicios/catalogo-modelos';
import { InferenciaService } from '../../servicios/inferencia.service';

@Component({
  selector: 'app-modelos',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    IonHeader, IonToolbar, IonTitle, IonContent, IonButtons, IonBackButton, IonButton,
    IonCard, IonCardHeader, IonCardTitle, IonCardSubtitle, IonCardContent,
    IonChip, IonLabel, IonIcon, IonNote, IonBadge, IonSpinner,
  ],
  templateUrl: './modelos.page.html',
  styleUrls: ['./modelos.page.scss'],
})
export class ModelosPage implements OnInit {
  modelos = signal<DefinicionModelo[]>([]);
  activoId = signal('');
  cargando = signal<string | null>(null);
  /** Ids ya descargados y residentes en memoria. */
  cargados = signal<Set<string>>(new Set());
  hayMetricas = signal(false);

  constructor(
    private inferencia: InferenciaService,
    private cdr: ChangeDetectorRef,
    private router: Router,
  ) {
    addIcons({
      layersOutline, checkmarkCircle, cloudDownloadOutline, trashOutline,
      flashOutline, scaleOutline, informationCircleOutline,
    });
  }

  async ngOnInit(): Promise<void> {
    const modelos = await this.inferencia.cargarCatalogo();
    this.modelos.set(modelos);
    this.activoId.set(this.inferencia.activo.id);
    this.hayMetricas.set(modelos.some(m => !!m.metricas));
    this.refrescarCargados();
    this.cdr.markForCheck();
  }

  private refrescarCargados(): void {
    this.cargados.set(new Set(
      this.modelos().filter(m => this.inferencia.estaCargado(m.id)).map(m => m.id)));
  }

  esActivo(m: DefinicionModelo): boolean {
    return m.id === this.activoId();
  }

  estaCargado(m: DefinicionModelo): boolean {
    return this.cargados().has(m.id);
  }

  /** Selecciona y vuelve a la cámara: es lo que el usuario quiere el 100% de las veces. */
  seleccionar(m: DefinicionModelo): void {
    this.inferencia.seleccionar(m.id);
    this.activoId.set(m.id);
    this.cdr.markForCheck();
    this.router.navigate(['/camara']);
  }

  /** Descarga el modelo sin activarlo. Sirve para dejarlo listo antes de una demo sin red. */
  async precargar(m: DefinicionModelo): Promise<void> {
    if (this.estaCargado(m) || this.cargando()) return;
    this.cargando.set(m.id);
    this.cdr.markForCheck();
    try {
      await this.inferencia.cargar(m.id);
    } catch (err) {
      console.error('No se pudo cargar', m.id, err);
    }
    this.cargando.set(null);
    this.refrescarCargados();
    this.cdr.markForCheck();
  }

  /** Libera la memoria del modelo. En teléfonos con poca RAM tener los 3 residentes pesa. */
  liberar(m: DefinicionModelo): void {
    this.inferencia.descargar(m.id);
    this.refrescarCargados();
    this.cdr.markForCheck();
  }

  porcentaje(v: number | undefined): string {
    return v === undefined || v === null ? '—' : `${(v * 100).toFixed(1)}%`;
  }
}
