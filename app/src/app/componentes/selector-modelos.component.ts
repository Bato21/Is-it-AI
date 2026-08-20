/**
 * SelectorModelosComponent — el catálogo, presentado como hoja sobre el visor.
 *
 * POR QUÉ ES UN COMPONENTE Y NO UNA RUTA
 * --------------------------------------
 * Antes esto era una página aparte a la que se navegaba. En un escritorio da igual; en un
 * teléfono es un error, porque navegar DESMONTA la pantalla de cámara: se apaga el stream,
 * al volver hay que pedirlo de nuevo (en algunos navegadores, pedir el permiso otra vez) y
 * se pierde el encuadre que costó conseguir. Elegir un modelo no debería costar nada de eso.
 *
 * Como hoja, la cámara sigue viva y encuadrada detrás. El usuario cambia de modelo y vuelve
 * a disparar sin haber perdido la escena.
 *
 * POR QUÉ SE MUESTRAN LAS MÉTRICAS
 * --------------------------------
 * Elegir un modelo es una decisión de ingeniería, no una preferencia estética: cada uno
 * tiene un compromiso distinto entre velocidad, peso y comportamiento frente a las clases.
 * Para que la elección sea informada hay que mostrar los números — y los REALES, los que
 * salieron del test de fotos, no adjetivos.
 *
 * Vienen de assets/modelos/catalogo.json, que genera export/generar_catalogo.py leyendo los
 * resultados_*.json de los entrenamientos. Si un modelo se reentrena, esta pantalla muestra
 * sus números nuevos sin recompilar nada.
 */
import { CommonModule } from '@angular/common';
import {
  ChangeDetectionStrategy, ChangeDetectorRef, Component, EventEmitter, OnInit, Output, signal,
} from '@angular/core';

import { DefinicionModelo } from '../servicios/catalogo-modelos';
import { InferenciaService } from '../servicios/inferencia.service';

@Component({
  selector: 'app-selector-modelos',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule],
  templateUrl: './selector-modelos.component.html',
  styleUrls: ['./selector-modelos.component.scss'],
})
export class SelectorModelosComponent implements OnInit {
  @Output() cerrar = new EventEmitter<void>();

  modelos = signal<DefinicionModelo[]>([]);
  activoId = signal('');
  cargando = signal<string | null>(null);
  cargados = signal<Set<string>>(new Set());
  hayMetricas = signal(false);

  constructor(private inferencia: InferenciaService, private cdr: ChangeDetectorRef) {}

  async ngOnInit(): Promise<void> {
    const modelos = await this.inferencia.cargarCatalogo();
    this.modelos.set(modelos);
    this.activoId.set(this.inferencia.activo.id);
    this.hayMetricas.set(modelos.some(m => !!m.metricas));
    this.refrescar();
    this.cdr.markForCheck();
  }

  private refrescar(): void {
    this.cargados.set(new Set(
      this.modelos().filter(m => this.inferencia.estaCargado(m.id)).map(m => m.id)));
  }

  esActivo(m: DefinicionModelo): boolean {
    return m.id === this.activoId();
  }

  estaCargado(m: DefinicionModelo): boolean {
    return this.cargados().has(m.id);
  }

  /** Seleccionar cierra la hoja: es lo que el usuario quiere el 100% de las veces. */
  seleccionar(m: DefinicionModelo): void {
    this.inferencia.seleccionar(m.id);
    this.activoId.set(m.id);
    this.cdr.markForCheck();
    this.cerrar.emit();
  }

  /** Descarga sin activar. Sirve para dejar los tres listos antes de una demo sin red. */
  async precargar(m: DefinicionModelo, evento: Event): Promise<void> {
    evento.stopPropagation();   // el clic no debe además seleccionar la tarjeta
    if (this.estaCargado(m) || this.cargando()) return;
    this.cargando.set(m.id);
    this.cdr.markForCheck();
    try {
      await this.inferencia.cargar(m.id);
    } catch (err) {
      console.error('No se pudo cargar', m.id, err);
    }
    this.cargando.set(null);
    this.refrescar();
    this.cdr.markForCheck();
  }

  /** Libera memoria. En teléfonos con poca RAM, tener los tres residentes pesa. */
  liberar(m: DefinicionModelo, evento: Event): void {
    evento.stopPropagation();
    this.inferencia.descargar(m.id);
    this.refrescar();
    this.cdr.markForCheck();
  }

  pct(v: number | undefined | null): string {
    return v === undefined || v === null ? '—' : `${(v * 100).toFixed(1)}`;
  }

  /** Letra del modelo (A/B/C) para el distintivo de la tarjeta. */
  inicial(m: DefinicionModelo): string {
    return m.etiqueta.charAt(0).toUpperCase();
  }
}
