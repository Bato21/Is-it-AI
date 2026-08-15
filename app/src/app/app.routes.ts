import { Routes } from '@angular/router';

/**
 * La app tiene UNA sola ruta.
 *
 * El selector de modelos era una ruta aparte y dejó de serlo: navegar desmonta la pantalla
 * de cámara, y eso apaga el stream, obliga a volver a pedirlo y hace perder el encuadre.
 * Ahora es una hoja (ion-modal con breakpoints) sobre el visor, que sigue vivo detrás.
 * Ver el encabezado de componentes/selector-modelos.component.ts.
 */
export const routes: Routes = [
  { path: '', redirectTo: 'camara', pathMatch: 'full' },
  {
    path: 'camara',
    loadComponent: () => import('./paginas/camara/camara.page').then(m => m.CamaraPage),
  },
  { path: '**', redirectTo: 'camara' },
];
