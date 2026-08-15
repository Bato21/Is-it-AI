import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'camara', pathMatch: 'full' },
  {
    path: 'camara',
    loadComponent: () => import('./paginas/camara/camara.page').then(m => m.CamaraPage),
  },
  {
    // El selector de modelos es una ruta aparte y con loadComponent (lazy) a propósito:
    // la pantalla de cámara es la que tiene que abrir rápido, y el catálogo con sus
    // métricas no hace falta hasta que el usuario lo pide.
    path: 'modelos',
    loadComponent: () => import('./paginas/modelos/modelos.page').then(m => m.ModelosPage),
  },
  { path: '**', redirectTo: 'camara' },
];
