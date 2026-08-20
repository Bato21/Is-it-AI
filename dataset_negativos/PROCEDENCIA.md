# Procedencia del conjunto negativo (v10)

Generado por `documentacion/traer_negativos.py` (semilla 123).

| Carpeta | N | Fuente | Licencia | Dominio |
|---|---|---|---|---|
| `faciles/` | 650 | COCO val2017 (`http://images.cocodataset.org/zips/val2017.zip`) | Imágenes CC BY 4.0; anotaciones CC BY 4.0 | **foto de cámara real** |
| `dificiles/` | 650 | Internet Archive (colecciones `arxiv` y `magazine_rack`), renderizados a 150 dpi con `pdftoppm` | Acceso abierto arXiv (licencia por paper) | **render** |

## Por qué dos familias

`faciles/` es el caso mayoritario de la app: el usuario apunta el teléfono a algo que no es
una diapositiva. Son fotografías reales, así que su dominio ya coincide con el de
despliegue y pueden entrar al test de fotos sin simulación.

`dificiles/` es el caso que decide la calidad de la puerta: páginas de documento con texto
denso, figuras y fondo claro, que es lo único visualmente cercano a una diapositiva. Al ser
renders reciben el mismo trato que los 300 renders del proyecto — entrenamiento +
`aug_pantalla.py` — para que la puerta no pueda resolver el problema por textura.

## Reproducir

```bash
python documentacion/traer_negativos.py
```

Es idempotente y el muestreo es determinista con la semilla 123.
