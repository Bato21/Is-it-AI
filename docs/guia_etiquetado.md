# Rúbrica de etiquetado — gradiente de IA en 3 clases

Meta: etiquetas **consistentes**. El modelo solo es tan bueno como sus etiquetas.
Ante la duda entre dos clases, elige la de **menor** nivel de IA (sé conservador
al acusar algo de ser IA).

Se etiqueta **por diapositiva** (imagen), no por presentación completa. Una misma
presentación puede tener diapositivas de las tres clases.

## `0_sin_ia` — sin huella de IA visible
- Fotos / capturas de pantalla reales, tablas y gráficos con ejes reales.
- Layout humano, idiosincrático (algo irregular, estilo personal, diagramas a mano).
- Contenido específico de dominio que una IA no fabricaría limpiamente.
- Presentaciones antiguas (pre-2023) suelen ser ejemplos seguros de `0_sin_ia`.

## `1_rastro_ia` — IA con intervención humana (rastro)
- Andamiaje claro de IA **más** ediciones humanas reales: datos reales, una
  captura, un logo de empresa, un gráfico retocado, nombres/números concretos.
- Imágenes de IA conviviendo con contenido auténtico.
- Base de plantilla pero con colores/estructura personalizados.

## `2_saturada_ia` — predominantemente generada por IA
Señales (varias, no solo una):
- Layout de plantilla IA: título centrado perfecto + 3 columnas de viñetas parejas.
- Imágenes IA (degradados súper suaves, personas falsas, texto deforme dentro de
  imágenes, dedos de más, logos derretidos).
- Tipografía y espaciado uniformes y por defecto en todo.
- Texto "voz IA": vago, en listas, sin datos específicos ni citas reales.

## Tips prácticos
- Busca **balance**: cantidades similares por clase. El desbalance sesga el modelo.
- Para este primer hito, **40-80 imágenes por clase** alcanzan para un modelo
  moderado. Calidad + balance > cantidad.
- Variedad de fuentes importa: muchas presentaciones distintas > muchas
  diapositivas de una sola.
- Revisa el balance con `python tools/dividir_dataset.py`.
