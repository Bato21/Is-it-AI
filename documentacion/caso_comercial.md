# Caso comercial — Auditoría de huella visual de IA en presentaciones

> Borrador para la 2ª entrega (req. 6, "uso potencialmente comercial"). Code redacta;
> Bato ajusta y lo defiende. Coherente con el reencuadre conceptual del proyecto.

## Producto

**Auditoría de huella visual de IA en presentaciones, a partir de una foto.** El usuario
saca una foto (o sube una captura) de una diapositiva y el sistema devuelve un **semáforo
0/1/2** con el **vector completo de probabilidades**:

| Nivel | Lectura |
|-------|---------|
| 🟢 `0_sin_ia`      | Sin huella visual de IA; parece hecha por humano. |
| 🟡 `1_rastro_ia`   | Huella parcial (IA con intervención humana clara). |
| 🔴 `2_saturada_ia` | Predominantemente generada por IA (plantilla/estética de generador). |

## Clientes ejemplo

- **Agencias de diseño / marketing:** verificar que un entregable cobrado como "diseño a
  medida" no sea, en realidad, un deck generado con un generador de presentaciones. Control
  de calidad interno o prueba ante el cliente final.
- **Integridad académica:** apoyo a docentes/instituciones para señalar trabajos con firma
  visual de generación automática (como insumo de conversación, no como veredicto).
- **Licitaciones y compras:** procesos que exigen material original / propio pueden usarlo
  como filtro rápido de "esto amerita revisión humana".

## Flujo de uso

```
foto de la diapo  ──►  modelo (MobileNetV3-Small)  ──►  semáforo 0/1/2 + [p0 / p1 / p2]
```

Una sola pantalla: cámara → "Analizar" → tres barras de probabilidad. Sin configuración,
sin subir la presentación entera: una foto basta. (Implementación de la app: Ionic +
TensorFlow.js + OpenCV.js; ver roadmap.)

## Qué **NO** promete (y por qué es una fortaleza)

- **No detecta la procedencia del texto ni el proceso de creación.** Una foto contiene el
  **resultado visual**, no el proceso. El producto mide **densidad de artefactos visuales
  de IA**, que es lo único aprendible desde una imagen.
- **No es un veredicto legal ni una prueba de autoría.** Es una señal de "huella visual"
  para priorizar revisión humana.
- Esta honestidad es **defendible, no una debilidad**: promete exactamente lo que puede
  cumplir. Un producto que dijera "detecto si esto lo hizo una IA" (incluido el texto)
  estaría mintiendo sobre lo que un clasificador de imágenes puede saber.

## Modelo de cobro (simple)

- **Por informe:** pago por análisis (o por lote de diapositivas) para uso ocasional.
- **Suscripción:** cuota mensual con volumen incluido para agencias / instituciones que
  auditan de forma recurrente.

## Límites honestos para la conversación de negocio

- Entrenado sobre **renders limpios**; el uso real es con **fotos** (perspectiva, reflejos,
  moiré). El *domain gap* está identificado y en medición (scripts de cámara de la entrega).
- La frontera **1↔2** es la más frágil del esquema; en un producto se comunicaría el
  **vector de probabilidades**, no solo la clase ganadora, para no sobre-prometer certeza.
- Cobertura acotada al **dominio de presentaciones**; no generaliza a cualquier imagen.

<!-- COMPLETAR: Bato — ajustar clientes/pricing a lo que quieras defender oralmente. -->
