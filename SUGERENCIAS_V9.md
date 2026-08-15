# Sugerencias y Hoja de Ruta para la Versión 9 (v9)

Este documento centraliza todas las recomendaciones y pasos a seguir para la construcción de la **v9** del modelo. El objetivo de esta versión no es explorar arquitecturas nuevas, sino **preparar el modelo para el despliegue comercial y asegurar que sobreviva al mundo real** (la cámara del teléfono a través de una aplicación Ionic).

---

## 1. Captura Fotográfica (Data Augmentation Natural)

El paso fundamental para cerrar la *brecha de dominio* (Domain Gap) es exponer a la red neuronal a los artefactos físicos de una fotografía de pantalla.

### Reglas para la captura del Dataset v9:
*   **Múltiples Pantallas:** Captura las mismas diapositivas mostrándolas en diferentes pantallas (ej. monitor externo mate, pantalla glossy de un notebook, y si es posible, un televisor). Diferentes paneles generan distintos patrones de interferencia (moiré).
*   **Condiciones Imperfectas:** 
    *   Toma fotos de lado o desde abajo para forzar la perspectiva trapezoidal.
    *   Deja luces prendidas para que el reflejo (glare) golpee la pantalla.
    *   No encuadres la diapositiva a la perfección; en algunas fotos incluye el marco físico del monitor para que el modelo aprenda a ignorarlo.
*   **Separación en el Directorio:** Almacena estas fotografías en un directorio nuevo (por ejemplo `dataset_fotos/`) manteniendo la misma estructura de clases (`0_sin_ia`, `1_rastro_ia`, `2_saturada_ia`).

---

## 2. Estrategia de Evaluación: Split de Validación Honesto

Si la aplicación en el mundo real consumirá exclusivamente fotos tomadas por celular, tu métrica de validación debe reflejar precisamente ese escenario.

*   **Error común a evitar:** Si mezclas renders limpios (PDFs) con las fotos y haces un split aleatorio, tu accuracy global (ej. 90%) será una ilusión impulsada por los renders limpios, mientras que el rendimiento en fotos de celular podría estar fallando estrepitosamente.
*   **Solución:** Entrena el modelo utilizando tanto renders limpios como fotos de celular, pero **asegúrate de que el conjunto de Validación / Testeo esté compuesto casi en su totalidad (o al 100%) por fotos reales**. Esto te dará la "Accuracy Comercial" real del proyecto.

---

## 3. Entrenamiento: Fine-Tuning en Dos Fases (El opcional T8)

En la v8 ya consolidaste la cabeza de clasificación óptima sobre MobileNetV3 mediante Optuna (por ejemplo, `2·128·drop0.2·adam·lr4.8e-3`). En la v9, "quema" (hardcodea) estos hiperparámetros. 

Para que la red aprenda a lidiar con el moiré y los artefactos de pantalla, debes ejecutar un **Fine-Tuning progresivo**:

1.  **Fase 1 (Backbone Congelado):** Entrena solo la cabeza (tal cual lo han hecho hasta ahora) hasta que converja.
2.  **Fase 2 (Backbone Descongelado Parcialmente):** Descongela las **últimas capas convolucionales** de MobileNetV3 (por ejemplo, los últimos 10-15 layers). 
3.  Vuelve a compilar el modelo, pero utiliza un **Learning Rate extremadamente bajo** (ej. `1e-5`). 
4.  Entrena por un par de épocas adicionales. Esto le da permiso a MobileNetV3 para modificar ligeramente los "filtros" que trae desde ImageNet y adaptarlos a las texturas específicas de un monitor.

---

## 4. Riesgo Principal: La Confusión Extrema (0 ↔ 2)

Al introducir fotos de pantallas, introducirás patrones extraños (moiré, ruido de compresión de la cámara del celular, cambios térmicos de color).
Existe el riesgo latente de que el modelo vea un patrón de moiré en una presentación humana limpia (`0_sin_ia`), asuma que ese "ruido" es producto de la Inteligencia Artificial, y clasifique la imagen como `2_saturada_ia`.

*   **Mitigación:** Monitorea la matriz de confusión con extremo cuidado. Si los errores en los extremos aumentan considerablemente, será necesario agregar aumentos de datos tradicionales en el generador (Gaussian Noise, JPEG compression) para endurecer la regularización.

---

## 5. El Despliegue: OpenCV.js y TensorFlow.js en Ionic

La meta de la v9 no es solo el modelo `.keras`, es el binario funcional dentro del navegador.

1.  **Exportación:** Corre el script `TensorFlow/export_tfjs/exportar_tfjs.py` sobre tu modelo `.keras` v9.
2.  **El Filtro de Calidad (OpenCV.js):** Sácale provecho a OpenCV.js en el frontend de Ionic. Antes de pasar el canvas al modelo, calcula la **Varianza del Laplaciano** para medir la nitidez (sharpness) de la foto tomada por el usuario. Si la varianza está por debajo de un umbral empírico, la foto está movida o desenfocada: lanza una alerta al usuario pidiéndole enfocar mejor y rechaza la imagen. Esto evita predicciones de baja confianza.
3.  **Inferencia No Bloqueante:** Al integrar el modelo en Angular (Ionic), utiliza `NgZone.runOutsideAngular(() => { ... })` o un Web Worker. Las redes neuronales consumen muchos recursos y, si lo haces en el ciclo normal, la aplicación parecerá "congelarse" ("Hanging UI") mientras se evalúa la foto.
4.  **Limpieza de Memoria WASM:** Cada matriz `cv.Mat` creada debe ser eliminada explícitamente (`mat.delete()`) después de usarla, o provocarán un memory leak masivo que cerrará el navegador móvil en 5 o 6 fotos.
