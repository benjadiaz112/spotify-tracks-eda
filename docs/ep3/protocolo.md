# Protocolo de evaluación EP3

**Equipo: Benjamín Palomino y Benjamín Díaz. Caso C, MLY1101.** Consultar también el [diseño previo del experimento](protocolo_prerregistro.md).

## Pregunta y población

Estimar `popularity`, una puntuación numérica observada entre 0 y 100, para apoyar la revisión de catálogo de un equipo de curaduría. Cada observación es una canción del catálogo disponible, y el grupo técnico representa un conjunto conservador de registros potencialmente relacionados. La fuente no proporciona fecha de captura, historial de reproducciones ni una población futura. El experimento no mide rendimiento de una campaña ni pronostica éxitos futuros.

## Métricas y reglas

| Métrica | Definición | Uso |
|---|---|---|
| MAE | Promedio de \(\lvert y_i-\hat y_i\rvert\), en puntos de popularidad. | Criterio primario de selección en CV agrupada; menor es mejor. |
| RMSE | Raíz de la media de los errores cuadrados, en puntos. | Describir sensibilidad a errores grandes. |
| R² | `1 - SSE/SST` en el conjunto evaluado; `SST` usa la media de sus valores observados. | Diagnóstico descriptivo de regresión. Puede ser negativo y no es un porcentaje de aciertos. |
| Cobertura a ±d | Proporción de todas las predicciones con error absoluto ≤d puntos. | Comunicar precisión operativa. Se informan `d = 5, 10, 15, 20` en ese orden. |
| Mejora de MAE | `100 × (1 - MAE_modelo/MAE_referencia)` en las mismas filas. | Mostrar valor incremental respecto a la mediana global y al HGB de EP2 reentrenado en esta partición. |

La pauta escrita usa **80%** como nivel de logro **Buen desempeño** y exige métricas adecuadas al problema. La indicación verbal sobre una evaluación superior a 80% no identifica una métrica. Para dar una lectura operativa sin cambiar de tarea, se propone **más de 80% de canciones dentro de ±15 puntos**, siempre con las otras tolerancias a la vista. Quince puntos es una tolerancia exploratoria amplia de la escala; requeriría validación con curadores antes de un uso real. Se aplica de forma inclusiva al error (`≤15`), con umbral de logro estrictamente `>80%`. Esta propuesta no convierte la regresión en clasificación, no sustituye el MAE ni certifica cumplimiento de una exigencia verbal cuyo significado no se ha confirmado. Si el profesor se refiere a `R² > 0,80`, se contrasta ese valor explícitamente con el R² medido.

## Controles experimentales

1. Preservar la etiqueta de popularidad observada por continuidad. Auditar registros duplicados, rangos y desacuerdos sin cambiar las etiquetas para favorecer una métrica.
2. Separar los grupos completos antes de ajustar cualquier modelo. Unir coincidencias por huella exacta de atributos y por clave normalizada de artista principal y título. Medir el cruce de IDs y grupos tanto en test como en los folds. La clave es conservadora, no una certificación de identidad de una grabación.
3. Evaluar a los candidatos en los mismos folds agrupados. Ajustar transformaciones y parámetros solo con entrenamiento. Comparar variantes predeterminadas y ajustadas, y cada ensamble con un componente o referencia pertinente usando un contraste pareado.
4. Congelar el ganador por el criterio de CV antes de evaluar la partición final. Un contraste desfavorable en test se informa como evidencia; no dispara una nueva selección. Las predicciones se acotan a [0, 100], dominio del objetivo.
5. Guardar los folds, las configuraciones probadas y las predicciones por fila. Informar la incertidumbre condicionada a la partición, los errores por subgrupo y la cobertura real. Declarar las limitaciones de evaluación en una instantánea.

El test de EP3 se reserva dentro de esta ejecución. Como la EP2 estudió la misma fuente y publicó su propio test, la comparación entre etapas es histórica; se informa el solapamiento en vez de presentar la nueva partición como un conjunto externo o temporal.

## Reglas propuestas para el negocio

Una puntuación aproximada puede sugerir el orden de revisión, pero no decidir publicación o inversión. Un piloto exploratorio de dos semanas compara la revisión habitual con la revisión asistida usando lotes comparables, orden contrabalanceado y registro de descartes. Metas propuestas: reducir al menos 15% el tiempo mediano por cada 100 canciones, aceptar al menos 70% de recomendaciones revisadas y perder como máximo 5 puntos porcentuales de cobertura de géneros. Deben verificarse conjuntamente antes de ampliar uso. No hay mediciones de este piloto ni un cálculo de ROI.

## Referencias

- Duoc UC, *Evaluación Parcial N.º 3: Instrucciones y pauta de evaluación*, MLY1101, 2026. Páginas 1–5 según numeración impresa.
- [GroupKFold, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GroupKFold.html): evaluación sin grupos compartidos entre entrenamiento y validación de un fold.
- [GridSearchCV, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GridSearchCV.html): búsqueda controlada con validación cruzada.
- [Métricas de evaluación, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/model_evaluation.html): interpretación de métricas de regresión.
- [Ensambles, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/ensemble.html): fundamentos de boosting y promedio de estimadores.
- [Errores comunes, scikit-learn 1.8](https://scikit-learn.org/1.8/common_pitfalls.html#data-leakage): separar el ajuste de transformaciones de la evaluación.
