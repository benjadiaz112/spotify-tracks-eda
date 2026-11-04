# Informe ejecutivo · Spotify Tracks · EP3

**Benjamín Palomino y Benjamín Díaz · MLY1101 · Caso C**

## Decisión propuesta

Usar la popularidad estimada y los perfiles acústicos como ayudas para ordenar una revisión humana de catálogo. La salida está en la escala observada del CSV, de 0 a 100. Sin historial de exposición, reproducciones o fechas no se puede usar como pronóstico de éxitos ni como criterio automático de inversión.

## Evidencia para revisar

**Resultado:** HistGradientBoosting ajustado obtuvo **MAE 5,02 puntos, R² 0,830 y 91,97% dentro de ±15 puntos** en 17.913 canciones reservadas. Frente al modelo de EP2 reentrenado y evaluado en las mismas canciones (MAE 11,38), el error bajó **55,89%**.

El ajuste del boosting redujo el MAE en los cinco folds. Promediar 100 árboles también ayudó: **6,94 → 5,25 puntos** frente al primer árbol del mismo ensamble en reserva. La mezcla calibrada redujo el RMSE, pero tuvo mayor MAE que el ganador.

Al simular revisar **3.582 canciones** (20% de la reserva), el modelo encontró **3.065** con popularidad observada ≥49: **85,57%** de la lista, frente al **62,00%** de EP2 y al **25,24% esperado** con selección uniforme. Esto justifica probar la priorización, no prometer ahorro. Con artistas no vistos, el MAE sube a **7,85**, frente a **4,69** para vistos.

De 114.000 filas se retienen 89.495 canciones únicas. La EP3 evita que una huella exacta de atributos o una coincidencia de artista principal y título cruce entrenamiento y prueba: se verifican cero cruces y quedan 17.913 canciones en prueba. El mismo control encuentra 876 claves que cruzaban las particiones de EP2. La comparación entre modelos se hace en la misma partición de EP3; la EP2 estudió la misma fuente y no proporciona un test independiente. Se guardan los conteos de limpieza, grupos, folds, parámetros y predicciones.

La selección usa el error absoluto medio (MAE) en validación agrupada. Se informan, además, RMSE, R² y proporciones dentro de tolerancias. El 80% escrito en la rúbrica es un nivel de logro. Para contrastar la solicitud de un valor superior a 80% se propone la cobertura dentro de ±15 puntos, junto a tolerancias de ±5, ±10 y ±20; esto no equivale a un R² superior a 0,80 ni confirma la interpretación del docente.

## Próximo paso

Proponer un piloto de dos semanas, con revisión manual y asistida de lotes comparables. Metas propuestas: reducción mediana ≥15% en minutos por 100 canciones, aceptación ≥70% y pérdida de cobertura de géneros ≤5 puntos porcentuales. No se ha ejecutado el piloto. Antes de ampliar uso se necesita validar los derechos de los datos, medir errores con datos recientes y evaluar efectos en géneros y catálogos con poca exposición.

[Informe técnico](informe_tecnico.md) · [Protocolo](protocolo.md) · [Resultados de la EP3](../../reports/eva3/)
