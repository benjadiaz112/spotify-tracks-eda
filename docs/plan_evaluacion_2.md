# Plan y criterios de aceptación - Evaluación 2

Caso C: continuación de Spotify Tracks, con regresión de `popularity` (0 a 100) y segmentación musical. La pauta entregada es la autoridad para los requisitos. Los ejemplos bancarios de las clases se adaptan al contexto musical, sin cambiar el dataset.

| Fase CRISP-DM | Tarea y decisión | Evidencia / aceptación |
|---|---|---|
| Negocio | Apoyar la revisión humana de catálogo y explorar perfiles musicales | Problema, alcance, KPI y límites explícitos en informe |
| Datos | Verificar CSV original, esquema, nulos, duplicados, rangos y sesgos | Hash real y auditoría generada desde datos |
| Preparación | Reutilizar limpieza de experiencia 1, separar antes de ajustar transformaciones | Validaciones automáticas y manifiesto train/test sin grupos de audio compartidos |
| Modelado supervisado | Referencias constantes, regresión lineal, árbol y ensamble de contraste | Selección por MAE en CV de entrenamiento, parámetros justificados |
| Modelado no supervisado | Escalado, K-Means K=1..10, codo y silueta, PCA | Perfiles en unidades originales, estabilidad y acciones de curaduría |
| Evaluación | MAE, MSE, RMSE, R², residuos y subgrupos | Test reservado, comparación con baseline, conclusión honesta |
| Despliegue | Inferencia local y monitoreo propuesto | Comando verificable, límites y mantenimiento documentados |

## Metas previas al entrenamiento

- MAE <= 15 puntos y reducción >= 10% frente al baseline de media. Se mantienen las metas de la experiencia 1, aunque no se alcancen.
- Reportar además baseline de mediana, óptimo entre constantes para MAE.
- Trazabilidad: hash del CSV, semilla 42, dependencias exactas, parámetros y particiones persistidos.
- Sin duplicados por `track_id` y sin audio idéntico entre train y test.
- K-Means: comparar 1..10, reportar silueta y advertir si los grupos se superponen. No imponer un umbral retrospectivo para simular éxito.
- Notebook integral ejecutado sin errores y pruebas de preparación, partición, serialización e inferencia.
- Defensa de 10 minutos con preguntas cruzadas para todos.

## Alcance y riesgos

Se estima popularidad observada en una instantánea. No existe fecha de lanzamiento ni historial para predecir popularidad futura o demostrar retorno financiero. No se entrenará con nombres, IDs, artistas o álbumes. El género conservado por la limpieza original es solo una etiqueta de varias posibles.

La retención >=95% de la experiencia 1 confundía duplicación con calidad. Se conservará como referencia histórica y se evaluará la retención sobre canciones únicas, mostrando también las filas brutas eliminadas. No se modificarán umbrales para mejorar métricas.

El informe final debe separar hechos medidos, hipótesis comerciales y recomendaciones. La entrega cubre la rúbrica, pero la calificación depende del docente y de la defensa individual.
