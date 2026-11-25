# Informe ejecutivo · Spotify Tracks

**Evaluación Final Transversal · MLY1101 · Caso C**

**Benjamín Palomino y Benjamín Díaz**

## Decisión

Proponemos un piloto con el HistGradientBoosting ajustado para ordenar la revisión de canciones. Curaduría conserva la decisión final. El problema es dónde concentrar un cupo de revisión limitado cuando hay más canciones que tiempo para examinarlas. El modelo estima la popularidad observada de 0 a 100 y el análisis acústico ayuda a explorar perfiles de catálogo.

## Evidencia

Se retuvieron 89.495 canciones únicas de 114.000 filas. Las grabaciones relacionadas permanecen juntas y se comprobaron cero cruces por ID, familia, huella de atributos y clave de grabación entre desarrollo y reserva. Se auditó el linaje de las filas, se ampliaron las distribuciones de audio del notebook y se compararon modelos con cinco folds agrupados.

El HGB ajustado tuvo el menor MAE medio en validación (5,129). En **17.913 canciones reservadas** obtuvo **5,02 puntos de MAE**, **8,50 de RMSE**, **R² de 0,830** y **91,97%** de predicciones dentro de ±15 puntos. Frente al HGB de EP2 reentrenado sobre la misma partición, el MAE bajó **55,89%**. En una comparación independiente del ajuste del HGB, ExtraTrees redujo el MAE un **24,35%** frente a su primer árbol. La mezcla calibrada obtuvo mejor RMSE en reserva y el HGB ajustado mantuvo la ventaja en el error absoluto promedio usado para seleccionar.

Con **3.582 canciones** por revisar, el HGB ajustado identificó **3.065** de popularidad alta. Eso equivale al **85,57%** de la lista, frente a **62,00%** de EP2 y **25,24%** esperado con selección uniforme. Se llamó alta a una popularidad de al menos 49, el percentil 75 de desarrollo. K-Means aporta dos perfiles acústicos para explorar listas tranquilas y activas, con separación modesta entre perfiles. [Comparación de modelos](../../reports/eva3/supervised_metrics.csv) · [Simulación de revisión](../../reports/eva3/business_candidates.csv).

## Condiciones para el piloto

El error sube de **4,69 a 7,85 puntos** cuando el artista carece de historial en desarrollo. Este grupo representa **7,71%** de la lista al cupo de 20% y **10,42%** de la reserva. Ampliar el cupo al 30% recupera más canciones de popularidad alta (83,59% frente a 67,78%), a cambio de más revisión y menor proporción de canciones altas en la lista (70,35% frente a 85,57%). Para proteger el acceso a revisión, se propone atender manualmente los casos sin historial y controlar la diversidad por género. La evaluación viene de una sola fuente sin fecha confirmada. El equipo debe verificar derechos de uso y obtener datos recientes antes de ampliar el alcance. [Auditoría de subgrupos](../../reports/eva3/holdout_subgroups.csv) · [Sensibilidad al cupo en el notebook](../../notebooks/07_examen_transversal.ipynb).

El piloto propuesto dura dos semanas. Comparará lotes similares de revisión manual y asistida. Las metas son **reducir ≥15%** el tiempo mediano por 100 canciones, **alcanzar ≥70%** de aceptación y **limitar a 5 puntos porcentuales** la pérdida de cobertura por género. Son metas pendientes de medición en un entorno operativo.

[Informe técnico y matriz de evaluación](informe_tecnico.md) · [Notebook](../../notebooks/07_examen_transversal.ipynb) · [Presentación](Spotify_ET.pptx)
