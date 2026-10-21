# Informe ejecutivo - Inteligencia musical en Spotify

**Evaluación Parcial N°2, MLY1101. Caso C. Continuación de experiencia 1.**

Equipo: Benjamín Palomino, Benjamín Díaz y Freymuth Christopher Isaac.

## Decisión que apoya el proyecto

Un equipo de curaduría necesita explorar su catálogo y estimar la popularidad observada de canciones mediante sus características de audio y género. Proponemos una herramienta de consulta local con revisión humana. El dataset es una instantánea sin fecha de extracción confirmada: el proyecto no demuestra que pueda anticipar futuros éxitos.

## Evidencia principal

El CSV recibido contiene 114,000 filas y 89,741 IDs únicos. La limpieza de experiencia 1 deja 89,495 canciones, retiene 99.73% de los IDs originales y no deja nulos. Detectamos además 6,213 filas con audio idéntico a otras de diferente ID. Para evitar contaminación entre entrenamiento y prueba, estas versiones permanecen en el mismo conjunto.

Reservamos 17,715 canciones (19.79%) para prueba y seleccionamos el modelo mediante validación cruzada agrupada de tres folds en las 71,780 restantes. Comparamos dos referencias constantes y tres modelos supervisados: regresión lineal, árbol de decisión y HistGradientBoosting.

El modelo seleccionado, **HistGradientBoosting**, obtiene **MAE 11.24 puntos**, RMSE 16.23 y R² 0.369 en prueba. Reduce el MAE en **34.19%** frente a la media constante (17.08). Cumple las metas previas MAE <= 15 y mejora >= 10%. El IC bootstrap agrupado del MAE es [10.99, 11.46], condicionado a esta prueba y a este modelo.

La comparación más exigente usa la media de popularidad por género calculada solo en entrenamiento: MAE 11.93. Frente a ella, la mejora del modelo completo es **5.74%**. La ablación que retira únicamente género, conservando las otras catorce variables e hiperparámetros, registra MAE 15.28. Esta evidencia indica una dependencia importante del género, sin demostrar un efecto causal del sonido.

La auditoría encontró 720 IDs con distintas popularidades entre filas originales (0.80% de IDs); la diferencia mediana es 1 punto y la máxima 44. Se mantiene la primera observación como referencia reproducible por continuidad y falta de fechas autoritativas. Entrenar con mediana cambia el MAE de CV de 11.519 a 11.530, sin modificar etiquetas de test ni la selección. La incertidumbre de esas etiquetas sigue declarada.

## Segmentación para curaduría

K-Means usa diez variables continuas de audio escaladas, excluye popularidad y género, y compara K=1..10. Se elige K=2 por la mayor silueta entre soluciones con grupos de al menos 3% del entrenamiento. El codo orientativo sugiere K=4, así que no existe un único óptimo respaldado por ambos criterios.

- Grupo 0: 75.3% del entrenamiento, energía 0.75, acústica 0.19 y tempo 126.2. Perfil relativamente energético para explorar selecciones de actividad.
- Grupo 1: 24.7%, energía 0.30, acústica 0.76 y tempo 109.9. Perfil relativamente acústico para explorar selecciones tranquilas.

Son nombres interpretativos, no etiquetas musicales verificadas. La silueta de prueba es 0.237, compatible con separación limitada. Las semillas alternativas dan ARI entre 0.997 y 0.997. PCA en dos dimensiones conserva 44.4% de varianza: sirve como mapa parcial, no como demostración de grupos perfectamente separados.

## Recomendación y límites

Recomendamos un piloto de exploración de catálogo con evaluación humana. No automatizar contratos ni presupuestos de promoción. El MAE para canciones de popularidad 81-100 es 44.91 en solo 105 casos: el modelo subestima los extremos. El percentil 95 del error absoluto es 37.30 puntos.

No se puede calcular ROI con estos datos: faltan costos, resultados de campañas y una evaluación causal. El [piloto propuesto](plan_piloto_negocio.md), aún no ejecutado, fija dos semanas y tres metas sujetas a aprobación: reducción mediana del tiempo >=15%, aceptación de recomendaciones >=70% y pérdida mediana de cobertura de géneros no mayor a 5 puntos porcentuales. Define fórmulas, responsables, doce pares de sesiones exploratorios y una plantilla vacía con calculadora validada. Antes de uso real se requiere una muestra reciente, evaluación temporal, revisión de la etiqueta género y validación de derechos de uso.

El desarrollo completo, fórmulas, decisiones, riesgos y matriz de rúbrica están en [informe técnico](informe_tecnico_eva2.md). El [notebook integral](../notebooks/05_proyecto_completo_eva2.ipynb) reproduce las etapas desde el CSV original.
