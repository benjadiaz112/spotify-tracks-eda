# Guía de defensa individual

Los tres integrantes deben poder explicar todo el proyecto. La distribución de exposición no limita las preguntas que puede hacer el docente. Este documento prepara la defensa técnica; las diapositivas se elaborarán posteriormente en Gamma.

## Explicación central en un minuto

Estudiamos si audio y género permiten estimar la popularidad observada de canciones de Spotify. Continuamos la limpieza de experiencia 1 y conservamos 89.495 canciones. Detectamos versiones con audio idéntico y las mantuvimos juntas al dividir train/test. Comparamos regresión lineal, árbol y un ensamble, con referencias constantes y validación cruzada agrupada. El ganador tuvo MAE 11,24 frente a 17,08 de la media, pero el género explica gran parte del rendimiento y los extremos tienen errores grandes. K-Means encontró dos macroperfiles relativamente energético y acústico, con separación limitada. Recomendamos exploración con revisión humana y una nueva validación temporal antes de uso real.

## Preguntas cruzadas y respuestas razonadas

### 1. ¿Cuál es el problema de negocio y qué decisión cambia?

Proponemos apoyar a un curador al explorar catálogo y consultar una estimación de popularidad observada. No hemos validado necesidades con una empresa real. La decisión respaldada es un piloto de revisión, no asignación automática de contratos o presupuesto.

### 2. ¿Por qué regresión y no clasificación?

`popularity` es una puntuación numérica de 0 a 100. Queremos estimar su magnitud. No definimos una etiqueta popular/no popular y por eso accuracy, F1 y matriz de confusión no corresponden a esta tarea.

### 3. ¿Aplicaron las seis fases CRISP-DM?

Sí: negocio y KPIs; comprensión de fuentes y calidad; limpieza y partición; modelos; evaluación con test, residuos y subgrupos; y despliegue local con inferencia y plan de mantenimiento. El despliegue implementado es batch local y no producción web.

### 4. ¿Qué mejoró respecto de experiencia 1?

Experiencia 1 preparaba datos, pero dejaba pendiente el modelado. Completamos comparación supervisada, clustering, evaluación y una inferencia utilizable. También auditamos audio idéntico de distinto ID, ajustamos la partición y aclaramos el hash según saltos de línea.

### 5. ¿Por qué conservaron solo 78,50% de las filas?

Gran parte son apariciones repetidas del mismo `track_id` bajo distintos géneros. De 89.741 IDs originales conservamos 89.495, el 99,73%. Los documentos antiguos usaban dos denominadores: filas en el problema e IDs en el informe. La referencia de 95% no se cumple sobre filas, pero sí sobre IDs. Mostramos ambas medidas sin sustituir una por otra para simular éxito ni afirmar que eso explique la nota anterior.

### 6. ¿Qué sesgo introduce quedarse con la primera fila por ID?

El archivo está dispuesto en bloques por género y un tema puede aparecer en varias etiquetas. Hay 16.299 IDs asociados a varios géneros. La primera etiqueta no tiene por qué representar el género principal. Afecta la interpretación y el modelo depende mucho de ella. Una nueva versión debería validar una codificación multietiqueta y evaluarse con otro holdout.

### 7. ¿Por qué no basta con quitar el ID de los predictores?

Dos IDs pueden representar versiones con exactamente las mismas características acústicas. Si aparecen en lados distintos, la prueba puede ser demasiado optimista. Agrupamos por 14 variables acústicas/binarias, sin target ni género, y evitamos grupos compartidos. No resuelve versiones casi idénticas ni toda dependencia entre artistas.

### 8. ¿Por qué el test no es exactamente 20% de canciones?

Se divide 80/20 de grupos de audio, manteniéndolos completos. Los tamaños de los grupos varían. El resultado es 71.780 canciones de train y 17.715 de test, 19,79% de filas de prueba. La estratificación usa deciles de mediana del target por grupo.

### 9. ¿Cómo evitan fuga en el escalado y la búsqueda?

Cada modelo es un pipeline con preparación. GroupKFold genera tres particiones sin audio exacto compartido. GridSearchCV ajusta codificadores y escalador solo con la parte de entrenamiento de cada fold. La elección se guarda antes de predecir el test. No usamos las métricas de test para elegir parámetros.

### 10. ¿Por qué esos algoritmos?

Regresión lineal y árbol son modelos abordados en clase y contrastan una relación lineal con reglas no lineales. HistGradientBoosting añade un ensamble de árboles. Los Dummy muestran si aprender aporta algo respecto de una constante. Mayor complejidad no garantiza mejor generalización: el criterio fue MAE de CV.

### 11. ¿Qué parámetros del ganador deben saber explicar?

220 iteraciones, tasa de aprendizaje 0,08, hasta 31 hojas y regularización L2 de 10. Las hojas controlan complejidad y L2 penaliza ajustes excesivos. Se probaron 15/31 hojas y L2 1/10. Early stopping quedó desactivado para evitar una validación interna por filas que mezclara grupos exactos. Semilla 42. No ajustamos iteraciones al observar el test.

### 12. ¿Qué significan MAE, RMSE y R²?

MAE 11,24: error absoluto medio de 11,24 puntos. RMSE 16,23: raíz del error cuadrático medio, más sensible a errores grandes. R² 0,369: alrededor de 36,9% de variación explicada en esta prueba según esa comparación matemática. No quiere decir 36,9% de canciones acertadas. R² puede ser negativo.

### 13. ¿Un error promedio de 11 puntos sirve para cada canción?

No. P95 del error absoluto es 37,30 puntos. Para popularidad 81-100 el MAE es 44,91 en 105 casos. El modelo tiende a acercarse a valores centrales y subestima los extremos. No es un detector fiable de futuros éxitos.

### 14. ¿La mejora de 34,19% cuenta toda la historia?

Es contra la constante global. La referencia por género de train ya logra MAE 11,93, de modo que la mejora adicional del modelo es aproximadamente 5,74%. La ablación corregida retira únicamente género, conserva las otras catorce variables y los hiperparámetros, y da MAE 15,28. El género aporta mucha señal. No estamos demostrando una relación causal entre sonido y éxito.

### 15. ¿Por qué agregar una referencia de mediana?

La media minimiza error cuadrático entre constantes y la mediana minimiza error absoluto. Como seleccionamos por MAE, conviene comparar con ambas. En esta muestra sus errores de prueba son casi iguales. Todos los valores se aprenden en train o dentro del fold correspondiente.

### 16. ¿Para qué sirve el bootstrap y qué no permite afirmar?

Remuestrea grupos exactos de test 600 veces. El IC del MAE es aproximadamente 10,99 a 11,46, condicionado al modelo y muestra observados. No incorpora reentrenamientos ni todas las relaciones por artista y no es un intervalo de error para cada canción.

### 17. ¿Qué hace K-Means y por qué escalar?

Asigna filas al centroide más cercano y actualiza centroides buscando reducir la suma de distancias cuadradas internas. Escalamos para que milisegundos no dominen indicadores 0-1. Se usan diez continuas de audio. Popularidad, género y nombres quedan fuera del ajuste.

### 18. ¿Por qué K=2 si el codo sugiere K=4?

No hay un óptimo único confirmado. Aplicamos una regla explícita: mayor silueta en muestra fija de train entre K=2..10, con todos los grupos de al menos 3%. K=2 gana con 0,248 en train y 0,237 en prueba. El codo orientativo sugiere cuatro y lo reportamos. Dos macroperfiles son una primera segmentación parsimoniosa, y cuatro grupos quedarían como alternativa a validar con curadores.

### 19. ¿Una silueta de 0,237 es excelente?

No. Indica separación limitada y fronteras superpuestas. Los grupos pueden ser útiles para explorar promedios, pero no los presentamos como estilos musicales definitivos. La estabilidad entre semillas, con ARI alrededor de 0,997, responde otra pregunta: reproducibilidad de la asignación, no calidad semántica ni estabilidad temporal.

### 20. ¿Qué representan los dos perfiles?

En train, el grupo 0 cubre 75,3%, con energía 0,75 y acústica 0,19. El grupo 1 cubre 24,7%, con energía 0,30 y acústica 0,76. Son relativamente energético y acústico/suave. Sugerir listas de actividad o tranquilas es una hipótesis que requiere escuchar y revisar diversidad, no un efecto comercial medido.

### 21. ¿PCA demuestra la separación de los grupos?

No. Dos componentes retienen solo 44,44% de varianza y hacen un mapa parcial. Se requieren seis para 80%. K-Means se ajusta en las diez dimensiones escaladas. La proyección no cambia cómo se entrenó el modelo ni prueba que existan géneros naturales.

### 22. ¿Validaron artistas nuevos o el futuro?

No de forma estricta. El 72,57% de las filas de test comparte la cadena de artistas con train. El error es 10,91 para cadenas vistas y 12,12 para no vistas, pero las colaboraciones pueden solaparse. Sin fechas no hay evaluación temporal. Haría falta una nueva prueba diseñada para esos usos.

### 23. ¿Qué riesgo ético es más relevante?

Confundir popularidad observada con valor artístico y usarla para excluir artistas menos expuestos. También hay dependencia del género, error desigual y poca capacidad en extremos. Proponemos revisión humana, auditoría de segmentos y diversidad. No tenemos atributos demográficos para certificar equidad entre grupos protegidos.

### 24. ¿Pueden prometer ROI o una nota 7?

No. Faltan datos de campañas, costos y resultados para ROI. La rúbrica tiene evidencia para cada indicador, pero la calificación depende del docente y de la defensa. La respuesta profesional es explicar lo comprobado y lo que sigue sin validarse.

### 25. ¿Cómo demuestran que funciona?

Ejecutar `python -m src.eva2.run`, reiniciar y ejecutar completo el notebook 05, correr `python -m unittest discover -s tests -v` y `python scripts/verify_delivery.py`. La CLI acepta un CSV sin target y devuelve predicciones. Las métricas se pueden recalcular desde `test_predictions.csv`.

## Preparación del equipo

Cada integrante debe abrir el notebook ejecutado, explicar por qué se separan grupos y reproducir la inferencia. Debe poder localizar `selection.json`, las métricas, el manifiesto de partición y los gráficos de K/PCA. Practiquen responder con el dato y su límite, sin memorizar cifras sin entenderlas.

### 26. ¿Cómo trataron las popularidades distintas del mismo ID?

Hay 720 IDs afectados, el 0,802% del total. El rango mediano es un punto, el máximo 44 y diez IDs superan 20 puntos. Se conserva la primera fila por continuidad y trazabilidad, sin afirmar que sea el valor verdadero de una fecha concreta. No hay fecha autoritativa para elegir el máximo o la última fila. Medimos sensibilidad cambiando solo las etiquetas de ajuste por mediana en los folds originales de train: MAE 11,530 frente a 11,519 con primera fila. No se cambió el test ni se eligió una regla por ganar esa comparación. La referencia imperfecta sigue declarada.

### 27. ¿Cuál es el KPI de negocio y qué queda por validar?

El piloto propuesto compara sesiones manuales y asistidas durante dos semanas. Metas para aprobación por curaduría: reducción mediana de tiempo por par >=15%, aceptación agregada >=70% y delta mediano de cobertura de géneros >=-5 pp. Normalizamos el tiempo a 100 canciones y calculamos aceptación como suma de aceptadas / suma de recomendaciones. Son metas propuestas, no resultados observados. La plantilla está vacía; las pruebas de la calculadora son sintéticas.

### 28. ¿Por qué 220 iteraciones y tasa 0,08?

Son una configuración fija de alcance computacional moderado y actualizaciones graduales. No demostramos que sean óptimas. La grilla explora hojas 15/31 y regularización L2 1/10 en CV agrupada; no ajustamos por test. El informe explica qué controla cada parámetro y que el ganador llega a bordes de una búsqueda limitada. Para ampliar la búsqueda haría falta un protocolo nuevo de evaluación.

### 29. ¿El profesor necesita Git para ejecutar el ZIP?

No. La entrega incluye un manifiesto de hashes que el verificador utiliza antes de buscar Git. Verifica código e input, mientras permite regenerar resultados. Se prueba explícitamente la ruta sin `.git` y el rechazo de archivos alterados. El manifiesto no es una firma digital. El empaquetador sí está pensado para trabajar desde el repositorio versionado del equipo.
