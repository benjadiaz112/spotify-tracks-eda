# Acta de verificación de la entrega

Fecha de revisión: 20 de octubre de 2026. Entorno observado: Windows, Python 3.14.4 y versiones de `requirements.txt`.

## Comprobaciones ejecutadas

- `python -m src.eva2.run`: terminó desde el CSV original, produjo modelos, métricas, segmentación, gráficos, informes y ejemplo de inferencia. La ejecución registrada duró 129,26 segundos en este equipo. El tiempo no es una garantía para otros equipos.
- `python scripts/build_notebook.py --execute`: se ejecutaron las ocho celdas de código del cuaderno integral desde un kernel limpio de Jupyter, sin salidas de error. El archivo conserva las salidas y diez figuras.
- `python -m src.eva2.predict data/examples/tracks_example.csv --output reports/eva2/example_predictions.csv`: produjo tres predicciones finitas dentro del dominio.
- `python scripts/verify_delivery.py`: pasaron 20 comprobaciones de entrega. El JSON conserva el detalle, las métricas recalculadas y la salida de las pruebas.
- Las seis pruebas unitarias terminaron en OK, incluida inferencia con género desconocido y rechazo de entradas inválidas.
- Se inspeccionaron visualmente los diez gráficos actuales para comprobar legibilidad y correspondencia con los resultados.

## Coherencia de resultados

Las predicciones guardadas reproducen MAE 11,242897, RMSE 16,227973 y R² 0,368596. La selección coincide con el ranking de CV. No hay IDs ni grupos de audio exacto compartidos entre train y test. K-Means compara 1..10 y no usa popularidad ni género. El informe distingue el criterio de silueta, que elige dos grupos, del codo orientativo de cuatro.

Se verificó que el hash del CSV en Git y en el directorio de trabajo sea el mismo. El cambio de hash respecto del README histórico corresponde a CRLF frente a LF, comprobado sobre los bytes del archivo. `.gitattributes` evita que la recuperación desde Git modifique el CSV de esta entrega.

Las cifras redondeadas del informe, README, notebook y guía de defensa corresponden a los mismos resultados. La mejora frente a la media por género se presenta como 5,74%, en contraste con 34,19% frente a la constante global. No se oculta el MAE alto de los extremos ni se convierte un KPI técnico en ROI comercial.

## Alcance de esta verificación

Se verificó la ejecución en el entorno existente. No se instaló un entorno nuevo desde cero, no se probó Colab ni otro sistema operativo y no hubo un piloto con usuarios de negocio. La licencia y fecha de la fuente pública no pudieron confirmarse independientemente. Estas limitaciones están documentadas.

Las diapositivas no forman parte de esta entrega local y quedan pendientes para Gamma según lo solicitado por el equipo. La guía de defensa y la matriz de rúbrica están disponibles para preparar esa etapa. Esta acta verifica evidencias técnicas, sin asignar una nota académica.

## Git y paquete

La rama de trabajo es `codex/spotify-evaluacion-2`. Los commits registran hitos reales de importación, planificación, preparación, modelos, análisis, documentación y verificación, con el orden de autores indicado por el equipo. No hay remoto configurado ni se realizó push.

El empaquetador exige un directorio sin cambios pendientes, toma los archivos versionados, agrega el commit y el historial en un manifiesto y comprueba cada SHA-256 después de cerrar el ZIP. El resultado se escribe en `output/`, fuera del conjunto versionado para evitar incluir el ZIP dentro de sí mismo.
