# Acta de verificación de la entrega

Fecha de revisión: 20 de octubre de 2026. Entorno observado: Windows, Python 3.14.4 y versiones de `requirements.txt`.

## Comprobaciones ejecutadas

- Se creó un entorno virtual nuevo y se instalaron las dependencias desde `requirements.txt`. `validate_environment.py` confirmó las doce versiones directas y `pip check` terminó sin conflictos. Las 104 dependencias resueltas se registraron en `requirements-lock-windows-py314.txt`.
- `python -m src.eva2.run`: terminó en ese entorno aislado desde el CSV original, produjo modelos, auditoría de etiquetas, ablación controlada, segmentación, gráficos, informes y ejemplo de inferencia. El tiempo efectivo está en `reports/eva2/run.json` y depende del equipo.
- `python scripts/build_notebook.py --execute`: se ejecutaron las diez celdas de código del cuaderno integral desde un kernel limpio que utiliza el Python del entorno virtual, sin salidas de error. El archivo conserva las salidas y diez figuras.
- `python -m src.eva2.predict data/examples/tracks_example.csv --output reports/eva2/example_predictions.csv`: produjo tres predicciones finitas dentro del dominio.
- `python scripts/verify_delivery.py`: pasaron 28 comprobaciones de entrega. El JSON conserva el detalle, las métricas recalculadas y la salida de las pruebas, con el contexto Git existente al comprobar.
- Las diecisiete pruebas unitarias terminaron en OK, incluidas inferencia, contratos de ablación, auditoría de etiquetas, integridad sin Git y fórmulas del piloto.
- Se inspeccionaron visualmente los diez gráficos actuales para comprobar legibilidad y correspondencia con los resultados.

## Coherencia de resultados

Las predicciones guardadas reproducen MAE 11,242897, RMSE 16,227973 y R² 0,368596. La selección coincide con el ranking de CV. No hay IDs ni grupos de audio exacto compartidos entre train y test. K-Means compara 1..10 y no usa popularidad ni género. El informe distingue el criterio de silueta, que elige dos grupos, del codo orientativo de cuatro.

Se verificó que el hash del CSV en Git y en el directorio de trabajo sea el mismo. El cambio de hash respecto del README histórico corresponde a CRLF frente a LF, comprobado sobre los bytes del archivo. `.gitattributes` evita que la recuperación desde Git modifique el CSV de esta entrega.

Las cifras redondeadas del informe, README, notebook y guía de defensa corresponden a los mismos resultados. La mejora frente a la media por género se presenta como 5,74%, en contraste con 34,19% frente a la constante global. No se oculta el MAE alto de los extremos ni se convierte un KPI técnico en ROI comercial.

La ablación corregida mantiene catorce predictores y retira solo `track_genre`, con MAE 15,276752. Se cuantifican 720 IDs con popularidad contradictoria. La sensibilidad cambia solo etiquetas de ajuste en train y conserva la referencia de validación: MAE 11,519260 con primera observación frente a 11,529900 al ajustar con mediana. No cambia el target de test ni selecciona una nueva política por esa métrica.

Se añadió un plan de piloto con fórmulas, plazo, responsables y metas propuestas. La plantilla está vacía y la aplicación la rechaza para impedir resultados comerciales inventados. Los ejemplos usados para probar fórmulas son exclusivamente sintéticos.

## Alcance de esta verificación

Se verificó la ejecución en Windows y Python 3.14.4 con un entorno instalado desde cero. No se probó Colab ni otro sistema operativo y no hubo un piloto con usuarios de negocio. La licencia y fecha de la fuente pública no pudieron confirmarse independientemente. Estas limitaciones están documentadas.

Las diapositivas no forman parte de esta entrega local y quedan pendientes para Gamma según lo solicitado por el equipo. La guía de defensa y la matriz de rúbrica están disponibles para preparar esa etapa. Esta acta verifica evidencias técnicas, sin asignar una nota académica.

## Git y paquete

La rama de trabajo es `feat/spotify-popularidad-clustering`. Los commits registran hitos reales de importación, planificación, preparación, modelos, análisis, documentación y verificación, con el orden de autores indicado por el equipo. El remoto confirmado es `https://github.com/benjadiaz112/spotify-tracks-eda.git`. La integración autorizada conecta los 19 commits de la experiencia 1 con los 13 nuevos, preservando ambos historiales y el informe técnico anterior. La publicación se realiza en esta rama sin modificar `main` ni utilizar force.

El empaquetador exige un directorio sin cambios pendientes para la entrega final; `--snapshot` permite revisar una copia antes del commit. Exporta los archivos académicos, excluye los materiales personales de preparación y coordinación, y comprueba cada SHA-256 después de cerrar el ZIP. El manifiesto contiene versión de formato y hashes de archivos. El resultado se escribe en `output/`, fuera del conjunto versionado para evitar incluir el ZIP dentro de sí mismo.

La verificación usa el manifiesto del ZIP antes de buscar Git. Hay pruebas que impiden invocar Git en ese modo y comprueban que fuentes o entrada alteradas se rechacen. Las salidas pueden regenerarse sin invalidar la identidad del código y dataset originales. El manifiesto identifica integridad, no una firma digital.

`python scripts/validate_archive.py` extrae el ZIP en una carpeta temporal, retira Git del entorno, comprueba dependencias, ejecuta inferencia y corre los 28 controles con las 18 pruebas. La opción `--retrain` añade entrenamiento desde el CSV extraído. El recibo efectivo indica si se usó esa opción y se guarda fuera del conjunto versionado en `output/validacion_zip.json`, asociado al hash del manifiesto. Este control permite repetir la comprobación final sin asumir que Git estará instalado en la máquina del evaluador.
