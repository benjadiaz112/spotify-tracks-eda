# Piloto propuesto de curaduría musical

**Estado: no ejecutado.** No existen mediciones comerciales ni ROI. Todas las metas y tamaños de este documento son propuestas de diseño que debe aprobar el responsable de curaduría antes de usar la herramienta. No provienen de resultados observados ni de una exigencia numérica del profesor.

## Objetivo, alcance y decisión

Objetivo propuesto: reducir el tiempo de revisar un catálogo sin deteriorar la utilidad percibida ni la diversidad de la selección. El piloto compara revisión manual con revisión asistida por score y perfiles musicales. La decisión será continuar, ajustar o detener un piloto controlado. No autoriza promociones, contratos ni inversión automática.

La predicción de popularidad no es una valoración artística. Ambas condiciones deben permitir escuchar las canciones y rechazar recomendaciones. El modelo no debe ocultar canciones con baja puntuación. Mantener visibles los límites sobre fecha del dataset, género y error en extremos.

## Diseño y calendario propuestos

- Duración: dos semanas. Semana 1 para validar metas, formar revisores y preparar lotes. Semana 2 para sesiones y análisis.
- Tamaño exploratorio: dos revisores y seis pares de sesiones por revisor, doce pares en total. No es un cálculo de potencia ni permite garantizar significancia estadística.
- Por par, dos lotes distintos de 100 canciones con composición comparable por género y popularidad histórica. Cada sesión debe seleccionar 20 canciones. Los mismos objetivos y criterios de curaduría se aplican a ambas condiciones.
- Asignar al azar qué lote recibe revisión manual o asistida. Contrabalancear el orden para reducir aprendizaje y fatiga. Un revisor no debe evaluar la misma canción dos veces en ese piloto.
- Registrar el tiempo con el mismo criterio de inicio/fin, incluyendo consulta de score y escucha. Registrar también fallas y abandonos. No borrar sesiones desfavorables: reportarlas separadas con motivo.
- El protocolo y las metas se fijan antes de observar resultados. El análisis usa pares del mismo revisor y carga de trabajo, sin tratar sesiones del mismo revisor como plenamente independientes.

No se usará este piloto para volver a seleccionar modelos con el test académico. Antes de uso comercial se necesita validar una muestra reciente y derechos de uso. La popularidad histórica sirve para preparar lotes comparables y no como etiqueta de éxito de una intervención.

## KPIs, fórmulas y umbrales de decisión

| KPI | Fórmula y unidad | Meta propuesta | Responsable propuesto |
|---|---|---|---|
| Tiempo normalizado | Minutos por 100 = segundos / 60 / canciones revisadas x 100. Reducción por par = 100 x (1 - tiempo_asistido / tiempo_manual) | Mediana de las reducciones por par >=15% al terminar semana 2 | Coordinador de curaduría recoge tiempos; responsable de datos calcula |
| Aceptación de recomendaciones | 100 x suma(recomendaciones aceptadas) / suma(recomendaciones mostradas) en sesiones asistidas | >=70% al terminar semana 2 | Revisores registran aceptación; coordinador revisa criterios |
| Cobertura de géneros | 100 x géneros distintos seleccionados / mínimo(géneros disponibles, canciones seleccionadas). Delta = cobertura_asistida - cobertura_manual, en puntos porcentuales | Mediana del delta por par >= -5 pp | Responsable de datos audita; coordinador evalúa diversidad |

La cobertura usa la etiqueta disponible, que puede ser incompleta por la deduplicación. No certifica diversidad cultural ni equidad. Deben revisarse también artistas seleccionados y muestras escuchadas, sin inferir características sensibles de las personas.

**Continuar** solo si se cumplen las tres metas y no se detecta una barrera relevante de calidad o derechos. **Ajustar y repetir con nueva muestra** si el tiempo mejora pero aceptación o diversidad fallan. **Detener** si no hay mejora útil o aparecen daños relevantes. Estas reglas son propuestas y no una decisión comercial ya validada. No convertir minutos ahorrados a dinero sin costos efectivamente aportados por el negocio.

## Registro reproducible

La plantilla `data/examples/pilot_template.csv` contiene únicamente encabezados. Una fila representa una sesión, con:

- `reviewer_alias`: alias del revisor, sin nombre personal.
- `pair_id`: identificador del par de lotes comparables.
- `method`: `manual` o `assisted`.
- `reviewed_tracks`, `selected_tracks`, `duration_seconds`: carga, selección y tiempo efectivo.
- `recommended_tracks`, `accepted_recommendations`: cero en manual; contadores reales en asistida.
- `selected_genres`, `available_genres`: géneros distintos seleccionados y disponibles en el lote.

Cada par debe incluir ambos métodos, igual número de canciones revisadas y seleccionadas, y lotes distintos según el protocolo. Los conteos deben ser enteros, no negativos y no exceder su población. La aplicación calcula métricas por par y guarda también un resumen JSON: reducción con mediana, aceptación con razón de sumas y delta de cobertura con mediana.

Una vez recogidos los datos reales:

```powershell
python -m src.eva2.pilot data/pilot_observations.csv --output reports/pilot/kpis_by_pair.csv
```

No ejecutar ese comando esperando resultados desde la plantilla vacía. El programa la rechaza expresamente. Los datos sintéticos de `tests/test_pilot.py` verifican fórmulas y no son evidencia de impacto comercial.
