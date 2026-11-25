# Spotify Tracks · Evaluación Final Transversal

**MLY1101 · Machine Learning · Caso C**
**Benjamín Palomino y Benjamín Díaz**

Estimamos la popularidad observada de las canciones para priorizar la revisión editorial de un catálogo. La evaluación final integra las mejoras del semestre y reúne la evidencia de los **12 indicadores de la pauta**.

La rama del ET reúne la entrega final y su historial de cambios. [Repositorio en GitHub](https://github.com/benjadiaz112/spotify-tracks-eda/tree/feat/spotify-et) · [Commits del ET](https://github.com/benjadiaz112/spotify-tracks-eda/commits/feat/spotify-et)

| Entregable | Contenido |
|---|---|
| [Informe ejecutivo](docs/et/informe_ejecutivo.md) | Decisión, resultados, riesgos y piloto propuesto. |
| [Informe técnico](docs/et/informe_tecnico.md) | Fuentes, preparación, CRISP-DM, modelos y matriz de los 12 indicadores. |
| [Presentación](docs/et/Spotify_ET.pptx) | Doce diapositivas con gráficos y conclusiones para la exposición. |
| [Notebook ejecutado](notebooks/07_examen_transversal.ipynb) | Reproducción del flujo, auditoría y análisis de la priorización. |
| [Guía de ejecución](docs/et/README.md) | Entorno, instrucciones de reproducción e inferencia de ejemplo. |
| [Datos originales](data/raw/Spotify_Tracks_Dataset.csv) | CSV de la asignatura. |
| [Resultados](reports/eva3/) y [gráficos](images/eva3/) | Evidencia del protocolo de EP3 que se reutiliza en el examen. |
| [Auditorías del ET](reports/et/) | Ablación de historial y diagnóstico de error por segmento. |

## Resultado y alcance

Se conservaron **89.495 canciones** de 114.000 filas. El **HGB ajustado** obtuvo el menor MAE medio en los cinco folds agrupados de desarrollo (5,129 puntos). En 17.913 canciones reservadas obtuvo **MAE de 5,02 puntos**, **RMSE de 8,50**, **R² de 0,830** y **91,97 % de predicciones dentro de ±15 puntos**. Cada indicador responde a una pregunta distinta.

Con un cupo fijo de 3.582 revisiones, el 85,57 % de las canciones seleccionadas tuvo popularidad observada alta (≥49), frente al 62,00 % de la referencia EP2. Es una simulación sobre la reserva histórica. En artistas sin historial de desarrollo el MAE sube a 7,85 puntos. El piloto propuesto mediría tiempo de revisión, aceptación y cobertura de géneros e incluiría revisión manual de ese grupo.

## Reproducir la entrega

El entorno comprobado usa Windows y Python 3.14. La [guía del ET](docs/et/README.md) explica cómo instalar las dependencias, comprobar la evidencia guardada y volver a entrenar desde el CSV. El repositorio conserva la [guía de EP3](docs/ep3/README.md) como antecedente del experimento.
