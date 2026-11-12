# Spotify Tracks · Evaluación Parcial 3

**MLY1101 · Machine Learning · Caso C**
**Benjamín Palomino y Benjamín Díaz**

El proyecto estima la popularidad observada de canciones para priorizar la revisión de un catálogo musical. La EP3 compara regresores, optimiza hiperparámetros, mide el aporte de ensambles y explora perfiles de audio con K-Means. El modelo apoya la decisión de curaduría con revisión humana.

## Entregables

| Archivo | Contenido |
|---|---|
| [Presentación EP3](docs/ep3/Spotify_EP3.pptx) | Once diapositivas con gráficos, resultados y conclusiones. |
| [Informe ejecutivo](docs/ep3/informe_ejecutivo.md) | Problema, resultados, límites y decisión propuesta. |
| [Informe técnico](docs/ep3/informe_tecnico.md) | Metodología, EDA, optimización, ensambles y matriz de la pauta. |
| [Notebook ejecutado](notebooks/06_evaluacion_3.ipynb) | Preparación, trazabilidad, EDA, comparación de modelos y evaluación. |
| [Guía de ejecución](docs/ep3/README.md) | Instrucciones para reproducir y verificar la EP3. |
| [Protocolo del experimento](docs/ep3/protocolo_prerregistro.md) | Partición, métricas y criterio de selección. |
| [Resultados medidos](reports/eva3/) | Folds, configuraciones, predicciones, auditorías y métricas. |
| [Gráficos](images/eva3/) | Distribuciones, correlaciones, comparación de modelos y segmentos. |
| [Dataset original](data/raw/Spotify_Tracks_Dataset.csv) | CSV incluido en el proyecto. |

## Resultados principales

Se retuvieron 89.495 canciones de las 114.000 filas originales. Se separaron familias relacionadas en 71.582 canciones de desarrollo y 17.913 de reserva. La selección se realizó por el menor MAE medio en cinco folds agrupados de desarrollo.

| Indicador del HGB ajustado | Resultado |
|---|---:|
| MAE en validación cruzada | 5,129 puntos |
| MAE en reserva | 5,020 puntos |
| RMSE en reserva | 8,501 puntos |
| R² en reserva | 0,830 |
| Predicciones a ±15 puntos | 91,97 % |

La referencia EP2 se reentrenó sobre la misma partición para comparar el MAE, que pasó de 11,38 a 5,02 puntos. ExtraTrees redujo el MAE un 24,35 % frente a su propio primer árbol. Las correlaciones de audio se calcularon exclusivamente sobre desarrollo.

En artistas sin historial en desarrollo, el MAE aumenta a 7,85 puntos. El piloto propuesto incluye revisión manual de esos casos. La fuente carece de fechas para evaluar predicciones futuras. El R² y la cobertura dentro de un margen son métricas diferentes.

## Ejecutar y verificar

Entorno validado en Windows con Python 3.14.4. Desde la raíz del repositorio en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py314.txt
.\.venv\Scripts\python.exe scripts/validate_environment.py --check-only
.\.venv\Scripts\python.exe scripts/verify_ep3.py --check-only
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Para ejecutar el entrenamiento y las demás etapas desde el CSV original:

```powershell
.\.venv\Scripts\python.exe -m src.eva3.run
```

Para abrir el notebook, iniciar JupyterLab y seleccionar el kernel del entorno:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

El [notebook de EP3](notebooks/06_evaluacion_3.ipynb) usa reentrenamiento completo por defecto. La copia ejecutada identifica el modo de inspección del ajuste supervisado guardado, con preparación, EDA y etapas posteriores recalculadas. El soporte de otros entornos requiere verificación.

## Estructura

```text
data/raw/                         Dataset original
docs/ep3/                         Informes, presentación y protocolos
notebooks/06_evaluacion_3.ipynb     Notebook de EP3
src/eva3/                         Preparación, modelos, evaluación y gráficos
models/eva3/                      Modelos guardados
reports/eva3/                     Evidencias y controles
images/eva3/                      Figuras del análisis
scripts/                          Construcción, verificación y empaquetado
tests/                            Pruebas del proyecto
```

## Empaquetar la entrega

Después de registrar los cambios en Git:

```powershell
.\.venv\Scripts\python.exe scripts/package_ep3.py
```

El ZIP se genera en `output/entrega_spotify_ep3.zip`. Incluye el dataset, notebook, código, modelos, informes, presentación y evidencias, con un manifiesto de integridad. El empaquetador verifica la copia extraída y reproduce las predicciones de ejemplo.

## Antecedentes

La [experiencia 1](docs/experiencia1/README_original.md) y el [informe de EP2](docs/informe_tecnico_eva2.md) documentan las entregas previas. La entrega vigente y el equipo actual corresponden a EP3.
