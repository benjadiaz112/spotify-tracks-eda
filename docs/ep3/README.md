# Spotify Tracks · Evaluación Parcial 3

**MLY1101 — Machine Learning · Caso C · Equipo: Benjamín Palomino y Benjamín Díaz.**

La EP3 continúa el caso de popularidad musical. La entrega técnica comprende la auditoría del CSV, preparación reproducible, regresión comparativa, optimización, ensambles, evaluación agrupada y segmentación exploratoria. La exposición y defensa individual de diez minutos se prepararán por separado.

| Lectura | Contenido |
|---|---|
| [Informe ejecutivo](informe_ejecutivo.md) | Decisión propuesta, hallazgos y condiciones del piloto. |
| [Informe técnico](informe_tecnico.md) | CRISP-DM, datos, optimización, ensambles, generalización, ética y matriz de la pauta. |
| [Protocolo de evaluación](protocolo.md) | Métricas, tolerancias, reglas de selección y alcance. |
| [Protocolo registrado antes de abrir la reserva](protocolo_prerregistro.md) | Criterio y metas fijados por adelantado. |
| [Diseño previo del experimento](protocolo_prerregistro.md) | Partición, presupuesto de búsqueda y criterios antes de evaluar. |
| [Notebook de la EP3](../../notebooks/06_evaluacion_3.ipynb) | Recorrido ejecutable del pipeline y lectura de evidencias. |
| [Resultados](../../reports/eva3/) | JSON/CSV de datos, modelos, folds y predicciones. |
| [Figuras](../../images/eva3/) | Diagnósticos generados con datos de la EP3. |

## Ejecutar en Windows

Desde la raíz del proyecto, con Python 3.14 y PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py314.txt
.\.venv\Scripts\python.exe -m src.eva3.run
.\.venv\Scripts\python.exe scripts/verify_ep3.py
```

El entrenamiento toma el [CSV original](../../data/raw/Spotify_Tracks_Dataset.csv), no modifica el archivo bruto y escribe sus evidencias bajo `reports/eva3/`, `models/eva3/` e `images/eva3/`. El tiempo depende del equipo. Abre el notebook con JupyterLab desde el mismo entorno y usa el kernel de ese Python. La versión archivada permite leer los resultados y ejecutar la verificación sin repetir la búsqueda de modelos.

Para repetir las cinco predicciones con el modelo incluido, usa el CSV de ejemplo sin objetivo:

```powershell
.\.venv\Scripts\python.exe -m src.eva3.predict reports/eva3/prediction_input_example.csv --output output/predictions_ep3.csv
```

El ejemplo usa registros de reserva para comprobar la interfaz, no como datos nuevos de producción. La salida contiene la popularidad estimada. Usa el modelo incluido y el entorno validado. El verificador contrasta las predicciones con la evidencia y funciona sin Git en el ZIP.

En el repositorio quedan los antecedentes de la EP2 con su equipo y resultados de entonces. Esos documentos no son la entrega vigente ni proporcionan un conjunto externo independiente para la EP3.
