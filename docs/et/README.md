# Spotify Tracks · Examen Final Transversal

**MLY1101 · Caso C · Benjamín Palomino y Benjamín Díaz**

Estimamos la popularidad observada de canciones y exploramos perfiles acústicos para ordenar la revisión de un catálogo. La entrega reúne los avances del semestre, documenta su evidencia y adapta la comunicación a los doce indicadores del examen.

| Entregable | Uso |
|---|---|
| [Informe ejecutivo](informe_ejecutivo.md) | Decisión de negocio, resultados, riesgos y piloto propuesto. |
| [Informe técnico](informe_tecnico.md) | Fuentes, metodología, preparación, modelos, evaluación y matriz de la pauta. |
| [Presentación](Spotify_ET.pptx) | Evidencias para la exposición de diez minutos. |
| [Notebook ejecutado](../../notebooks/07_examen_transversal.ipynb) | Recorrido reproducible desde el CSV hasta la interpretación. |
| [Datos originales](../../data/raw/Spotify_Tracks_Dataset.csv) | Archivo de la asignatura incluido en la entrega. |
| [Evidencias del modelo](../../reports/eva3/) y [figuras](../../images/eva3/) | Resultados, particiones, búsqueda, comparación de modelos y gráficos. |
| [Auditorías del ET](../../reports/et/) | Ablación de historial y comparación operativa de segmentos. |

## Abrir la entrega

Si descargaste el ZIP completo, extráelo y abre la carpeta `spotify-et`. Si descargaste las dos partes, extráelas en el mismo directorio para que ambas escriban en la misma carpeta `spotify-et`. El archivo `MANIFIESTO_ENTREGA.json` registra el commit y las huellas del conjunto completo.

Empieza por el informe ejecutivo y abre el notebook para ver sus salidas. Los resultados y modelos guardados permiten revisar la entrega antes de volver a entrenar.

## Reproducir en Windows

Desde la raíz del proyecto, con Python 3.14 y PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py314.txt
.\.venv\Scripts\python.exe scripts/validate_environment.py --check-only
.\.venv\Scripts\python.exe -m jupyterlab
```

En JupyterLab abre `notebooks/07_examen_transversal.ipynb`, elige el kernel del entorno y ejecuta las celdas en orden. La ejecución normal vuelve a ajustar los modelos. Para recorrer el notebook usando el ajuste supervisado guardado, ejecuta `$env:SPOTIFY_ET_REUSE = '1'` antes de iniciar JupyterLab. Para volver al ajuste completo, abre una nueva terminal o limpia esa variable antes de iniciar JupyterLab. El notebook escribe la reproducción bajo `output/et_reproduccion/`.

Desde la raíz puedes comprobar las evidencias y reproducir cinco predicciones con el modelo guardado:

```powershell
.\.venv\Scripts\python.exe scripts/verify_ep3.py --check-only
.\.venv\Scripts\python.exe -m src.eva3.predict reports/eva3/prediction_input_example.csv
```

La salida de inferencia queda en `output/predictions_ep3.csv`. Los registros de ejemplo pertenecen a la reserva. La carpeta `reports/eva3/` identifica la etapa en que se midió la evidencia reutilizada en el examen.

El notebook recalcula las auditorías de historial y segmentos en su carpeta de reproducción. En el modo de ajuste guardado, la ablación vuelve a entrenar sus dos variantes con parámetros y particiones fijos.

## Generar los archivos de entrega

Desde el repositorio limpio y con el entorno instalado:

```powershell
.\.venv\Scripts\python.exe scripts/package_et.py
```

El empaquetador verifica los enlaces locales, los archivos extraídos y la inferencia de ejemplo. Genera `output/entrega_spotify_et.zip` y, cuando el archivo supera 25.000.000 bytes, dos ZIP de menos de ese límite con el mismo contenido repartido entre ambos.
