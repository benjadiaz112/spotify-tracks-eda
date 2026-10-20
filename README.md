# Spotify Tracks - Evaluación Parcial N°2

Proyecto de MLY1101 que continúa la experiencia 1: estima popularidad observada con regresión y descubre perfiles musicales con K-Means y PCA.

**Equipo:** fluzxks, benjadiaz112 y Freymuth Christopher Isaac. **Caso C.**

## Abrir la entrega

- [Informe ejecutivo](docs/informe_ejecutivo.md): síntesis de negocio, resultados y decisión propuesta.
- [Informe técnico y matriz de rúbrica](docs/informe_tecnico_eva2.md): desarrollo completo con CRISP-DM, EDA, modelos y ética.
- [Notebook integral ejecutado](notebooks/05_proyecto_completo_eva2.ipynb): reproduce todo desde raw.
- [Guía de defensa](docs/defensa_individual.md): guion de diez minutos y preguntas cruzadas.
- [Resultados verificables](reports/eva2/): métricas, parámetros, particiones y predicciones.

El dataset original **sí está incluido** en `data/raw/Spotify_Tracks_Dataset.csv`. No necesitas descargar otro ni una cuenta de Spotify. SHA-256: `335e5ee0a07a3f30b20ca9860c87562c98082a032e0ecd90e8ca0a86d965fe58`.

## Resultado medido

| Modelo | MAE CV | MAE prueba | RMSE prueba | R² prueba |
|---|---:|---:|---:|---:|
| HistGradientBoosting, seleccionado por CV | 11,519 | 11,243 | 16,228 | 0,369 |
| Regresión lineal | 12,183 | 11,908 | 16,679 | 0,333 |
| Árbol de decisión | 14,019 | 13,847 | 17,969 | 0,226 |
| Referencia de mediana | 17,288 | 17,085 | 20,427 | -0,0005 |
| Referencia de media | 17,290 | 17,085 | 20,425 | -0,0002 |

Prueba de 17.715 canciones separando grupos de audio idéntico. Mejora del MAE: 34,19% frente a la constante global y aproximadamente 5,74% frente a la media por género aprendida en train. El género aporta gran parte de la señal. Los errores crecen mucho para canciones de popularidad alta. El uso respaldado es exploración con revisión humana y no predicción de futuros éxitos.

K-Means elige dos macroperfiles, energético y acústico, con una regla explícita de silueta y tamaño mínimo. Silueta de prueba: 0,237. El codo orientativo sugiere cuatro grupos. PCA en dos dimensiones conserva 44,44% de varianza. El informe explica estas limitaciones.

## Instalación local

Entorno ejecutado: **Python 3.14.4 en Windows**, scikit-learn 1.8.0 y dependencias exactas en `requirements.txt`. No se han probado otros sistemas. Desde una terminal abierta en esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se usa el ejecutable del entorno directamente para evitar problemas al activar scripts en PowerShell. No pegues los comandos en una celda Python.

## Reproducir la evaluación completa

Desde la raíz y en la misma terminal:

```powershell
.\.venv\Scripts\python.exe -m src.eva2.run
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_delivery.py
```

El primer comando verifica el CSV, limpia, divide, entrena, selecciona por CV, evalúa, agrupa y genera figuras, informes y modelos. El tiempo depende del equipo. Los resultados actuales se escriben en `reports/eva2/`, `images/eva2/` y `models/eva2/`.

Para abrir el cuaderno:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Abre `notebooks/05_proyecto_completo_eva2.ipynb`, selecciona el kernel del entorno y usa **Restart Kernel and Run All Cells**. Los notebooks 01-04 son antecedentes y no son prerrequisitos. Para regenerar el notebook ejecutado desde terminal:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py --execute
```

El notebook encuentra la raíz buscando el CSV en el directorio actual y sus padres. En Colab debes subir y descomprimir el proyecto completo y disponer de las dependencias indicadas. Esa plataforma no se probó en esta ejecución.

## Comprobar inferencia

```powershell
.\.venv\Scripts\python.exe -m src.eva2.predict data/examples/tracks_example.csv --output reports/eva2/example_predictions.csv
```

El ejemplo incluye tres canciones del holdout sin el objetivo. La salida contiene `predicted_popularity` entre 0 y 100 y `unknown_genre`. Un género nuevo permite ejecutar, pero requiere revisar cobertura. La herramienta rechaza entradas vacías, faltantes y valores inválidos. Carga únicamente los modelos propios del proyecto: joblib no es un formato seguro para archivos de procedencia desconocida.

## Estructura y continuidad

```text
data/raw/                CSV original entregado
data/processed/          Preparación histórica de experiencia 1
data/examples/           Entrada de ejemplo sin target
notebooks/01..04          Antecedentes históricos
notebooks/05_*.ipynb      Notebook integral vigente
src/data,analysis,features/ Código original conservado
src/eva2/                Pipeline actual de principio a fin
models/eva2/             Modelos nuevos y metadatos
images/eva2/             Figuras actuales
reports/eva2/            Evidencias calculadas y verificación
docs/                    Informes, fuentes y defensa
scripts/                 Construcción de notebook y verificación
tests/                   Pruebas de fuga, contrato y persistencia
```

El [README anterior](docs/experiencia1/README_original.md) se conserva como antecedente, incluidas sus cifras históricas. La entrega corrige el hash y sustituye la partición por filas por otra agrupada. Calcula resultados desde raw y no usa los antiguos `data/processed/train.csv`, `test.csv` ni `models/preprocessing_pipeline.joblib`.

## Fuentes, riesgos y mantenimiento

[Fuentes y clases](docs/fuentes_eva2.md). [Plan CRISP-DM](docs/plan_evaluacion_2.md). [Equipo y Git](docs/equipo_y_git.md).

La fuente pública y licencia son las declaradas en experiencia 1. Su versión y licencia actual no se pudieron confirmar independientemente. Validar derechos antes de publicar el dataset. No se conoce la fecha de extracción. El género conservado depende del orden del archivo. No se estiman costos ni ROI sin campañas. El informe propone monitoreo y una nueva evaluación temporal antes de uso comercial.
