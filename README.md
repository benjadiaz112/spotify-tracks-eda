# Spotify Tracks - Análisis y preparación para Machine Learning

Proyecto académico de Machine Learning orientado a estudiar las características de canciones de Spotify y preparar una solución capaz de estimar su popularidad.

## Estado del proyecto

El repositorio incluye:

- comprensión del problema de negocio;
- objetivos y KPIs;
- análisis exploratorio de datos;
- evaluación de calidad y limpieza;
- preparación reproducible para modelamiento;
- evaluación de sesgos, ética y privacidad.

El informe completo está disponible en [`docs/informe_tecnico.md`](docs/informe_tecnico.md).

## Problema de negocio

El proyecto responde la pregunta:

> ¿Es posible estimar la popularidad de una canción utilizando sus características de audio y su género musical?

La variable objetivo es `popularity`, un valor numérico entre 0 y 100. El problema se plantea como regresión.

## Fuente de datos

- Dataset: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- Autor en Kaggle: MaharshiPandya.
- Licencia declarada: `Database: Open Database, Contents: © Original Authors`.
- Archivo utilizado: `data/raw/Spotify_Tracks_Dataset.csv`.
- SHA-256 del archivo original: `B202FA49909B2D5CEF71A04B1D21243CFEB36414535F2CA9272AA646721177BD`.

La página de Kaggle describe un rango de 125 géneros, mientras la copia local contiene 114 antes de limpiar. Esto indica una posible diferencia de versión y se considera una limitación de trazabilidad.

## Resultados principales

- Datos originales: 114.000 filas y 21 columnas.
- Datos limpios: 89.495 canciones únicas y 20 columnas.
- Nulos después de limpiar: 0.
- Popularidad promedio: 33,20.
- Popularidad mediana: 33.
- Canciones explícitas: promedio 36,89; no explícitas: 32,86.
- Las correlaciones individuales con popularidad son débiles; las mayores corresponden a `loudness` y `danceability`.
- Entrenamiento: 71.596 registros.
- Prueba: 17.899 registros.
- Variables finales después de codificar: 142.

Los resultados describen asociaciones y no demuestran causalidad.

## Estructura

```text
spotify-tracks-eda/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|   |-- informe_tecnico.md
|   |-- problema_negocio.md
|   |-- limpieza_datos.md
|   |-- resultados_eda.md
|   |-- preparacion_datos.md
|   `-- etica_privacidad.md
|-- images/
|   `-- eda/
|-- models/
|   `-- preprocessing_pipeline.joblib
|-- notebooks/
|   |-- 01_eda_inicial.ipynb
|   |-- 02_limpieza_datos.ipynb
|   |-- 03_eda_completo.ipynb
|   `-- 04_preparacion_datos.ipynb
|-- reports/
|   |-- data_quality/
|   |-- eda/
|   `-- preparation/
|-- src/
|   |-- analysis/
|   |-- data/
|   `-- features/
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Instalación

Requisitos:

- Python 3.11 o superior.
- Git.

Crear un entorno virtual e instalar dependencias:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS o Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducción

Ejecutar los notebooks en orden:

1. `notebooks/01_eda_inicial.ipynb`
2. `notebooks/02_limpieza_datos.ipynb`
3. `notebooks/03_eda_completo.ipynb`
4. `notebooks/04_preparacion_datos.ipynb`

Las etapas principales también pueden ejecutarse como scripts:

```bash
python src/data/clean_data.py
python src/analysis/eda_complete.py
python src/features/prepare_features.py
```

Estos comandos generan nuevamente:

- `data/processed/spotify_tracks_clean.csv`;
- `data/processed/train.csv`;
- `data/processed/test.csv`;
- gráficos en `images/eda/`;
- reportes en `reports/`;
- `models/preprocessing_pipeline.joblib`.

## Metodología

El proyecto utiliza CRISP-DM:

1. Comprensión del negocio.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelamiento, planteado como etapa siguiente.
5. Evaluación, pendiente de un modelo entrenado.
6. Despliegue, fuera del alcance de esta entrega.

## Ética y uso responsable

La predicción no debe utilizarse para decidir automáticamente qué artistas reciben promoción o financiamiento. `popularity` no representa calidad musical y puede reflejar desigualdades previas de exposición. Los riesgos y controles se detallan en [`docs/etica_privacidad.md`](docs/etica_privacidad.md).

## Limitaciones

- La fecha de extracción no está documentada.
- La popularidad cambia con el tiempo.
- Faltan variables de promoción, seguidores y presencia en listas.
- La representación de artistas y géneros es desigual.
- La limpieza conserva un género por canción duplicada.
- La fuente describe más géneros que la copia local.
- Todavía no se ha entrenado ni evaluado un modelo predictivo.

## Próximos pasos

- Entrenar un modelo base y modelos de regresión.
- Comparar MAE y RMSE con los KPIs.
- Evaluar errores por género y representación del artista.
- Realizar validación temporal y con artistas no vistos.
- Preparar la presentación y defensa técnica.
