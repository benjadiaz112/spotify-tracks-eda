# Spotify Tracks - Prediccion de popularidad

Este repositorio desarrolla un proyecto de Machine Learning para estudiar y estimar la popularidad de canciones de Spotify a partir de sus caracteristicas musicales.

## Problema de negocio

El proyecto busca responder si es posible estimar la popularidad de una cancion utilizando sus caracteristicas de audio y su genero musical. La definicion completa, los objetivos, KPIs, usuarios y herramientas se encuentran en [`docs/problema_negocio.md`](docs/problema_negocio.md).

## Objetivo de esta etapa

Conocer la estructura del conjunto de datos y revisar su calidad antes de preparar variables o entrenar modelos.

El notebook incluye:

- carga y vista inicial de los datos;
- dimensiones, columnas y tipos de datos;
- revision de valores nulos y duplicados;
- resumen estadistico de variables numericas;
- distribuciones simples de popularidad y duracion;
- conclusiones iniciales sobre calidad de datos.

## Estructura

```text
spotify-tracks-eda/
|-- data/
|   |-- raw/
|   |   `-- Spotify_Tracks_Dataset.csv
|   `-- processed/
|       `-- spotify_tracks_clean.csv
|-- docs/
|   |-- problema_negocio.md
|   |-- limpieza_datos.md
|   `-- resultados_eda.md
|-- images/
|   `-- eda/
|-- models/
|-- notebooks/
|   |-- 01_eda_inicial.ipynb
|   |-- 02_limpieza_datos.ipynb
|   `-- 03_eda_completo.ipynb
|-- reports/
|   |-- data_quality/
|   |   `-- cleaning_summary.csv
|   `-- eda/
|-- src/
|   |-- analysis/
|   |   `-- eda_complete.py
|   `-- data/
|       `-- clean_data.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Ejecucion

1. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Abrir `notebooks/01_eda_inicial.ipynb`.
3. Ejecutar los notebooks en orden.

La limpieza tambien puede ejecutarse directamente con:

```bash
python src/data/clean_data.py
```

El EDA completo y sus gráficos pueden regenerarse con:

```bash
python src/analysis/eda_complete.py
```

## Hallazgos iniciales

- El dataset contiene 114.000 filas y 21 columnas.
- Solo hay 3 valores nulos: uno en `artists`, uno en `album_name` y uno en `track_name`.
- No existen filas completamente duplicadas.
- Existen 24.259 IDs de canciones repetidos; esto debe revisarse antes de modelar.
- La columna `Unnamed: 0` funciona como indice exportado y no aporta informacion analitica.
- Se detectan duraciones iguales a cero y valores muy altos, que pueden ser anomalias.

## Alcance actual

El repositorio incluye la comprension del problema de negocio, limpieza reproducible y EDA completo. La preparación de variables y el modelamiento se desarrollarán en etapas posteriores.
