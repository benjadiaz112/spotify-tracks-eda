# Spotify Tracks - EDA inicial

Este repositorio contiene la primera parte del analisis exploratorio de datos (EDA) para el caso **Spotify Tracks** de la asignatura Machine Learning.

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
|   `-- raw/
|       `-- Spotify_Tracks_Dataset.csv
|-- notebooks/
|   `-- 01_eda_inicial.ipynb
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
3. Ejecutar las celdas en orden.

## Hallazgos iniciales

- El dataset contiene 114.000 filas y 21 columnas.
- Solo hay 3 valores nulos: uno en `artists`, uno en `album_name` y uno en `track_name`.
- No existen filas completamente duplicadas.
- Existen 24.259 IDs de canciones repetidos; esto debe revisarse antes de modelar.
- La columna `Unnamed: 0` funciona como indice exportado y no aporta informacion analitica.
- Se detectan duraciones iguales a cero y valores muy altos, que pueden ser anomalias.

## Alcance

Esta entrega cubre solamente el EDA inicial. La limpieza avanzada, transformacion de variables y modelamiento se desarrollaran en etapas posteriores.
