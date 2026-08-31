# Preparación de datos para modelamiento

## Objetivo

Dejar los datos listos para entrenar y evaluar modelos de regresión destinados a estimar `popularity`, evitando que información del conjunto de prueba influya en la preparación.

## Variable objetivo

La variable objetivo es `popularity`, un puntaje numérico entre 0 y 100. Por esta razón, el problema se mantiene como regresión.

## Predictores seleccionados

### Variables continuas escaladas

- `duration_ms`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`

Se utiliza `StandardScaler`, ajustado solamente con entrenamiento.

### Variables categóricas codificadas

- `track_genre`
- `key`
- `mode`
- `time_signature`

Se aplica One-Hot Encoding con soporte para categorías desconocidas en prueba.

### Variable binaria

- `explicit`: convertida de booleano a 0/1.

## Variables excluidas

- `track_id`
- `artists`
- `album_name`
- `track_name`

Son identificadores o textos de alta cardinalidad. Utilizarlos directamente podría hacer que un modelo memorice canciones o artistas, en lugar de aprender relaciones generalizables.

## Separación de datos

- Entrenamiento: 80%, equivalente a 71.596 registros.
- Prueba: 20%, equivalente a 17.899 registros.
- Semilla: 42.
- Estratificación: deciles auxiliares de la variable objetivo para mantener distribuciones semejantes.

La estratificación solo se utiliza para definir la división; `popularity` no entra al pipeline como predictor.

## Prevención de fuga de información

1. La división train/test ocurre antes de ajustar transformaciones.
2. `StandardScaler` aprende promedio y desviación únicamente desde entrenamiento.
3. One-Hot Encoding aprende categorías únicamente desde entrenamiento.
4. Prueba se transforma con el pipeline ya ajustado, sin volver a entrenarlo.
5. La variable objetivo y los identificadores no forman parte de los predictores.

## Archivos generados

- `data/processed/train.csv`: predictores originales seleccionados y objetivo para entrenamiento.
- `data/processed/test.csv`: predictores originales seleccionados y objetivo para evaluación final.
- `models/preprocessing_pipeline.joblib`: pipeline ajustado exclusivamente con entrenamiento.
- `reports/preparation/feature_names.csv`: nombres de las variables después de la transformación.
- `reports/preparation/split_summary.csv`: tamaños de cada conjunto.

Los CSV conservan las variables seleccionadas en formato legible. Durante el modelamiento deben transformarse usando el pipeline guardado. Esta estrategia evita almacenar matrices One-Hot muy grandes y mantiene una única definición reproducible del preprocesamiento.

## Reproducción

```bash
python src/features/prepare_features.py
```

También puede ejecutarse `notebooks/04_preparacion_datos.ipynb` en orden.
