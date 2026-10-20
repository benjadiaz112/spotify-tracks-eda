# Limpieza y calidad de los datos

## Objetivo

Crear una version consistente del dataset para continuar el EDA y preparar el futuro modelamiento, sin modificar el archivo original.

## Reglas aplicadas

1. Se elimina `Unnamed: 0` porque representa un indice exportado.
2. Se elimina el registro sin `artists`, `album_name` y `track_name`.
3. Se conserva un registro por `track_id` para evitar que una misma cancion aparezca varias veces durante el modelamiento.
4. Se aceptan duraciones entre 30 segundos y 20 minutos, un rango razonable para analizar canciones.
5. Se eliminan registros con `tempo` igual o menor que cero o con compas fuera del rango 1 a 5.
6. Se valida que las caracteristicas normalizadas de audio esten entre 0 y 1.
7. Se valida que `popularity` permanezca entre 0 y 100.

## Resultados

| Indicador | Resultado |
|---|---:|
| Filas originales | 114.000 |
| IDs unicos antes de limpieza | 89.741 |
| Filas limpias | 89.495 |
| Columnas finales | 20 |
| Valores nulos finales | 0 |
| IDs repetidos finales | 0 |
| Retencion sobre IDs unicos | 99,73% |

La diferencia principal se debe a que el archivo original contiene 24.259 apariciones repetidas de `track_id`. Despues de trabajar con canciones unicas, solo 246 registros adicionales no cumplen las reglas de calidad.

## Archivos generados

- `data/processed/spotify_tracks_clean.csv`: dataset listo para continuar el analisis.
- `reports/data_quality/cleaning_summary.csv`: cantidad de filas afectadas por cada regla.
- `src/data/clean_data.py`: funciones y ejecucion reproducible de la limpieza.
- `notebooks/02_limpieza_datos.ipynb`: explicacion paso a paso.

## Limitacion

Una cancion puede estar asociada a mas de un genero. Al conservar un solo registro por `track_id`, se mantiene el primer genero encontrado. Esta decision evita duplicados y fuga de informacion, pero puede disminuir la representacion de algunos generos. La regla puede revisarse en la etapa de preparacion de datos si el genero resulta importante para el modelo.
