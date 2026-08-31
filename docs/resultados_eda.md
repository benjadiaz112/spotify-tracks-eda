# Resultados del análisis exploratorio completo

## Datos analizados

El análisis utiliza `data/processed/spotify_tracks_clean.csv`, que contiene 89.495 canciones y 20 variables después de aplicar las reglas de limpieza documentadas.

## Distribución de popularidad

- Promedio: 33,20.
- Mediana: 33.
- Desviación estándar: 20,59.
- Rango observado: 0 a 100.
- El 50% central de las canciones se encuentra entre 19 y 49 puntos.

La canción con mayor popularidad es `Unholy (feat. Kim Petras)`, de Sam Smith y Kim Petras, con 100 puntos. En el extremo inferior existen varias canciones con popularidad igual a cero.

## Popularidad por género

Los géneros con mayor mediana incluyen `pop`, `metal`, `k-pop`, `pop-film` y `hip-hop`. Los géneros `soul`, `iranian`, `romance`, `jazz`, `rock`, `latin` y `country` presentan medianas muy bajas en esta versión del dataset.

Estas diferencias deben interpretarse con cautela. Durante la limpieza se conservó un registro por `track_id`, por lo que una canción asociada a varios géneros conserva solamente el primero encontrado. Además, popularidad baja no significa menor calidad musical.

## Contenido explícito

| Tipo | Canciones | Promedio | Mediana |
|---|---:|---:|---:|
| No explícita | 81.793 | 32,86 | 33 |
| Explícita | 7.702 | 36,89 | 37 |

Las canciones explícitas presentan cerca de cuatro puntos adicionales de popularidad promedio. Este resultado es descriptivo: no permite afirmar que el contenido explícito provoque mayor popularidad, porque también pueden influir el género, el artista, la promoción o el público objetivo.

## Relación entre características y popularidad

| Variable | Correlación con popularidad |
|---|---:|
| `loudness` | 0,0725 |
| `danceability` | 0,0655 |
| `acousticness` | -0,0388 |
| `duration_ms` | -0,0210 |
| `energy` | 0,0140 |
| `valence` | -0,0116 |
| `tempo` | 0,0087 |

Ninguna correlación lineal individual supera 0,08. Esto indica que una sola característica musical no explica bien la popularidad. Un modelo futuro deberá considerar combinaciones de variables, posibles relaciones no lineales y la influencia de factores externos que no aparecen en el dataset.

## Valores extremos

La regla del rango intercuartil identifica:

- 4.926 valores extremos en `loudness` (5,50%).
- 4.159 valores extremos en `duration_ms` (4,65%).
- 368 valores extremos en `tempo` (0,41%).
- 346 valores extremos en `danceability` (0,39%).
- 11 valores extremos en `popularity` (0,01%).

No se eliminan automáticamente. Después de la limpieza inicial, estos valores pueden representar canciones reales con características poco habituales. Su tratamiento debe definirse de acuerdo con la sensibilidad del modelo.

## Variables relevantes

- `popularity`: variable objetivo del proyecto.
- `track_genre`: relevante por las diferencias observadas entre grupos, aunque requiere una codificación cuidadosa.
- `explicit`: muestra diferencias en promedio y mediana entre categorías.
- `loudness` y `danceability`: presentan las asociaciones lineales más altas con popularidad.
- `energy`, `acousticness`, `valence`, `tempo` y `duration_ms`: mantienen valor analítico porque podrían interactuar entre sí o aportar a modelos no lineales.

`track_id`, `track_name`, `artists` y `album_name` no deberían utilizarse directamente en un modelo inicial. Podrían producir memorización o introducir información difícil de aplicar a canciones nuevas.

## Conclusiones

1. La popularidad tiene una dispersión amplia y no se explica por una sola característica de audio.
2. El género y el contenido explícito permiten observar diferencias entre grupos, pero no establecen causalidad.
3. Volumen y bailabilidad son las características numéricas más asociadas con popularidad, aunque su relación es débil.
4. Los valores extremos restantes parecen plausibles y deben estudiarse durante el modelamiento, no eliminarse de forma automática.
5. La preparación de datos deberá codificar categorías, excluir identificadores y comparar modelos capaces de representar relaciones no lineales.

## Evidencias generadas

Los gráficos se encuentran en `images/eda/` y las tablas reproducibles en `reports/eda/`. Todo puede regenerarse ejecutando `python src/analysis/eda_complete.py` o el notebook `notebooks/03_eda_completo.ipynb`.
