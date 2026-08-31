# Definicion del problema de negocio

## 1. Contexto

Spotify publica una gran cantidad de canciones de distintos artistas y generos. Para artistas, sellos discograficos y equipos de marketing puede ser dificil estimar que tan popular sera una cancion y decidir donde concentrar sus esfuerzos de promocion.

El conjunto de datos disponible contiene informacion de 114.000 registros, como genero musical, duracion, energia, bailabilidad, volumen, tempo y popularidad. Estas variables permiten estudiar si las caracteristicas de una cancion se relacionan con su nivel de popularidad.

## 2. Problema de negocio

Los equipos que promocionan musica necesitan una forma objetiva de estimar la popularidad de una cancion a partir de sus caracteristicas musicales. Sin esta informacion, las decisiones pueden depender solamente de la intuicion y no de evidencia obtenida desde los datos.

El proyecto busca responder la siguiente pregunta:

> ¿Es posible estimar la popularidad de una cancion utilizando sus caracteristicas de audio y su genero musical?

La solucion se plantea inicialmente como un problema de **regresion**, porque `popularity` es una variable numerica entre 0 y 100.

## 3. Objetivo general

Desarrollar una solucion reproducible de Machine Learning que permita estimar la popularidad de una cancion a partir de sus caracteristicas disponibles en el dataset de Spotify.

## 4. Objetivos especificos

1. Comprender la estructura y calidad del conjunto de datos.
2. Identificar valores nulos, duplicados y posibles anomalias.
3. Analizar la relacion entre popularidad y las caracteristicas musicales.
4. Preparar los datos para que puedan ser utilizados por un modelo de regresion.
5. Entrenar y evaluar un modelo base en una etapa posterior.
6. Reconocer posibles sesgos y limitaciones del uso de los datos.

## 5. Usuarios de la solucion

- Artistas que quieran conocer el potencial de sus canciones.
- Sellos discograficos que necesiten priorizar campañas de promocion.
- Equipos de marketing musical.
- Curadores de listas de reproduccion.

La prediccion debe utilizarse como apoyo para tomar decisiones y no como reemplazo del criterio humano.

## 6. Variable objetivo y variables predictoras

### Variable objetivo

- `popularity`: puntaje numerico entre 0 y 100.

### Posibles variables predictoras

- `duration_ms`
- `explicit`
- `danceability`
- `energy`
- `key`
- `loudness`
- `mode`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `time_signature`
- `track_genre`

Los identificadores y nombres, como `track_id`, `track_name`, `artists` y `album_name`, deben revisarse antes de modelar para evitar que el modelo memorice canciones concretas en vez de aprender patrones generales.

## 7. KPIs del proyecto

| KPI | Forma de medicion | Meta inicial |
|---|---|---:|
| Calidad de datos | Registros utilizables / registros totales | Al menos 95% |
| Error absoluto medio (MAE) | Promedio del error absoluto en puntos de popularidad | Menor o igual a 15 |
| Mejora frente al modelo base | Reduccion del MAE respecto a predecir siempre el promedio | Al menos 10% |
| Reproducibilidad | Notebooks ejecutables en orden sin errores | 100% |
| Variables relevantes documentadas | Variables analizadas y justificadas | Al menos 5 |

Las metas de rendimiento son iniciales y deberan revisarse cuando se construya el modelo base.

## 8. Fuente de datos

Se utiliza el archivo `Spotify_Tracks_Dataset.csv`, incluido en `data/raw/`. Contiene 114.000 registros y 21 columnas relacionadas con canciones y caracteristicas de audio.

El archivo fue proporcionado para el caso Spotify Tracks. La direccion exacta de descarga original debe incorporarse al informe cuando sea confirmada, para asegurar la trazabilidad de la fuente y su licencia de uso.

## 9. Herramientas utilizadas

| Herramienta | Uso en el proyecto | Justificacion |
|---|---|---|
| Python | Desarrollo del analisis | Es adecuado para analisis de datos y Machine Learning. |
| pandas | Carga, exploracion y manipulacion | Permite trabajar de forma simple con datos tabulares. |
| Matplotlib y seaborn | Graficos | Facilitan la identificacion visual de patrones y anomalias. |
| Jupyter Notebook | Desarrollo reproducible | Integra explicaciones, codigo y resultados. |
| Git | Control de versiones | Registra los cambios realizados en cada etapa. |
| GitHub | Colaboracion y respaldo | Permite compartir el proyecto y revisar su historial. |

## 10. Relacion con CRISP-DM

Esta etapa corresponde principalmente a **comprension del negocio**. El EDA inicial ya comienza la fase de **comprension de los datos**. Las siguientes ramas abordaran preparacion de datos, analisis completo y modelamiento.

## 11. Criterio de exito

El proyecto sera considerado exitoso si los datos pueden prepararse de manera reproducible y si el futuro modelo mejora de forma medible frente a una prediccion base sencilla. Los resultados deberan explicarse junto con sus limitaciones y no presentarse como una garantia de exito comercial.
