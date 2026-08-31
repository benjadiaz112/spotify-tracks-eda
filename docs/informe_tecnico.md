# Informe técnico - Spotify Tracks

## Resumen ejecutivo

Este proyecto aplica las primeras etapas de un proceso de Machine Learning al caso Spotify Tracks. El objetivo es estudiar si las características musicales y el género permiten estimar la popularidad de una canción.

Se analizaron 114.000 registros. Después de revisar nulos, duplicados y valores no válidos, se obtuvo un dataset de 89.495 canciones únicas. El EDA muestra una popularidad promedio de 33,20 y relaciones lineales débiles entre popularidad y las características numéricas. El género y el contenido explícito presentan diferencias descriptivas, pero no permiten establecer causalidad.

Los datos fueron separados en entrenamiento y prueba. Se creó un pipeline reproducible que escala variables continuas, codifica categorías y evita ajustar transformaciones con datos de prueba. También se documentaron riesgos de representación, privacidad y perjuicio para artistas emergentes.

## 1. Problema de negocio

Artistas, sellos y equipos de promoción necesitan priorizar recursos en un mercado con una gran cantidad de canciones. Las decisiones basadas únicamente en intuición pueden complementarse con evidencia obtenida desde datos.

Pregunta principal:

> ¿Es posible estimar la popularidad de una canción utilizando sus características de audio y su género musical?

La solución se plantea como regresión porque `popularity` es numérica y se encuentra entre 0 y 100.

Usuarios potenciales:

- artistas;
- sellos discográficos;
- equipos de marketing musical;
- curadores de listas de reproducción.

La predicción es una señal de apoyo y no debe reemplazar el criterio humano.

## 2. Objetivos

### Objetivo general

Preparar una solución reproducible de Machine Learning que permita estimar la popularidad de una canción a partir de sus características disponibles.

### Objetivos específicos

1. Comprender la estructura y calidad del dataset.
2. Identificar nulos, duplicados y anomalías.
3. Analizar patrones asociados con popularidad.
4. Identificar variables relevantes.
5. Preparar los datos sin fuga de información.
6. Evaluar sesgos, privacidad y limitaciones.

## 3. KPIs

| KPI | Meta | Estado actual |
|---|---:|---|
| Retención de registros utilizables sobre IDs únicos | Al menos 95% | 99,73%; cumplido. |
| Notebooks ejecutables en orden | 100% | Validación realizada; cumplido. |
| Variables relevantes documentadas | Al menos 5 | Más de 5; cumplido. |
| MAE del modelo | Menor o igual a 15 | Pendiente de modelamiento. |
| Mejora de MAE frente al promedio | Al menos 10% | Pendiente de modelamiento. |

## 4. Fuente y licencia

La copia utilizada corresponde a:

- Nombre: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
- Responsable en Kaggle: MaharshiPandya.
- Formato: CSV.
- Licencia declarada en Kaggle: `Database: Open Database, Contents: © Original Authors`.
- Archivo local: `data/raw/Spotify_Tracks_Dataset.csv`.
- SHA-256: `B202FA49909B2D5CEF71A04B1D21243CFEB36414535F2CA9272AA646721177BD`.

La descripción de Kaggle explica que el puntaje de popularidad se basa principalmente en reproducciones totales y recientes. El valor puede cambiar con el tiempo.

### Observación de versión

La página del dataset menciona 125 géneros. El archivo local tiene 114 antes de limpiar y 113 después de aplicar las reglas de calidad. No se dispone de una fecha o número de versión dentro del ZIP descargado. Para reproducibilidad se conserva el archivo y su hash.

La licencia debe revisarse antes de redistribuir los datos fuera del contexto académico. Los derechos sobre nombres, metadatos y contenidos originales continúan perteneciendo a sus respectivos titulares.

## 5. Herramientas

| Herramienta | Uso |
|---|---|
| Python | Desarrollo y automatización. |
| pandas | Carga, limpieza y análisis tabular. |
| Matplotlib y seaborn | Visualizaciones. |
| scikit-learn | División, escalamiento y codificación. |
| Jupyter Notebook | Explicación reproducible. |
| Git y GitHub | Control de versiones y colaboración. |

## 6. Metodología CRISP-DM

### 6.1 Comprensión del negocio

Se definieron el problema, usuarios, objetivos y KPIs. Se estableció que popularidad es una señal comercial limitada y no una medición de calidad musical.

Evidencia: `docs/problema_negocio.md`.

### 6.2 Comprensión de los datos

Se revisaron dimensiones, tipos, nulos, duplicados, estadísticas, distribuciones, relaciones y valores extremos.

Evidencias:

- `notebooks/01_eda_inicial.ipynb`;
- `notebooks/03_eda_completo.ipynb`;
- `docs/resultados_eda.md`;
- `images/eda/`.

### 6.3 Preparación de los datos

Se eliminaron columnas sin valor analítico, nulos esenciales, IDs duplicados y registros con rangos no válidos. Luego se separaron predictores y objetivo, se excluyeron identificadores y se creó un pipeline de transformación.

Evidencias:

- `notebooks/02_limpieza_datos.ipynb`;
- `notebooks/04_preparacion_datos.ipynb`;
- `src/data/clean_data.py`;
- `src/features/prepare_features.py`.

### 6.4 Modelamiento

No forma parte de la implementación actual. Los conjuntos y el pipeline quedaron preparados para comparar un modelo base, regresión lineal y modelos no lineales.

### 6.5 Evaluación

La evaluación predictiva está pendiente. Deberá incluir MAE y RMSE globales y por grupos relevantes, especialmente género y representación del artista.

### 6.6 Despliegue

Está fuera del alcance de esta entrega. Antes de un uso real se necesitan validación temporal, monitoreo, control de acceso y revisión humana.

## 7. Calidad y limpieza

### Datos originales

- 114.000 filas.
- 21 columnas.
- Tres valores nulos, concentrados en un registro.
- 24.259 apariciones repetidas de `track_id`.
- Una columna de índice exportado: `Unnamed: 0`.

### Reglas aplicadas

1. Eliminar `Unnamed: 0`.
2. Eliminar el registro sin artista, álbum o canción.
3. Conservar una fila por `track_id`.
4. Conservar duraciones entre 30 segundos y 20 minutos.
5. Excluir tempo o compás no válidos.
6. Validar características normalizadas entre 0 y 1.
7. Verificar popularidad entre 0 y 100.

### Resultado

- 89.495 canciones.
- 20 columnas.
- 0 nulos.
- 0 IDs repetidos.
- Retención de 99,73% respecto a los IDs únicos originales.

Archivo: `data/processed/spotify_tracks_clean.csv`.

## 8. Resultados del EDA

### Popularidad

- Promedio: 33,20.
- Mediana: 33.
- Desviación estándar: 20,59.
- Rango: 0 a 100.
- El 50% central se encuentra entre 19 y 49.

### Género

Los géneros presentan diferencias en mediana y promedio. `pop`, `metal`, `k-pop`, `pop-film` y `hip-hop` se ubican entre las medianas más altas en la copia limpia. El resultado está condicionado por la regla que conserva un solo género por `track_id`.

### Contenido explícito

- No explícitas: promedio 32,86; mediana 33.
- Explícitas: promedio 36,89; mediana 37.

La diferencia es descriptiva. Puede estar confundida por género, mercado, artista o promoción.

### Relaciones numéricas

| Variable | Correlación con popularidad |
|---|---:|
| `loudness` | 0,0725 |
| `danceability` | 0,0655 |
| `acousticness` | -0,0388 |
| `duration_ms` | -0,0210 |
| `energy` | 0,0140 |
| `valence` | -0,0116 |
| `tempo` | 0,0087 |

Las correlaciones son débiles. La popularidad probablemente depende de combinaciones no lineales y de factores comerciales o culturales ausentes.

### Valores extremos

La regla IQR detectó principalmente valores extremos en `loudness` y duración. No fueron eliminados automáticamente porque pueden representar canciones válidas.

## 9. Preparación para modelamiento

### Objetivo

- `popularity`.

### Variables continuas escaladas

- duración;
- bailabilidad;
- energía;
- volumen;
- presencia de habla;
- acústica;
- instrumentalidad;
- sonido en vivo;
- valencia;
- tempo.

### Variables categóricas codificadas

- género;
- tonalidad;
- modo;
- compás.

`explicit` se transforma a 0/1.

Se excluyen `track_id`, artista, álbum y nombre de canción para reducir memorización y fuga indirecta.

### División

- Entrenamiento: 71.596 filas.
- Prueba: 17.899 filas.
- Semilla: 42.
- Variables transformadas: 142.

El pipeline se ajusta únicamente con entrenamiento y se guarda en `models/preprocessing_pipeline.joblib`.

## 10. Ética y privacidad

Riesgos principales:

- desbalance entre géneros;
- mayor representación de artistas recurrentes;
- perjuicio para artistas emergentes;
- retroalimentación entre predicción y promoción;
- interpretación incorrecta de popularidad como calidad;
- uso reputacional de nombres;
- vinculación de IDs con otras fuentes;
- cambio de tendencias con el tiempo.

Medidas:

- medir errores por género y frecuencia del artista;
- realizar validación con artistas no vistos;
- evitar decisiones automáticas de financiamiento o exclusión;
- mantener revisión humana;
- reservar oportunidades para diversidad y exploración;
- minimizar identificadores;
- monitorear cambios y actualizar el modelo.

Detalle completo: `docs/etica_privacidad.md`.

## 11. Conclusiones generales

1. La calidad del dataset permite continuar con modelamiento después de una limpieza documentada.
2. Ninguna característica de audio individual explica bien la popularidad.
3. Género y contenido explícito muestran diferencias, pero no causalidad.
4. Los datos están listos para comparar modelos mediante un pipeline reproducible.
5. Las métricas globales deberán complementarse con evaluación por grupos.
6. La predicción debe usarse como apoyo y no como decisión automática.

## 12. Limitaciones

- No existe fecha exacta de extracción.
- La popularidad es dinámica y dependiente de la plataforma.
- No hay variables sobre promoción, seguidores, listas o país.
- Artistas y géneros tienen representación desigual.
- La regla de deduplicación conserva un género por canción.
- La cantidad de géneros no coincide con la descripción actual de Kaggle.
- No se ha entrenado un modelo predictivo.
- Los resultados no pueden interpretarse como relaciones causales.

## 13. Reproducción

### Instalar dependencias

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS o Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecutar scripts

Desde la raíz del proyecto:

```bash
python src/data/clean_data.py
python src/analysis/eda_complete.py
python src/features/prepare_features.py
```

### Ejecutar notebooks

Abrir Jupyter y ejecutar en orden:

```bash
jupyter notebook
```

1. `01_eda_inicial.ipynb`.
2. `02_limpieza_datos.ipynb`.
3. `03_eda_completo.ipynb`.
4. `04_preparacion_datos.ipynb`.

### Resultados esperados

- dataset limpio;
- conjuntos train/test;
- pipeline de preparación;
- seis gráficos principales;
- tablas de calidad, EDA y preparación.

## 14. Próximos pasos

1. Crear una línea base con la media de entrenamiento.
2. Entrenar regresión lineal y modelos no lineales.
3. Comparar MAE y RMSE.
4. Revisar métricas por grupo.
5. Validar con artistas no vistos y datos de otro periodo.
6. Preparar presentación y defensa técnica.
