"""Documentos de la evaluación generados desde los resultados, sin cifras manuales."""
import json
from pathlib import Path

import pandas as pd

from .data import ROOT


def md_table(df, digits=3):
    """Tabla Markdown sin dependencia opcional tabulate."""
    df = df.copy()
    for col in df.select_dtypes(include="number"):
        integer = pd.api.types.is_integer_dtype(df[col].dtype)
        df[col] = df[col].map(lambda x: (str(int(x)) if integer else f"{x:.{digits}f}") if pd.notna(x) else "N/A")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines.extend("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False, name=None))
    return "\n".join(lines)


def build_reports(root: Path = ROOT):
    out = root / "reports/eva2"
    def read(name):
        return json.loads((out / name).read_text(encoding="utf-8"))
    qa, ev, cl, selection = [read(f) for f in ["data_audit.json", "evaluation.json", "clustering.json", "selection.json"]]
    results = pd.read_csv(out / "supervised_metrics.csv")
    sensitivity = pd.read_csv(out / "sensitivity_genre.csv")
    target_audit = read("target_audit.json")
    label_sensitivity = read("label_sensitivity.json")
    label_folds = pd.read_csv(out / "label_sensitivity_folds.csv")
    profiles = pd.read_csv(out / "cluster_profiles_train.csv")
    elbow_profiles = pd.read_csv(out / "cluster_profiles_elbow_train.csv")
    stability = pd.read_csv(out / "cluster_stability.csv")
    bands = pd.read_csv(out / "errors_by_target_band.csv")
    genre_errors = pd.read_csv(out / "errors_by_track_genre.csv").query("n >= 30").sort_values("mae", ascending=False)
    explicit = pd.read_csv(out / "errors_by_explicit.csv")
    artists = pd.read_csv(out / "errors_by_artist_seen.csv")
    imp = pd.read_csv(out / "permutation_importance.csv")
    metrics = ev["metrics"]
    best = results.iloc[0]
    base = results.set_index("model").loc["Dummy_media"]
    genre_baseline = sensitivity.iloc[0]
    gain_genre = 100 * (1 - metrics["mae"] / genre_baseline.mae)
    p0, p1 = profiles.iloc[0], profiles.iloc[1]
    ci = ev["confidence"]["mae_95_ci"]
    executive = f"""# Informe ejecutivo - Inteligencia musical en Spotify

**Evaluación Parcial N°2, MLY1101. Caso C. Continuación de experiencia 1.**

Equipo: fluzxks, benjadiaz112 y Freymuth Christopher Isaac.

## Decisión que apoya el proyecto

Un equipo de curaduría necesita explorar su catálogo y estimar la popularidad observada de canciones mediante sus características de audio y género. Proponemos una herramienta de consulta local con revisión humana. El dataset es una instantánea sin fecha de extracción confirmada: el proyecto no demuestra que pueda anticipar futuros éxitos.

## Evidencia principal

El CSV recibido contiene {qa['raw_rows']:,} filas y {qa['raw_unique_tracks']:,} IDs únicos. La limpieza de experiencia 1 deja {qa['clean_rows']:,} canciones, retiene {qa['unique_retention_pct']:.2f}% de los IDs originales y no deja nulos. Detectamos además {qa['repeated_audio_rows']:,} filas con audio idéntico a otras de diferente ID. Para evitar contaminación entre entrenamiento y prueba, estas versiones permanecen en el mismo conjunto.

Reservamos {qa['test_rows']:,} canciones ({qa['test_pct']:.2f}%) para prueba y seleccionamos el modelo mediante validación cruzada agrupada de tres folds en las {qa['train_rows']:,} restantes. Comparamos dos referencias constantes y tres modelos supervisados: regresión lineal, árbol de decisión y HistGradientBoosting.

El modelo seleccionado, **{ev['selected_model']}**, obtiene **MAE {metrics['mae']:.2f} puntos**, RMSE {metrics['rmse']:.2f} y R² {metrics['r2']:.3f} en prueba. Reduce el MAE en **{ev['improvement_vs_mean_pct']:.2f}%** frente a la media constante ({base.mae:.2f}). Cumple las metas previas MAE <= 15 y mejora >= 10%. El IC bootstrap agrupado del MAE es [{ci[0]:.2f}, {ci[1]:.2f}], condicionado a esta prueba y a este modelo.

La comparación más exigente usa la media de popularidad por género calculada solo en entrenamiento: MAE {genre_baseline.mae:.2f}. Frente a ella, la mejora del modelo completo es **{gain_genre:.2f}%**. La ablación que retira únicamente género, conservando las otras catorce variables e hiperparámetros, registra MAE {sensitivity.iloc[1].mae:.2f}. Esta evidencia indica una dependencia importante del género, sin demostrar un efecto causal del sonido.

La auditoría encontró {target_audit['conflicting_ids_retained']} IDs con distintas popularidades entre filas originales ({target_audit['conflicting_ids_pct']:.2f}% de IDs); la diferencia mediana es {target_audit['median_range']:.0f} punto y la máxima {target_audit['maximum_range']:.0f}. Se mantiene la primera observación como referencia reproducible por continuidad y falta de fechas autoritativas. Entrenar con mediana cambia el MAE de CV de {label_sensitivity['cv_mae_first_training']:.3f} a {label_sensitivity['cv_mae_median_training']:.3f}, sin modificar etiquetas de test ni la selección. La incertidumbre de esas etiquetas sigue declarada.

## Segmentación para curaduría

K-Means usa diez variables continuas de audio escaladas, excluye popularidad y género, y compara K=1..10. Se elige K={cl['selected_k']} por la mayor silueta entre soluciones con grupos de al menos 3% del entrenamiento. El codo orientativo sugiere K={cl['elbow_chord_k']}, así que no existe un único óptimo respaldado por ambos criterios.

- Grupo 0: {p0.pct:.1f}% del entrenamiento, energía {p0.energy:.2f}, acústica {p0.acousticness:.2f} y tempo {p0.tempo:.1f}. Perfil relativamente energético para explorar selecciones de actividad.
- Grupo 1: {p1.pct:.1f}%, energía {p1.energy:.2f}, acústica {p1.acousticness:.2f} y tempo {p1.tempo:.1f}. Perfil relativamente acústico para explorar selecciones tranquilas.

Son nombres interpretativos, no etiquetas musicales verificadas. La silueta de prueba es {cl['silhouette_test_sample']:.3f}, compatible con separación limitada. Las semillas alternativas dan ARI entre {stability.ari_vs_42.min():.3f} y {stability.ari_vs_42.max():.3f}. PCA en dos dimensiones conserva {cl['pca_two_components_variance']:.1%} de varianza: sirve como mapa parcial, no como demostración de grupos perfectamente separados.

## Recomendación y límites

Recomendamos un piloto de exploración de catálogo con evaluación humana. No automatizar contratos ni presupuestos de promoción. El MAE para canciones de popularidad 81-100 es {bands.iloc[-1].mae:.2f} en solo {int(bands.iloc[-1].n)} casos: el modelo subestima los extremos. El percentil 95 del error absoluto es {ev['residual_p95_absolute']:.2f} puntos.

No se puede calcular ROI con estos datos: faltan costos, resultados de campañas y una evaluación causal. El [piloto propuesto](plan_piloto_negocio.md), aún no ejecutado, fija dos semanas y tres metas sujetas a aprobación: reducción mediana del tiempo >=15%, aceptación de recomendaciones >=70% y pérdida mediana de cobertura de géneros no mayor a 5 puntos porcentuales. Define fórmulas, responsables, doce pares de sesiones exploratorios y una plantilla vacía con calculadora validada. Antes de uso real se requiere una muestra reciente, evaluación temporal, revisión de la etiqueta género y validación de derechos de uso.

El desarrollo completo, fórmulas, decisiones, riesgos y matriz de rúbrica están en [informe técnico](informe_tecnico_eva2.md). El [notebook integral](../notebooks/05_proyecto_completo_eva2.ipynb) reproduce las etapas desde el CSV original.
"""
    report = f"""# Informe técnico - Spotify Tracks, Evaluación Parcial N°2

**MLY1101 - Machine Learning. Caso C.** Equipo: fluzxks, benjadiaz112 y Freymuth Christopher Isaac.

Este documento continúa la experiencia 1. Sus cifras se generan desde los CSV/JSON de `reports/eva2/`. Los antecedentes históricos permanecen en `docs/` y notebooks 01-04; este informe y el notebook 05 son la entrega vigente. La síntesis para decisión está en [informe ejecutivo](informe_ejecutivo.md).

## 1. Comprensión del negocio

### Problema, usuario y objetivo

Un equipo de curaduría o análisis musical necesita estudiar las características de su catálogo y contar con una estimación de popularidad observada. La pregunta predictiva es: **¿cuánta señal aportan el audio y el género para estimar `popularity` y qué perfiles musicales se pueden descubrir?** El objetivo general es desarrollar, comparar y evaluar una solución reproducible de regresión, complementada con segmentación, para apoyar una revisión humana del catálogo.

Objetivos específicos: auditar el dataset y sus versiones repetidas; preparar variables sin fuga; comparar al menos dos algoritmos supervisados; elegir K con evidencia; explicar errores y sesgos; entregar código, notebook y documentación reproducibles. El uso comercial es un escenario propuesto para el caso académico, no una necesidad de una empresa validada mediante entrevistas.

**Alcance:** estimar una puntuación de 0 a 100 en una instantánea y agrupar canciones por características acústicas. No se predicen reproducciones, ingresos, calidad musical ni popularidad futura. Sin fecha de lanzamiento o de captura no se puede construir una prueba temporal. No hay una intervención real para demostrar impacto comercial.

### KPIs previos y criterios de aceptación

| Indicador | Definición y meta previa | Resultado medido | Lectura |
|---|---|---|---|
| MAE de prueba | media de errores absolutos, <=15 puntos | {metrics['mae']:.3f} | Meta técnica cumplida |
| Mejora vs. media | 100 x (1 - MAE_modelo / MAE_media), >=10% | {ev['improvement_vs_mean_pct']:.2f}% | Meta técnica cumplida |
| Comparación exigente | Mejora vs. media por género de train, descriptiva sin meta previa | {gain_genre:.2f}% | Valor incremental más acotado |
| Retención de filas brutas | Meta histórica >=95% de experiencia 1 | {qa['raw_retention_pct']:.2f}% | No cumple esa definición histórica |
| Retención de IDs únicos | Registros limpios / IDs originales, diagnóstico adicional | {qa['unique_retention_pct']:.2f}% | Distingue duplicación de descarte por calidad |
| Integridad de partición | Cero IDs y grupos de audio compartidos | {qa['shared_tracks']} IDs, {qa['shared_audio_groups']} grupos | Control contra fuga exacta |
| Reproducibilidad | Notebook integral ejecutable y pruebas de contratos | Ver acta de verificación y notebook con salidas | Evidencia técnica, no impacto de negocio |

Los documentos de experiencia 1 usan denominadores distintos: `problema_negocio.md` formula calidad sobre registros totales, mientras `informe_tecnico.md` reporta retención de IDs únicos. No son indicadores equivalentes. Esta entrega muestra ambos: 78,50% sobre filas está por debajo de 95% bajo esa definición, y 99,73% sobre IDs supera 95% bajo la otra. Las filas brutas incluyen duplicaciones por género y no equivalen a canciones distintas. No se cambia el denominador después de evaluar para simular éxito ni se atribuye la nota anterior a esta diferencia.

### KPIs de negocio y reglas del piloto propuesto

**Estado: no ejecutado.** Estas metas son propuestas de diseño para aprobación del responsable de curaduría, no resultados ni umbrales exigidos por el docente. El [plan del piloto](plan_piloto_negocio.md) define dos semanas, dos revisores y doce pares de sesiones exploratorios. Cada sesión revisaría 100 canciones y seleccionaría 20 de lotes distintos y comparables, con asignación aleatoria y orden contrabalanceado. El tamaño es propuesto, sin cálculo de potencia ni garantía de significancia.

| KPI | Definición operativa | Meta propuesta a dos semanas | Responsable propuesto |
|---|---|---|---|
| Tiempo de curaduría | Minutos por 100 = segundos / 60 / revisadas x 100; reducción por par = 100 x (1 - tiempo_asistido / tiempo_manual) | Mediana de reducciones >=15% | Coordinador de curaduría registra; datos calcula |
| Aceptación | 100 x suma(aceptadas) / suma(recomendaciones) en sesiones asistidas | >=70% | Revisores registran; coordinador revisa |
| Cobertura de géneros | 100 x géneros distintos seleccionados / mínimo(géneros disponibles, seleccionadas); delta asistida - manual | Mediana del delta >= -5 pp | Datos audita; curaduría interpreta |

Continuar el piloto exige las tres metas y revisión de calidad/derechos. Si solo mejora el tiempo, ajustar y repetir; si no hay utilidad o aparecen daños, detener. Registrar fallas y abandonos, sin ocultar sesiones desfavorables. La cobertura usa una etiqueta incompleta y no certifica diversidad cultural ni equidad.

`data/examples/pilot_template.csv` contiene solo encabezados. `src/eva2/pilot.py` valida pares y calcula métricas cuando haya observaciones reales; rechaza la plantilla vacía. Sus pruebas usan ejemplos sintéticos identificados como tales. No hay base para afirmar ahorro comercial ni ROI.

## 2. Metodología CRISP-DM y gestión del trabajo

| Fase | Aplicación concreta | Evidencia |
|---|---|---|
| Comprensión del negocio | Delimitar regresión, usuarios, KPIs y decisiones humanas | Sección 1 y plan de evaluación |
| Comprensión de datos | Identidad del archivo, esquema, nulos, duplicados, distribuciones | `data_audit.json`, `schema.csv`, EDA |
| Preparación | Limpieza histórica explícita, grupos acústicos, partición, pipelines por fold | `src/eva2/data.py`, manifiesto y pruebas |
| Modelamiento | Referencias, lineal, árbol, boosting, K-Means y PCA | CV completa, parámetros, modelos persistidos |
| Evaluación | Test reservado, residuos, bootstrap y subgrupos | Secciones 6-8, CSV de predicciones |
| Despliegue | Inferencia local ejecutable, contrato y plan de monitoreo | CLI de predicción y sección 10 |

CRISP-DM es iterativo. El hallazgo de audio repetido llevó a revisar la partición propuesta en experiencia 1 antes de entrenar. La auditoría por género se añadió para interpretar el resultado después de congelar la elección del modelo, sin usarla para reajustar parámetros. El [plan](plan_evaluacion_2.md) define las tareas y sus criterios de aceptación. Los parámetros, particiones y resultados calculados quedan persistidos para reproducir y revisar cada etapa.

## 3. Fuentes y comprensión de los datos

Fuente inmediata confirmada: CSV dentro del ZIP de la experiencia 1 entregado por el equipo, sin descargar datos alternativos. Fuente pública declarada en experiencia 1: [Spotify Tracks Dataset, MaharshiPandya](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). La licencia anotada allí no pudo revalidarse durante esta consulta. Consultar [fuentes y trazabilidad](fuentes_eva2.md) antes de redistribuir públicamente.

- Archivo original: `data/raw/Spotify_Tracks_Dataset.csv`.
- SHA-256 recibido y calculado: `{qa['sha256']}`.
- {qa['raw_rows']:,} filas, {qa['raw_columns']} columnas y {qa['raw_unique_tracks']:,} IDs diferentes.
- {qa['raw_missing_cells']} celdas nulas, correspondientes al registro sin artista, álbum y nombre.
- {qa['raw_genres']} géneros antes de limpiar y {qa['clean_genres']} después.
- Fecha de captura, país, método de muestreo y población musical completa: no confirmados.

El README antiguo anotaba el hash b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd. Se comprobó que es el hash del CSV al normalizar CRLF a LF, tal como hizo Git en la importación inicial. No hay evidencia de cambios en los datos por esta diferencia. Esta entrega preserva los bytes CRLF recibidos mediante `.gitattributes`, verifica su hash y rechaza un CSV distinto hasta revisar su procedencia. La cantidad observada de géneros corresponde a esta copia y no se extrapola a otras versiones.

### Diccionario de variables utilizadas

| Variable | Tipo y significado operativo | Tratamiento |
|---|---|---|
| popularity | Objetivo numérico observado de 0 a 100 | Exclusivo de y, nunca predictor |
| duration_ms | Duración en milisegundos | Continua, dominio 30.000 a 1.200.000 por criterio heredado |
| danceability, energy | Indicadores de bailabilidad y energía, 0 a 1 | Continuos |
| loudness | Indicador de sonoridad en dB | Continuo, valores negativos admisibles |
| speechiness, acousticness | Indicadores de voz hablada y carácter acústico, 0 a 1 | Continuos |
| instrumentalness, liveness | Indicadores de instrumentalidad y presencia de ambiente en vivo, 0 a 1 | Continuos |
| valence | Indicador de carácter positivo del audio, 0 a 1 | No equivale al estado emocional de una persona |
| tempo | Tempo estimado | Continuo positivo |
| track_genre | Género asignado en el archivo | Categoría nominal, una etiqueta por ID conservado |
| key, mode, time_signature | Tonalidad codificada, modo y compás | Categorías, no magnitudes ordinales |
| explicit | Bandera binaria de contenido explícito | 0/1 |
| track_id, artists, album_name, track_name | Identificación y metadatos textuales | Excluidos del modelo; ID para trazabilidad, artista solo para auditoría |
| Unnamed: 0 | Índice exportado | Eliminado |

Son características ya presentes en el CSV; no se descargó ni procesó audio. El registro de tipos, faltantes y cardinalidad está en `schema.csv`.

## 4. Preparación y EDA

### Limpieza y continuidad con experiencia 1

Se reutiliza `src/data/clean_data.py`: retirar índice, eliminar la fila sin metadatos, conservar primer `track_id`, aplicar duración 30 s-20 min y exigir tempo positivo, compás 1-5 y características normalizadas válidas. Los límites de duración son una decisión de alcance heredada, no una ley sobre música válida. Por ello pueden excluir interludios y piezas largas y sesgar la población estudiada.

{md_table(pd.read_csv(out / 'cleaning.csv'), 0)}

Resultado: {qa['clean_rows']:,} canciones con IDs únicos y cero nulos. No se eliminan canciones por popularidad extrema ni por un boxplot. Un valor extremo puede ser válido. Los imputadores dentro de cada pipeline son una defensa de la transformación; en los datos auditados no sustituyeron valores. La CLI rechaza faltantes en nuevas entradas para no esconder mala calidad.

La deduplicación conserva el primer género del archivo, que está dispuesto en bloques por género. No es necesariamente el género principal del artista. Hay {target_audit['multi_genre_ids']:,} IDs originales asociados a varios géneros. La regla puede sesgar su atribución. Se conserva para continuidad y se declara como limitación central. Una futura versión deberá tratar género como multietiqueta o construir una regla validada con curadores antes de evaluar otro holdout.

### Auditoría específica de la variable objetivo

Se agrupan las filas originales por `track_id` y se cuentan etiquetas distintas, primera observación, mediana, mínimo, máximo y rango. Hay **{target_audit['conflicting_ids_retained']} IDs retenidos con popularidad contradictoria**, equivalentes a {target_audit['conflicting_ids_pct']:.3f}% de IDs originales. El rango mediano es {target_audit['median_range']:.0f} punto y el máximo {target_audit['maximum_range']:.0f}; {target_audit['range_greater_than_20']} IDs superan 20 puntos de diferencia. Primera observación y mediana difieren en {target_audit['first_differs_from_median_retained']} canciones limpias. Detalle: `target_conflicts.csv`; resumen y criterio: `target_audit.json`.

**Política:** mantener la primera fila por continuidad con experiencia 1 y como definición reproducible de la referencia, sin afirmar que sea la popularidad verdadera en una fecha concreta. Sin fecha autoritativa no se puede justificar que la última fila, el máximo o la mediana sean la verdad. No se cambia el target del test después de haber visto resultados ni se elige una política por mejorar ese score.

Se añadió una sensibilidad limitada al entrenamiento original: mismo HGB y parámetros congelados, mismos tres folds agrupados, cambiar únicamente las etiquetas usadas para ajustar por la mediana de filas del mismo ID. Cada validación mantiene la referencia original para que la comparación mida la misma tarea. Los IDs de entrenamiento y validación permanecen separados; no se incluyen IDs de test en ajustes ni decisiones.

{md_table(label_folds)}

El MAE medio de CV es {label_sensitivity['cv_mae_first_training']:.4f} con primera observación y {label_sensitivity['cv_mae_median_training']:.4f} al entrenar con mediana, diferencia {label_sensitivity['delta_median_minus_first']:+.4f} puntos; cambian {label_sensitivity['training_labels_changed']} etiquetas de train. No demuestra que una regla sea la verdad temporal ni superior en otra población. Se mantiene la política por trazabilidad y ausencia de fuente autoritativa, no por elegir el menor número. La calidad imperfecta del objetivo sigue siendo una limitación medible.

### Hallazgos exploratorios

La media de popularidad es {qa['popularity_mean']:.2f} y la mediana {qa['popularity_median']:.0f}. Las correlaciones lineales individuales de audio con popularidad son débiles: la mayor absoluta es aproximadamente 0,128 para instrumentalidad. Esto no excluye relaciones no lineales, pero hace poco razonable atribuir el éxito a una sola propiedad acústica. EDA describe asociaciones de esta muestra, sin demostrar causalidad ni representatividad de Spotify completo.

![Distribución de popularidad](../images/eva2/01_popularity.png)

La figura compara distribuciones normalizadas de train/test. La división preserva aproximadamente la distribución del objetivo a nivel de grupo, sin exigir que cada versión cuente como unidad independiente. `descriptive.csv`, `genres.csv` y `explicit.csv` contienen estadísticas para profundizar.

![Correlaciones](../images/eva2/02_correlations.png)

### Partición y prevención de fuga

Además de los IDs repetidos se detectan {qa['repeated_audio_rows']:,} filas con huella exacta de las 14 variables de audio/binarias, agrupadas en {qa['audio_groups']:,} huellas. {qa['audio_groups_multiple_targets']:,} grupos contienen popularidades distintas. La misma acústica puede tener distintas puntuaciones, lo que también muestra información que el audio no permite explicar.

Se dividen **grupos completos**, estratificando por deciles de la mediana de popularidad de cada grupo, con semilla 42. La separación es 80/20 de grupos y resulta en {qa['train_rows']:,} canciones de entrenamiento y {qa['test_rows']:,} de prueba ({qa['test_pct']:.2f}% de filas). Es una adaptación fundamentada del 80/20 de clase: se prioriza que versiones exactas permanezcan juntas. `split_manifest.csv` contiene ID, grupo y conjunto.

No hay IDs ni grupos exactos compartidos. Las huellas no detectan versiones casi iguales, remasterizaciones o relaciones por álbum. Los artistas sí pueden repetirse: {qa['test_artists_seen_pct']:.2f}% de las filas de test tienen la misma cadena de artistas que alguna fila de train. Por tanto, esta evaluación no es una validación estricta para artistas nuevos. La auditoría de cadenas vistas/no vistas no resuelve todas las colaboraciones.

La validación cruzada interna es GroupKFold de 3 folds sobre train. Cada estimador incorpora imputación, codificación y escalado en un `Pipeline` ajustado de nuevo dentro de cada fold. El test no participa en escalado, entrenamiento, búsqueda de parámetros ni decisión de modelo. La selección se escribe en `selection.json` antes de predecir sobre test. EDA global describe la muestra y no se utiliza para seleccionar predictores o ajustar umbrales a sus resultados.

## 5. Construcción de modelos supervisados

La tarea es **regresión**, porque se estima una puntuación numérica. Accuracy, F1 y matriz de confusión no son métricas de esta tarea. No se binariza artificialmente la popularidad para simular un problema diferente.

| Modelo | Por qué se usa | Configuración y control |
|---|---|---|
| Dummy media | Referencia de experiencia 1 y comparación de R² | Media de y de cada fold/train |
| Dummy mediana | Referencia adecuada para pérdida absoluta | Mediana de y de cada fold/train |
| Regresión lineal | Modelo trabajado en clase, interpretable y económico | Mínimos cuadrados, tol=1e-6, intercepto, sin búsqueda adicional |
| Árbol de decisión | Modelo de clase que capta reglas no lineales | max_depth [8,16,24] x min_samples_leaf [10,30], semilla 42 |
| HistGradientBoosting | Contraste adicional con ensamble de árboles | 220 iteraciones, learning_rate=0,08, max_leaf_nodes [15,31], L2 [1,10], semilla 42 |

Lineal y árbol satisfacen los dos algoritmos supervisados solicitados. El ensamble permite contrastar si una mayor flexibilidad mejora la capacidad de generalización. No se elige por ser más complejo.

Predictores: diez continuas, cuatro categóricas y `explicit`. Para lineal y árbol se aplica OneHotEncoder con categorías desconocidas ignoradas. Para el ensamble se usa OrdinalEncoder y se declaran expresamente como categóricas las columnas codificadas, evitando tratarlas como orden numérico. Categorías no vistas se codifican -1 y se marca el género desconocido en inferencia. StandardScaler de continuas facilita la regresión y una transformación coherente; no es necesario para los umbrales de los árboles.

La búsqueda evalúa 13 configuraciones en los mismos 3 folds (39 ajustes de CV, más refits). `cv_all_candidates.csv` conserva parámetros y resultados. Se selecciona el menor MAE medio de CV, incluso si una referencia constante resultara mejor. Se recortan todas las predicciones a [0,100], tanto al puntuar CV como en test e inferencia; es una regla de dominio predefinida y no un ajuste posterior para favorecer al ganador. Los estimadores minimizan error cuadrático al entrenar, mientras la selección prioriza MAE por su lectura de negocio.

Parámetros elegidos del árbol: `{selection['best_parameters']['Arbol_decision']}`. Del ensamble: `{selection['best_parameters']['HistGradientBoosting']}`. Se desactiva early stopping interno en el ensamble para que no introduzca una validación por filas que mezcle grupos exactos. El número de iteraciones permanece fijo en la grilla predefinida.

La profundidad 8/16/24 explora reglas de complejidad creciente y los mínimos de 10/30 observaciones por hoja limitan hojas muy específicas. En el ensamble, 15/31 hojas contrastan capacidad, y L2=1/10 contrasta regularización. Las 220 iteraciones y tasa 0,08 son una configuración fija de alcance computacional moderado y actualización gradual, no un óptimo descubierto ni una exigencia docente. La semilla 42 facilita repetir el experimento. El ganador de la grilla alcanza el mayor número de hojas y L2 probados: la búsqueda es limitada y no permite declarar optimalidad global. Los valores se fijaron sin buscar mejoras en el test; ampliar la búsqueda requeriría un protocolo nuevo de evaluación.

## 6. Evaluación e interpretación supervisada

### Fórmulas y unidades

Para errores e_i = y_i - pred_i: MAE = media(|e_i|); MSE = media(e_i²); RMSE = raíz(MSE); R² = 1 - suma(e_i²)/suma((y_i - media(y))²). MAE y RMSE usan puntos de popularidad, MSE puntos al cuadrado. R² puede ser negativo si se predice peor que usar la media del test como referencia matemática. No es el porcentaje de canciones acertadas.

{md_table(results[['model','cv_mae','cv_std','train_mae','mae','rmse','r2']], 4)}

MSE del modelo seleccionado: **{metrics['mse']:.3f} puntos²**. La tabla se ordena por CV, no por test. Se reportan todos en el mismo holdout para comparación, pero la selección ya estaba congelada. CV_std expresa dispersión entre 3 folds y no es un intervalo de confianza.

![Comparación de modelos](../images/eva2/03_model_comparison.png)

En palabras de negocio, el ganador se equivoca en promedio **{metrics['mae']:.2f} puntos**. Frente a predecir siempre la media, reduce el error de {base.mae:.2f} a {metrics['mae']:.2f}. El RMSE de {metrics['rmse']:.2f}, mayor que MAE, muestra influencia de errores grandes. R²={metrics['r2']:.3f} implica alrededor de {100*metrics['r2']:.1f}% de variación explicada en esta prueba; permanece mucha variación sin explicar.

El MAE de entrenamiento del ganador es {best.train_mae:.2f}, CV {best.cv_mae:.2f} y test {best.mae:.2f}. Existe brecha entre entrenamiento y validación. Estos números no prueban ausencia de sobreajuste ni garantizan rendimiento futuro, pero permiten cuantificarlo. La prueba ligeramente mejor que CV puede responder a composición aleatoria del holdout.

### Incertidumbre y residuos

Con 600 remuestreos bootstrap de grupos de audio de test, el IC percentil 95% del MAE es **[{ci[0]:.2f}, {ci[1]:.2f}]**. El IC de mejora vs. media es [{ev['confidence']['improvement_pct_95_ci'][0]:.2f}%, {ev['confidence']['improvement_pct_95_ci'][1]:.2f}%]. Se conservan juntas las versiones exactas. Es incertidumbre condicional a este modelo y este holdout, no incorpora dependencia entre artistas ni variabilidad de reentrenar y no es un intervalo para una canción individual.

Sesgo residual medio real - predicho: {ev['residual_bias']:.2f}. P95 del error absoluto: {ev['residual_p95_absolute']:.2f} puntos. Esto impide interpretar MAE como una cota de error de cada predicción. Se recortaron {int(best.clipped_count)} predicciones del ganador por exceder [0,100].

![Residuos](../images/eva2/04_residuals.png)

{md_table(bands[['target_band','n','mae','bias','actual_mean','predicted_mean']])}

El modelo sobreestima canciones poco populares y subestima las muy populares. El intervalo 81-100 tiene {int(bands.iloc[-1].n)} casos y MAE {bands.iloc[-1].mae:.2f}. No se recomienda como detector automático de futuros éxitos. El error por banda se analiza después de predecir y usa el objetivo real exclusivamente para auditoría, no como información disponible en producción.

### Dependencia del género y valor incremental

![Importancia](../images/eva2/05_importance.png)

Permutar `track_genre` aumenta el MAE en {imp.iloc[0].mae_increase:.2f} puntos en una muestra fija de 2.500 filas de test, con cuatro repeticiones. Es importancia predictiva dependiente de este modelo y muestra, no causalidad. Las variables correlacionadas pueden compartir información y la permutación romper combinaciones realistas.

Como auditoría posterior sin modificar la selección, se compara una media por género aprendida únicamente en train y una ablación HGB que **retira exclusivamente `track_genre`**. Conserva las diez continuas, `key`, `mode`, `time_signature` y `explicit`: catorce predictores. Mantiene hiperparámetros, folds y reglas de evaluación; los índices categóricos se ajustan a la nueva matriz para que tonalidad, modo y compás sigan siendo categorías. No vuelve a optimizar ni modifica la selección. `sensitivity_protocol.json` enumera las variables retenidas y la configuración común. Sustituye al contraste anterior que retiraba otras categorías y no aislaba el género.

{md_table(sensitivity)}

La mejora frente a la media por género es **{gain_genre:.2f}%**, mucho menor que {ev['improvement_vs_mean_pct']:.2f}% frente a la constante global. Por eso el valor incremental debe describirse con prudencia. El modelo depende de cómo el dataset asigna géneros y no demuestra que características acústicas causen popularidad. Los contrastes están en `sensitivity_genre.csv` y su protocolo confirma que no alteraron la selección.

## 7. Aprendizaje no supervisado

### Justificación, escalado y selección de K

Se propone agrupar sonidos similares para explorar perfiles de catálogo sin usar popularidad como etiqueta. K-Means es adecuado como primera técnica de clase por su sencillez y centroides interpretables; usa distancias euclidianas y favorece grupos aproximadamente compactos, por lo que no descubre necesariamente todos los estilos musicales. Se usan las diez continuas de audio, sin objetivo, género, IDs ni nombres.

StandardScaler se ajusta solo en train. Así la duración en milisegundos no domina sobre indicadores 0-1. Se mantienen variables originales sin borrar valores extremos válidos; el escalado y K-Means pueden seguir siendo sensibles a ellos. Se comparan K=1..10 con n_init=10, max_iter=300 y semilla 42 sobre todas las filas de entrenamiento.

![Codo y silueta](../images/eva2/06_k_selection.png)

{md_table(pd.read_csv(out / 'clustering_diagnostics.csv')[['k','inertia','silhouette','smallest_cluster_pct']])}

La inercia siempre disminuye al aumentar K, por lo que su mínimo no selecciona un óptimo útil. El codo orientativo se calcula como la mayor distancia a la cuerda normalizada y sugiere **K={cl['elbow_chord_k']}**. No reemplaza la revisión de la curva. La regla fijada en código elige la mayor silueta de una muestra fija de 2.500 filas de train entre K=2..10, exigiendo que cada grupo cubra al menos 3% del entrenamiento. K=1 no tiene silueta definida.

Se elige **K={cl['selected_k']}**, con silueta de train {cl['silhouette_train_sample']:.3f} y de prueba {cl['silhouette_test_sample']:.3f}. Silueta va de -1 a 1: una cifra cercana a 0 indica fronteras poco claras. Estos valores reflejan perfiles amplios con superposición, no una taxonomía musical definitiva. El codo y la silueta difieren: se prioriza la regla explícita de separación y parsimonia para dos macroperfiles. También se calcularon perfiles de la alternativa K={cl['elbow_chord_k']} en train/test para sostener esa discusión con datos, sin cambiar la regla de selección:

{md_table(elbow_profiles[['cluster','n','pct','energy','acousticness','speechiness','instrumentalness','tempo']])}

La alternativa aporta más detalle, pero su silueta en la misma muestra fija es {pd.read_csv(out / 'clustering_diagnostics.csv').set_index('k').loc[cl['elbow_chord_k'], 'silhouette']:.3f}, inferior a la del K elegido. Requiere interpretación de escucha antes de adoptarla. No se afirma que cuatro grupos sean incorrectos ni que la silueta decida por sí sola su valor comercial.

Estabilidad entre semillas:

{md_table(stability)}

ARI cercano a 1 sugiere asignaciones similares bajo estas inicializaciones, aun cuando los números de cluster se permuten. No evalúa estabilidad temporal ni cambios en la muestra. Popularidad se resume después de agrupar, exclusivamente para descripción, nunca para construir o elegir grupos.

### Perfiles y aplicación de negocio

{md_table(profiles[['cluster','n','pct','energy','danceability','acousticness','instrumentalness','loudness','tempo','popularity']])}

![Perfiles](../images/eva2/08_cluster_profiles.png)

Grupo 0, **relativamente energético**: mayor energía y sonoridad, menor carácter acústico. Hipótesis de uso: ayudar al curador a explorar candidatos para listas de actividad. Grupo 1, **relativamente acústico y suave**: menor energía, mayor acústica e instrumentalidad. Hipótesis de uso: explorar selecciones tranquilas. Los nombres se asignan al interpretar promedios; no proceden de una validación de escucha ni garantizan adecuación a una actividad.

Se recomienda muestrear canciones de cada grupo y revisar diversidad de géneros y artistas antes de cualquier lista. La diferencia de popularidad media entre grupos es descriptiva y no justifica promover un perfil ni excluir el otro. No se calculó aumento de reproducciones ni retorno comercial. Los perfiles de test se guardan separados en `cluster_profiles_test.csv` para no confundir validación con entrenamiento.

### PCA y límite de la visualización

![PCA](../images/eva2/07_pca_clusters.png)

PCA se ajusta en los datos escalados de train y proyecta test. Dos componentes conservan **{cl['pca_two_components_variance']:.2%}** de la varianza. Se requieren **{cl['pca_components_for_80pct']} componentes** para alcanzar 80%. El agrupamiento se ajusta en las diez dimensiones, no en este mapa reducido; un solapamiento o separación visual en dos dimensiones no demuestra por sí solo la calidad del clustering.

![Varianza PCA](../images/eva2/10_pca_variance.png)

## 8. Ética, sesgos y seguridad

| Riesgo observado o plausible | Evidencia / límite | Control propuesto |
|---|---|---|
| Confundir popularidad con mérito | La etiqueta refleja exposición observada, no calidad | Revisión humana, no exclusión de artistas por score |
| Dependencia del género | Importancia y baseline por género; una etiqueta por ID | Mostrar límites, revisar anotación y diversidad de selección |
| Desempeño desigual | Errores desglosados por género, explicit y artistas vistos | Revisar segmentos con n suficiente, mostrar baseline del grupo |
| Sesgo de extremos | MAE alto en bandas populares | No automatizar promoción o contratos |
| Privacidad y atribución | Nombres de artistas presentes, sin datos de oyentes | No incorporar identidades al predictor, reportar agregados |
| Versiones y fuga residual | Audio exacto agrupado, versiones cercanas sin resolver | Evaluar futuras particiones por grabación y artistas |
| Procedencia y derechos | Fecha y licencia actual no confirmadas | Validar derechos antes de distribución pública |
| Persistencia de modelos | Joblib usa deserialización de objetos | Cargar solo archivos propios, verificar versiones y reentrenar |

Que no haya datos de oyentes no implica que todos los nombres de artistas carezcan de relevancia para privacidad o atribución. No se infieren atributos sensibles. El informe de errores no incluye nombres de artistas; solo IDs y la bandera de cadena de artista vista en train. Las métricas de equidad aquí son descriptivas por segmentos disponibles, no una certificación de ausencia de discriminación. No existen atributos demográficos para auditar grupos protegidos.

### Auditoría de subgrupos

{md_table(explicit[['explicit','n','mae','baseline_mae','improvement_pct']])}

{md_table(artists[['artist_seen','n','mae','baseline_mae','improvement_pct']])}

No se interpreta una diferencia de MAE como causalidad o discriminación probada: las distribuciones de popularidad y tamaños también difieren. La referencia por grupo ayuda a contextualizarlo. `artist_seen` compara cadenas completas, no individuos tras separar colaboraciones. No es una prueba externa de artistas desconocidos.

![Errores por género](../images/eva2/09_subgroups.png)

Cinco géneros con mayor error entre los que tienen al menos 30 casos:

{md_table(genre_errors.head(5)[['track_genre','n','mae','baseline_mae','improvement_pct']])}

Los grupos con menos de 30 observaciones permanecen en el CSV completo con una bandera que impide presentarlos como evidencia estable. El umbral es una regla de reporte, no prueba de significancia ni garantía estadística. No se hicieron contrastes múltiples para declarar diferencias significativas.

## 9. Arquitectura, reproducibilidad y pruebas

Flujo: CSV inmutable y hash -> limpieza -> huellas y partición -> pipelines y CV -> selección congelada -> prueba y auditoría -> K-Means/PCA -> gráficos e informes -> inferencia local. La etapa no supervisada usa la misma reserva de prueba y no depende de etiquetas del supervisado. No se requiere acceso a una API de Spotify ni credenciales.

El [README](../README.md) incluye instalación y comandos exactos. `python -m src.eva2.run` parte del CSV original y regenera todos los resultados de evaluación, gráficos, informes y modelos actuales. El notebook 05 ejecuta las mismas funciones en etapas y conserva salidas. No es un cuaderno que solo muestre resultados guardados. Los notebooks 01-04 son antecedentes históricos y no son prerrequisitos del 05.

Semilla 42, hasta cuatro hilos, tres folds de grupos y grillas explícitas. Entorno medido: Python {read('environment.json')['python']}; las versiones de paquetes se fijan en requirements y en `environment.json`. Cambios de hardware o bibliotecas numéricas pueden producir diferencias pequeñas. Se incluye el CSV para ejecutar sin descargar datos; no se promete soporte de versiones que no se probaron.

Pruebas automáticas: rechazo de hash distinto; huellas independientes de target/género/ID; separación reproducible de audio exacto; ajuste de escalado solo con train; validación de entradas inválidas; serialización e inferencia con género desconocido. La verificación final comprueba también ausencia de errores en notebook, coherencia de métricas y presencia de entregables. Consultar `reports/eva2/verification.json` para el resultado efectivo, sin asumirlo antes de ejecutarlo.

Las pruebas de regresión adicionales cubren: ablación que conserva todas las variables excepto género; auditoría de etiquetas sin mutar la referencia; integridad del ZIP sin invocar Git; rechazo de fuentes o CSV alterados; fórmulas del piloto y rechazo de una plantilla sin observaciones. En el repositorio se valida el CSV contra Git; en el ZIP se utiliza primero `MANIFIESTO_ENTREGA.json`, aun si la carpeta está dentro de otro repositorio. El código y entradas deben coincidir con el manifiesto, mientras las salidas pueden regenerarse. No se requiere instalar Git para ejecutar la entrega extraída. El constructor del notebook crea un kernel privado con el mismo Python que lo ejecuta, evitando seleccionar accidentalmente otro entorno global.

## 10. Despliegue local y mantenimiento

Se implementa una herramienta batch local: `python -m src.eva2.predict data/examples/tracks_example.csv --output reports/eva2/example_predictions.csv`. Acepta las 15 variables, valida dominios, carga el pipeline completo propio y entrega puntuaciones acotadas más bandera de género desconocido. El ejemplo procede de test, excluye el target y sirve para comprobar el contrato, no como nueva prueba independiente. No hay despliegue web ni producción comercial en esta entrega.

Ficha del modelo: entrada de audio/metadatos descrita arriba; salida de 0 a 100; población limitada por filtros de limpieza; uso previsto de consulta de catálogo; usos no respaldados: decisiones automáticas de financiamiento, inferencia causal y popularidad futura. El modelo persistido conserva su hash de datos, versión sklearn y parámetros.

Plan propuesto: responsable técnico del equipo revisa nulos, dominios y categorías por cada lote; responsable de curaduría revisa muestras y diversidad; cuando existan nuevas etiquetas compara MAE y errores por segmento. Revisar antes de uso si aparecen nuevas categorías frecuentes o un aumento de MAE relativo >10% frente a la referencia de validación del piloto. Ese umbral es una propuesta operativa sin validación económica. Conservar versión previa para retorno y reentrenar únicamente después de construir una nueva prueba temporal. No reusar indefinidamente este test para seleccionar mejoras.

## 11. Conclusiones

Se implementaron tres modelos supervisados y dos referencias, con selección por CV agrupada y prueba independiente. {ev['selected_model']} cumple los dos objetivos técnicos iniciales, con MAE {metrics['mae']:.2f} y mejora {ev['improvement_vs_mean_pct']:.2f}% frente a la constante. La mejora frente al género por sí solo es {gain_genre:.2f}%, y el error en extremos sigue siendo elevado. El resultado respalda exploración con revisión humana, no promesas de éxito musical o ROI.

K-Means aporta dos perfiles interpretables y estables ante semillas, pero la separación es limitada y no coincide con el codo de cuatro grupos. PCA muestra una parte de la estructura y no certifica estilos. La principal mejora metodológica frente a experiencia 1 es completar modelado/evaluación, controlar versiones exactas entre conjuntos y sostener las recomendaciones con errores y limitaciones medidos.

## 12. Matriz de trazabilidad de la rúbrica

| Indicador y peso | Evidencia de implementación | Justificación e interpretación |
|---|---|---|
| IE5: metodología, 20% | Secciones 1-4, 9-10; plan CRISP-DM; hash, esquema, limpieza, partición y CLI | Seis fases conectadas con objetivos y KPIs; partición revisada por audio repetido |
| IE6: supervisados, 30% | Secciones 5-6; lineal, árbol y ensamble; grillas, CV y selección persistida | Regresión para objetivo numérico, selección por MAE y control de fuga entre grupos |
| IE7: no supervisados, 30% | Sección 7; escalado, K=1..10, codo/silueta, perfiles, estabilidad y PCA | K=2 maximiza silueta elegible; K=4 es alternativa explorada; perfiles y varianza limitada explícitos |
| IE8: interpretación, 20% | Secciones 6, 8 y 11; métricas, residuos, contraste por género y límites | Mejora medible para consulta de catálogo; errores altos en extremos limitan decisiones de promoción |

La matriz se respalda con el protocolo de ablación controlada, la auditoría de etiquetas y sensibilidad sin modificar el test, los KPIs comerciales propuestos en el [plan de piloto](plan_piloto_negocio.md) y las comprobaciones reproducibles de `reports/eva2/verification.json`.

Entregables formales: informe Markdown, síntesis ejecutiva, notebook ejecutado, dataset, modelos, figuras, fuentes y carpeta organizada. Esta matriz identifica evidencias; no asigna una nota ni sustituye el juicio del docente o la defensa individual.

## 13. Referencias

[Material del profesor y fuentes primarias verificadas](fuentes_eva2.md). La pauta entregada define los requisitos, y las guías 2.1-2.4 orientan el método. Se adaptan los ejemplos bancarios al caso Spotify. Documentación adicional del ensamble: [HistGradientBoostingRegressor, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). [Regresión lineal](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LinearRegression.html) y [árbol de regresión](https://scikit-learn.org/1.8/modules/generated/sklearn.tree.DecisionTreeRegressor.html). Consulta: 20 de octubre de 2026.
"""
    (root / "docs/informe_ejecutivo.md").write_text(executive, encoding="utf-8")
    (root / "docs/informe_tecnico_eva2.md").write_text(report, encoding="utf-8")
    return {"technical": root / "docs/informe_tecnico_eva2.md", "executive": root / "docs/informe_ejecutivo.md"}
