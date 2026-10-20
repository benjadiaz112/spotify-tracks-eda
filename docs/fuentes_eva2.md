# Fuentes y comprobaciones

## Material entregado

- Pauta de Evaluación Parcial N°2: copia textual en `docs/pauta_evaluacion_2.txt`. Exige informe Markdown, notebook ejecutable, dataset, estructura profesional, dos modelos supervisados y una técnica no supervisada.
- 2.1.1: CRISP-DM. 2.1.2: estructura de proyecto y separación de prueba.
- 2.2.1 y 2.2.2: regresión lineal y árboles, partición entrenamiento/prueba.
- 2.3.1 y 2.3.2: escalado, K-Means de 1 a 10, codo, silueta y PCA.
- 2.4.1 y 2.4.2: MAE, MSE, RMSE, R² e interpretación por subgrupos.
- ZIP de experiencia 1: código, documentación, notebooks y CSV original. Los artefactos anteriores se conservan como antecedentes, no como métricas de modelos nuevos.

Los extractos de los ocho PDF están en `docs/referencias_clase/`. El texto puede perder diagramas del PDF. La pauta completa es el documento que define la entrega, mientras las guías aportan el enfoque de clase.

## Consulta externa (20 de octubre de 2026)

Fuentes primarias utilizadas para verificar implementación:

- [scikit-learn: prevención de fuga y pipelines](https://scikit-learn.org/stable/common_pitfalls.html). Ajustar transformaciones solo en entrenamiento y dentro de cada fold.
- [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html). Mantener grupos separados en validación.
- [scikit-learn: análisis de silueta para K-Means](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). Contrastar cohesión y separación para diferentes K.
- [Spotify Tracks Dataset, MaharshiPandya, Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). Fuente declarada en experiencia 1. La página no devolvió contenido legible durante la consulta; no se confirmó independientemente su licencia ni versión actual.

La experiencia 1 declara la licencia `Database: Open Database, Contents: © Original Authors`. Esto es un antecedente documental, no una verificación actual de derechos. Se utiliza exclusivamente la copia entregada para la actividad académica. Validar licencia antes de redistribuir públicamente el dataset. No se descargaron nuevos datos.

## Identidad del archivo recibido

CSV: `data/raw/Spotify_Tracks_Dataset.csv`.

SHA-256 calculado: `335e5ee0a07a3f30b20ca9860c87562c98082a032e0ecd90e8ca0a86d965fe58`.

El hash anotado en el README de experiencia 1 era diferente. No se presupone la causa ni se afirma que se verificó aquella versión. El pipeline de esta entrega registra y valida el hash del archivo realmente recibido.
