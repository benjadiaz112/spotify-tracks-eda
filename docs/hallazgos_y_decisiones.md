# Hallazgos, evidencia y decisiones

Resumen de lectura rápida. Las cifras completas y protocolos están en el informe técnico. Este documento no es una presentación.

| Hallazgo medido | Evidencia | Interpretación | Decisión respaldada |
|---|---|---|---|
| 114.000 filas representan 89.741 IDs y quedan 89.495 canciones limpias | `data_audit.json`, `cleaning.csv` | El 78,50% de retención bruta refleja sobre todo duplicación. La retención de IDs es 99,73% | Explicar ambas medidas y el incumplimiento de la meta histórica sobre filas |
| 6.213 filas repiten audio exacto de otro ID | Huellas y `split_manifest.csv` | Quitar IDs repetidos no basta para independizar versiones | Partición y CV por huella, sin audio exacto compartido |
| Correlaciones de audio con popularidad débiles | `correlations.csv` | Una característica aislada no explica el éxito | Comparar varios modelos sin atribuir causalidad |
| Ganador MAE 11,24 frente a 17,08 de la constante | `cv_summary.csv`, `supervised_metrics.csv` | Cumple metas técnicas previas con mejora 34,19% | Piloto de consulta con revisión humana |
| Media por género ya logra MAE 11,93 | `sensitivity_genre.csv`, importancia de permutación | El valor incremental es aproximadamente 5,74% y depende de la etiqueta de género | Revisar la asignación de género y explicar la comparación más exigente |
| Popularidad 81-100 tiene MAE 44,91, n=105 | `errors_by_target_band.csv` | El promedio global oculta una fuerte subestimación de extremos | No usar como detector automático de hits ni para asignar inversión |
| K=2 tiene la mayor silueta admisible, pero el codo sugiere K=4 | `clustering_diagnostics.csv`, `clustering.json` | No existe un único óptimo apoyado por ambos diagnósticos | Dos macroperfiles como primera segmentación, cuatro como alternativa a validar |
| Silueta de test 0,237 y ARI alrededor de 0,997 | `clustering.json`, `cluster_stability.csv` | Asignaciones estables con separación moderada. Son cualidades distintas | No presentar los grupos como estilos definitivos |
| Dos componentes PCA retienen 44,44% y seis alcanzan 80% | `pca_variance.csv` | El mapa pierde información relevante | Interpretar la proyección como apoyo visual |
| Falta fecha de extracción y una prueba temporal | Fuente entregada y esquema | No hay evidencia de predicción futura ni representatividad de todo Spotify | Nueva muestra reciente y evaluación temporal antes de uso comercial |

Las recomendaciones comerciales son propuestas. No hay campañas ejecutadas, mejoras de ingresos ni ahorro medido. Los perfiles de actividad/tranquilidad son hipótesis interpretativas que requieren curaduría y escucha.
