# Evaluación de ética, sesgos y privacidad

## 1. Propósito del análisis

El proyecto busca estimar la popularidad de canciones a partir de características musicales y género. La predicción puede apoyar decisiones de análisis o promoción, pero no debe utilizarse como una garantía de éxito ni como el único criterio para asignar oportunidades a artistas.

Los resultados describen patrones del dataset disponible. No prueban causalidad y no representan necesariamente a toda la música publicada en Spotify.

## 2. Evidencia de representación

El dataset limpio contiene 89.495 canciones, 31.374 combinaciones distintas de artistas y 113 géneros.

| Indicador | Resultado | Riesgo asociado |
|---|---:|---|
| Artistas que aparecen una sola vez | 64,12% | El modelo puede aprender poco sobre artistas con baja representación. |
| Canciones concentradas en el 1% de artistas más frecuentes | 17,27% | Los artistas recurrentes pueden influir más en el aprendizaje. |
| Registros por género | Entre 73 y 1.000 | Algunos géneros tendrán estimaciones menos confiables. |
| Canciones con popularidad igual a 0 | 10,53% | El cero puede mezclar falta de éxito con falta de exposición o medición. |
| Canciones con popularidad mayor o igual a 70 | 3,49% | Los casos de alta popularidad son escasos. |
| Canciones explícitas | 8,61% | La comparación con canciones no explícitas está desbalanceada. |

Estas cifras deben recalcularse si cambia el dataset o la regla de limpieza.

## 3. Sesgo por género musical

Los géneros no tienen la misma cantidad de registros. En el dataset limpio, algunos se acercan a 1.000 canciones, mientras `reggaeton` contiene 73. Además, durante la limpieza se conservó un solo registro por `track_id`, manteniendo el primer género disponible. Esta decisión puede reducir la representación de canciones asociadas a varios géneros.

Consecuencias posibles:

- Mayor error en géneros con pocas observaciones.
- Predicciones que favorezcan patrones de géneros dominantes.
- Interpretación incorrecta de una popularidad baja como menor calidad artística.
- Pérdida de información en canciones multigénero.

Medidas propuestas:

- Medir MAE y RMSE por género, además del resultado global.
- Informar el número de casos utilizado para cada métrica.
- No publicar conclusiones para grupos con muestras demasiado pequeñas.
- Comparar una versión multietiqueta del género en trabajos futuros.
- Revisar si la eliminación de `track_id` repetidos afecta desproporcionadamente a algún género.

## 4. Diferencias de representación entre artistas

La mayoría de los artistas aparece muy pocas veces, mientras algunos tienen más de cien canciones. Por ejemplo, el artista más frecuente tiene 260 registros. Un modelo puede ajustarse mejor a estilos y trayectorias de artistas recurrentes y producir resultados menos confiables para artistas nuevos.

Los nombres de artistas fueron excluidos del pipeline inicial. Esta decisión disminuye la posibilidad de memorizar artistas concretos, pero no elimina por completo el sesgo: género y características de audio pueden funcionar como variables indirectas relacionadas con ciertos grupos.

Medidas propuestas:

- Evaluar errores según frecuencia del artista: una aparición, entre dos y diez, y más de diez.
- Mantener los nombres fuera del modelo base.
- Realizar una validación adicional reservando artistas completos para prueba.
- Informar incertidumbre o advertencias cuando la canción pertenezca a un segmento poco representado.

## 5. Sesgo hacia canciones conocidas o comerciales

`popularity` refleja atención obtenida en la plataforma. Esa atención puede depender de campañas publicitarias, presencia en listas, tamaño de la audiencia previa, sello discográfico, antigüedad de la canción, país y tendencias culturales. Estas variables no están disponibles en el dataset.

Si una predicción alta se utiliza para decidir qué canciones promocionar, se puede producir un ciclo de retroalimentación:

1. Las canciones parecidas a éxitos anteriores reciben predicciones altas.
2. Reciben mayor promoción y visibilidad.
3. Obtienen más reproducciones.
4. Los datos futuros refuerzan el mismo patrón.

Este ciclo podría reducir la diversidad y dificultar el acceso de propuestas nuevas.

Medidas propuestas:

- No automatizar decisiones de inversión o promoción.
- Reservar una parte de las oportunidades para exploración y diversidad.
- Complementar la predicción con evaluación humana.
- Monitorear cambios en la representación de géneros y artistas después de usar el modelo.
- Actualizar el modelo periódicamente y comparar con datos recientes.

## 6. Limitaciones de `popularity`

La popularidad no equivale a calidad, valor cultural ni satisfacción del público. Es una medición dependiente de Spotify y puede cambiar con el tiempo.

Limitaciones principales:

- El dataset no indica la fecha exacta de medición del puntaje.
- No se conoce cuánto tiempo llevaba publicada cada canción.
- Un valor cero puede significar baja audiencia, falta de exposición o información desactualizada.
- La popularidad puede variar por país y contexto cultural.
- La distribución tiene pocos casos de popularidad alta.
- No existen variables sobre promoción, seguidores o inclusión en listas.

Por estas razones, el modelo debe presentarse como una estimación limitada al contexto de los datos disponibles.

## 7. Privacidad e identificadores

El dataset contiene `track_id`, nombres de canciones, álbumes y artistas. No incluye historiales individuales de escucha, correos, ubicaciones de usuarios ni otras variables personales sensibles.

Aun así, se deben considerar los siguientes riesgos:

- `track_id` es un identificador persistente que facilita vincular registros con otras fuentes.
- Los nombres identifican públicamente a artistas y pueden generar evaluaciones reputacionales.
- Combinar estos datos con información privada o perfiles de usuarios aumentaría el riesgo de reidentificación.
- La fuente exacta y la licencia del archivo todavía deben documentarse antes de reutilizar o redistribuir los datos fuera del contexto académico.

Medidas propuestas:

- No utilizar nombres ni IDs como predictores del modelo base.
- Conservar solo las variables necesarias para el propósito definido.
- No incorporar datos de usuarios sin consentimiento y una evaluación adicional de privacidad.
- Limitar el acceso a archivos de trabajo y registrar cambios mediante Git.
- Confirmar la licencia y condiciones de uso de la fuente original.
- Evitar publicar rankings que puedan presentarse como juicios sobre artistas.

## 8. Riesgo para artistas emergentes

Los artistas emergentes suelen tener menor exposición previa y menos ejemplos en el dataset. El modelo podría interpretar esa falta de historial como bajo potencial y recomendar menos promoción, reforzando la desventaja inicial.

Medidas propuestas:

- No usar la predicción como filtro automático de exclusión.
- Crear una categoría de revisión especial para artistas con poca representación.
- Evaluar el modelo mediante una prueba que reserve artistas no vistos.
- Mostrar un rango de incertidumbre junto con la predicción.
- Combinar criterios de popularidad con diversidad, novedad y objetivos editoriales.

## 9. Riesgos del atributo `explicit`

El EDA muestra una diferencia descriptiva de popularidad entre canciones explícitas y no explícitas. Sin embargo, `explicit` puede estar asociado con género, mercado, idioma o público. Interpretarlo aisladamente puede producir conclusiones morales o culturales incorrectas.

Se debe:

- Evitar afirmar que el contenido explícito causa popularidad.
- Comparar su efecto controlando otras variables.
- Medir desempeño por ambos grupos.
- No usar esta variable para censurar o excluir contenido automáticamente.

## 10. Matriz de riesgos y controles

| Riesgo | Probabilidad | Impacto | Control principal |
|---|---|---|---|
| Error elevado en géneros pequeños | Alta | Alto | Métricas por género y tamaño mínimo de muestra. |
| Menor precisión para artistas emergentes | Alta | Alto | Validación por artista y revisión humana. |
| Ciclo de popularidad y promoción | Media | Alto | No automatizar decisiones y reservar exploración. |
| Interpretar popularidad como calidad | Alta | Alto | Advertencia explícita y comunicación responsable. |
| Uso reputacional de nombres | Media | Medio | Excluir nombres e impedir rankings individuales. |
| Vinculación de `track_id` con otras fuentes | Media | Medio | Minimización y control de acceso. |
| Categorías desconocidas en producción | Media | Medio | One-Hot Encoding tolerante y monitoreo. |
| Cambio de tendencias con el tiempo | Alta | Alto | Evaluación temporal y actualización periódica. |

## 11. Evaluación responsable del futuro modelo

Antes de utilizar un modelo se deben calcular:

- MAE y RMSE globales.
- MAE por género, contenido explícito y nivel de representación del artista.
- Diferencia entre el mejor y peor error grupal.
- Cantidad de observaciones detrás de cada resultado.
- Desempeño en artistas no vistos durante el entrenamiento.
- Estabilidad de las métricas en datos de otro periodo.

Una métrica global aceptable no será suficiente si existen grupos con errores considerablemente mayores.

## 12. Uso responsable

Usos aceptables:

- Apoyar análisis académicos.
- Explorar relaciones entre características musicales.
- Comparar modelos y estudiar limitaciones.
- Generar una señal adicional para decisiones humanas.

Usos no recomendados:

- Decidir automáticamente qué artistas reciben financiamiento.
- Excluir canciones solo por una predicción baja.
- Presentar popularidad como medida de calidad artística.
- Elaborar rankings reputacionales de artistas.
- Combinar los datos con información privada de usuarios sin autorización.

## 13. Conclusión ética

El proyecto puede aportar información útil, pero el dataset refleja desigualdades de exposición y representación. La popularidad es una señal incompleta y dinámica. El modelo debe mantenerse como herramienta de apoyo, acompañado por evaluación grupal, transparencia, revisión humana, monitoreo y oportunidades explícitas para artistas y géneros menos representados.
