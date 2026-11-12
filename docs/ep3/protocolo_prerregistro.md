# EP3 Spotify Tracks · Protocolo de evaluación

**Benjamín Palomino y Benjamín Díaz · MLY1101, caso C.**

Decisiones registradas antes de evaluar la reserva de EP3. La pauta oficial
evalúa optimización (IE9, 20%), ensambles (IE10, 30%), generalización
(IE11, 20%) y decisión de negocio (IE12, 30%). Su 80% es una categoría de
logro. El equipo reportó además un mínimo de modelo superior al 80%, sin
una métrica acordada en el material recibido.

1. **Problema y objetivo.** Priorizar canciones candidatas para revisión de
   curaduría. Mantener la regresión de popularidad observada de 0 a 100. No
   cambiar el objetivo, recortar canciones difíciles ni mover el umbral a
   posteriori para cumplir el porcentaje.
2. **Origen y calidad.** Conservar el CSV versionado. El CSV del ZIP de la
   asignatura coincide tras normalizar saltos de línea; no se agregan fuentes
   comerciales ni fechas inexistentes. Documentar filtros, duplicación,
   conflictos de etiqueta y cobertura por género. El objetivo principal es
   el de la primera observación válida por ID, como en EP2.
3. **Partición.** Semilla 3103. Reserva 20% de componentes; cada componente
   vincula coincidencias de audio y de artista principal más título
   normalizado. La estratificación solo organiza grupos. Guardar la membresía
   y comprobar intersecciones vacías. Son canciones del mismo dataset
   histórico, sin independencia temporal ni externa.
4. **Selección.** MAE de validación cruzada agrupada en desarrollo, menor
   es mejor. Comparar constante, regresión regularizada, árbol y ensambles.
   Demostrar el efecto de parámetros en el mismo modelo y el efecto de
   agregar árboles con las mismas entradas. Cualquier peso aprendido usa
   predicciones fuera de fold; no usar reserva para parámetros o pesos.
   El CV de búsqueda sirve para seleccionar; la reserva cuantifica el error
   de la decisión ya fijada. No interpretar selección CV como evaluación
   libre del optimismo de selección.
5. **Métricas.** MAE primario, RMSE y R² secundarios. Informar cobertura
   absoluta en ±5, ±10, ±15 y ±20 puntos. Como criterio operativo **propuesto**
   antes de evaluar EP3: más de 80% de canciones dentro de ±15 puntos. No
   equipararlo con accuracy, R² mayor a 0,80 ni aprobación docente.
   Reportar intervalos por remuestreo de componentes y el resultado de
   subgrupos con cobertura desigual.
6. **Segmentación.** K-Means sobre audio escalado en desarrollo; contrastar
   silueta, codo, estabilidad, perfiles y proyección PCA. No usar género o
   popularidad para definir clusters.
7. **Utilidad.** Simular revisar el 20% de la reserva. Definir popularidad
   alta con el percentil 75 de desarrollo. Medir precisión y cobertura de
   positivos al mismo presupuesto, popularidad media y géneros cubiertos.
   Comparar con la expectativa de selección uniforme; no atribuir impacto
   comercial ni reemplazar revisión humana.
