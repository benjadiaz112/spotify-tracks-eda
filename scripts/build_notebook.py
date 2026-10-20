"""Crear el cuaderno integral y opcionalmente ejecutarlo desde un kernel limpio."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md("""# Spotify Tracks - Evaluación Parcial N°2

**MLY1101, Caso C. Equipo: fluzxks, benjadiaz112 y Freymuth Christopher Isaac.**

Continuamos la experiencia 1 y ejecutamos CRISP-DM completo: negocio, datos, preparación, modelos supervisados, segmentación y evaluación. Este cuaderno parte del CSV original, vuelve a entrenar y guarda evidencia. Usa **Restart Kernel and Run All Cells**. Instala `requirements.txt` desde terminal antes de ejecutarlo. Los scripts importados contienen la implementación y sus comentarios.

La pregunta es cuánta señal aportan audio y género para estimar popularidad **observada**. No tenemos fechas para pronosticar popularidad futura ni datos de campañas para medir ROI. Metas técnicas previas: MAE <=15 y mejora >=10% frente a la media. Los perfiles musicales apoyarán exploración humana, sin decisiones automáticas sobre artistas.""")
    code("""from pathlib import Path
import sys, json, time
from IPython.display import display, Markdown, Image
import pandas as pd

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'data/raw/Spotify_Tracks_Dataset.csv').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Abre el notebook dentro del proyecto completo con data/raw y src.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eva2.data import prepare
from src.eva2.supervised import fit_supervised
from src.eva2.clustering import fit_clustering
from src.eva2.sensitivity import audit_genre, audit_label_policy
from src.eva2.plots import make_plots
from src.eva2.reporting import build_reports
from src.eva2.predict import predict
from src.features.prepare_features import MODEL_FEATURES
start = time.perf_counter()
print('Raíz:', ROOT)
print('Python:', sys.version.split()[0])""")
    md("""## 1. Datos, calidad y preparación

Verificamos el hash del archivo recibido y reutilizamos la limpieza de experiencia 1. El descarte por duración es una decisión heredada y puede excluir piezas válidas. Un ID puede aparecer en varios géneros y la regla conserva su primera etiqueta, limitación que debemos defender.

La separación usa grupos de audio exactamente igual, para que versiones de distinta ID no crucen conjuntos. Se estratifican los grupos según deciles de su popularidad mediana, con 80/20 de grupos. No se ajusta ningún escalador antes de separar. El objetivo solo interviene en estratificación, nunca en la huella o en los predictores.""")
    code("""data = prepare(ROOT)
display(pd.Series(data['audit'], name='valor').to_frame())
display(pd.read_csv(ROOT / 'reports/eva2/cleaning.csv'))
display(data['clean'].describe(include='number').T)
assert data['audit']['shared_tracks'] == 0
assert data['audit']['shared_audio_groups'] == 0
assert data['audit']['clean_missing_cells'] == 0""")
    code("""# EDA: relaciones descriptivas, no causales.
display(pd.read_csv(ROOT / 'reports/eva2/genres.csv').sort_values('mean', ascending=False).head(10))
display(pd.read_csv(ROOT / 'reports/eva2/explicit.csv'))
display(pd.read_csv(ROOT / 'reports/eva2/correlations.csv', index_col=0)['popularity'].sort_values())""")
    md("""### Auditoría de etiquetas y criterio de referencia

No suponemos que duplicados del mismo ID tengan siempre idéntica popularidad. La auditoría cuantifica primera fila, mediana y rango. Se conserva la primera observación por continuidad y falta de fecha autoritativa. Esto define una referencia reproducible, sin convertirla en verdad temporal. En la sección 4 mediremos sensibilidad a entrenar con mediana usando solo train, sin cambiar el target de test ni seleccionar una regla por su score.""")
    code("""display(json.loads((ROOT / 'reports/eva2/target_audit.json').read_text(encoding='utf-8')))
display(pd.read_csv(ROOT / 'reports/eva2/target_conflicts.csv')
        .sort_values('range', ascending=False).head(10))""")
    md("""## 2. Modelos supervisados

Es una regresión: `popularity` es numérica entre 0 y 100. Comparamos referencias de media y mediana, regresión lineal, árbol de decisión y un ensamble HistGradientBoosting. Los dos primeros algoritmos aprendidos en clase cubren el mínimo supervisado y el ensamble amplía la comparación.

GridSearchCV usa los mismos tres folds agrupados para todos y reentrena cada pipeline dentro del fold. La selección por MAE de CV se guarda antes de calcular métricas de test. No se cambia el ganador al mirar test. Los parámetros y resultados de las 13 configuraciones quedan en CSV y JSON. Las predicciones se acotan a [0,100] como regla de dominio en CV, test e inferencia.""")
    code("""supervised = fit_supervised(data, ROOT)
display(supervised['results'])
display(pd.Series(supervised['evaluation']))
display(json.loads((ROOT / 'reports/eva2/selection.json').read_text(encoding='utf-8')))
assert not json.loads((ROOT / 'reports/eva2/selection.json').read_text(encoding='utf-8'))['test_used_for_selection']""")
    md("""## 3. K-Means, perfiles y PCA

Seleccionamos diez continuas de audio, sin popularidad ni género. Escalamos solo con train y probamos K=1..10. La regla elige la mayor silueta en una muestra fija de 2.500 filas de train, exigiendo grupos >=3%. El codo es un diagnóstico adicional y puede discrepar con esa regla.

La prueba se transforma con el escalador y los centroides aprendidos. Repetimos inicializaciones con semillas alternativas y calculamos ARI. PCA se ajusta en train y proyecta test: dos componentes pueden omitir gran parte de la estructura. Los promedios de popularidad por grupo se calculan después del ajuste y solo describen los grupos.""")
    code("""clusters = fit_clustering(data, ROOT)
display(clusters['diagnostics'])
display(pd.read_csv(ROOT / 'reports/eva2/cluster_profiles_train.csv'))
display(pd.read_csv(ROOT / 'reports/eva2/cluster_profiles_test.csv'))
display(pd.read_csv(ROOT / 'reports/eva2/cluster_profiles_elbow_train.csv'))
display(pd.read_csv(ROOT / 'reports/eva2/cluster_stability.csv'))
display(pd.read_csv(ROOT / 'reports/eva2/pca_variance.csv'))
display(json.loads((ROOT / 'reports/eva2/clustering.json').read_text(encoding='utf-8')))""")
    md("""## 4. Evaluación crítica y auditoría de género

MAE es el error absoluto medio en puntos de popularidad. RMSE penaliza más los errores grandes. R² compara variación explicada frente a la media de prueba, y no mide porcentaje de canciones acertadas. El bootstrap por grupos mide incertidumbre condicionada a este modelo y a esta prueba, sin certificar rendimiento futuro.

Comparamos también con una referencia que aprende la media por género solo en train y con una ablación que retira **solo género**, conservando las otras catorce variables y los parámetros del ensamble. La sensibilidad de etiquetas cambia únicamente y de ajuste por mediana en los folds de train y evalúa contra la misma referencia first. Estas auditorías no cambian la selección ni el objetivo de test. Una dependencia fuerte del género exige prudencia porque la deduplicación conservó una sola etiqueta por ID.""")
    code("""display(audit_genre(data, ROOT))
display(audit_label_policy(data, ROOT))
display(json.loads((ROOT / 'reports/eva2/label_sensitivity.json').read_text(encoding='utf-8')))
for name in ['target_band', 'explicit', 'artist_seen']:
    print('Errores por', name)
    display(pd.read_csv(ROOT / f'reports/eva2/errors_by_{name}.csv'))
display(pd.read_csv(ROOT / 'reports/eva2/errors_by_track_genre.csv')
        .query('n >= 30').sort_values('mae', ascending=False).head(10))
display(pd.read_csv(ROOT / 'reports/eva2/permutation_importance.csv'))""")
    md("""## 5. Evidencias visuales

La distribución y las correlaciones sustentan el EDA. Los residuos revelan sesgo hacia valores centrales y el error por género evita ocultar diferencias bajo el promedio. El codo y la silueta sustentan la elección de K. Los centroides permiten interpretar los perfiles en relación con train, y PCA muestra una proyección parcial de test.""")
    code("""make_plots(data, ROOT)
for name in ['01_popularity.png', '02_correlations.png', '03_model_comparison.png',
             '04_residuals.png', '05_importance.png', '06_k_selection.png',
             '07_pca_clusters.png', '08_cluster_profiles.png',
             '09_subgroups.png', '10_pca_variance.png']:
    display(Image(filename=str(ROOT / 'images/eva2' / name), width=850))""")
    md("""## 6. Inferencia local y conclusiones

Creamos una entrada de tres canciones de test sin el objetivo real. La inferencia valida las 15 variables y usa el pipeline persistido. Esta demostración comprueba el funcionamiento de entrada/salida y no es una nueva validación independiente.

Uso recomendado: exploración con revisión humana. Popularidad no equivale a mérito. Faltan fechas, contexto promocional y muestreo representativo. No se justifican decisiones automáticas de inversión. Todos los integrantes deben poder explicar los modelos, los errores y la discrepancia entre codo y silueta.""")
    code("""example = data['clean'].iloc[data['test_idx'][:3]][['track_id'] + MODEL_FEATURES]
(ROOT / 'data/examples').mkdir(exist_ok=True)
example.to_csv(ROOT / 'data/examples/tracks_example.csv', index=False)
example_result = predict(example, ROOT)
example_result.to_csv(ROOT / 'reports/eva2/example_predictions.csv', index=False)
display(example_result)
paths = build_reports(ROOT)
print('Tiempo de cómputo del cuaderno (s):', round(time.perf_counter() - start, 2))
print('Informes actualizados:', paths)
display(Markdown((ROOT / 'docs/informe_ejecutivo.md').read_text(encoding='utf-8')))""")
    md("""## 7. KPIs de negocio y piloto propuesto

El piloto no se ha ejecutado. El plan propone dos semanas y doce pares exploratorios, con metas sujetas a aprobación de curaduría: reducción mediana del tiempo por par >=15%, aceptación agregada >=70% y delta mediano de cobertura de géneros >=-5 pp. Define responsables, condiciones comparables y reglas para continuar, ajustar o detener. Son objetivos de validación, no beneficios comerciales medidos.

La calculadora `src/eva2/pilot.py` usa tiempos normalizados por 100 canciones, aceptación como razón de sumas y cobertura según géneros disponibles/cupos. Sus pruebas son sintéticas. Ver `docs/plan_piloto_negocio.md`. La plantilla siguiente debe permanecer vacía hasta recoger sesiones reales.""")
    code("""from src.eva2.pilot import pilot_metrics
pilot_template = pd.read_csv(ROOT / 'data/examples/pilot_template.csv')
display(pilot_template)
assert pilot_template.empty
try:
    pilot_metrics(pilot_template)
except ValueError as error:
    print('Control esperado, sin inventar mediciones:', error)""")
    md("""## Fuentes y reproducción

Pauta y ocho PDF de clases entregados por el equipo, experiencia 1 y documentación primaria de scikit-learn. Referencias, enlaces y limitaciones de procedencia en `docs/fuentes_eva2.md`. El informe técnico contiene la matriz IE5/IE6/IE7/IE8 y las fórmulas.

Para verificar desde terminal: `python -m unittest discover -s tests -v` y `python scripts/verify_delivery.py`. La evidencia efectiva queda en `reports/eva2/verification.json`. No se asigna una calificación automáticamente.""")
    nb.cells = cells
    nbf.validate(nb)
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    nb = build()
    dest = ROOT / "notebooks/05_proyecto_completo_eva2.ipynb"
    if args.execute:
        runtime = ROOT / "tmp/jupyter"
        runtime.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"JUPYTER_RUNTIME_DIR": str(runtime), "IPYTHONDIR": str(runtime / "ipython"),
                    "MPLCONFIGDIR": str(runtime / "matplotlib")})
        kernel_dir = runtime / "kernels/spotify-eva2"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Spotify EVA2 (entorno actual)", "language": "python",
        }), encoding="utf-8")
        client = NotebookClient(nb, timeout=600, kernel_name="spotify-eva2",
                                resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
                                on_cell_start=lambda cell, cell_index, **kwargs: print(
                                    f"Celda {cell_index + 1}/{len(nb.cells)}: {cell.cell_type}", flush=True))
        client.create_kernel_manager().kernel_spec_manager.kernel_dirs = [str(runtime / "kernels")]
        client.execute(env=env)
    nbf.write(nb, dest)
    print(dest)


if __name__ == "__main__":
    main()
