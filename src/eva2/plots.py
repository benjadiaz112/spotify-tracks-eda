"""Gráficos reproducibles, etiquetados y legibles para informe y defensa."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT
from src.features.prepare_features import CONTINUOUS_FEATURES as NUM

GREEN, BLUE, ORANGE = "#16865a", "#2875a9", "#d98835"


def save(fig, name, root):
    dest = root / "images/eva2"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_plots(data: dict, root: Path = ROOT):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.constrained_layout.use": True})
    out, clean = root / "reports/eva2", data["clean"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for idx, label, color in [(data["train_idx"], "Entrenamiento", GREEN), (data["test_idx"], "Prueba", BLUE)]:
        ax.hist(clean.iloc[idx].popularity, bins=np.arange(0, 105, 5), density=True,
                histtype="step", linewidth=2, label=label, color=color)
    ax.set(title="Popularidad: distribución en la partición agrupada", xlabel="Popularidad (0 a 100)", ylabel="Densidad")
    ax.legend()
    save(fig, "01_popularity.png", root)

    corr = clean[NUM + ["popularity"]].corr()
    fig, ax = plt.subplots(figsize=(10.5, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.set_title("Correlaciones lineales: asociación, sin inferencia causal")
    fig.colorbar(im, ax=ax, shrink=.8, label="Correlación de Pearson")
    save(fig, "02_correlations.png", root)

    metrics = pd.read_csv(out / "supervised_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metrics))
    ax.bar(x - .18, metrics.cv_mae, .36, color=BLUE, label="CV agrupada (selección)")
    ax.bar(x + .18, metrics.mae, .36, color=GREEN, label="Test (evaluación final)")
    ax.axhline(15, color=ORANGE, linestyle="--", label="Meta previa MAE = 15")
    ax.set_xticks(x, metrics.model.str.replace("_", " "), rotation=12, ha="right")
    ax.set(ylabel="Error absoluto medio (puntos)", title="Comparación de modelos sobre las mismas particiones")
    ax.legend(fontsize=9)
    save(fig, "03_model_comparison.png", root)

    predictions = pd.read_csv(out / "test_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sample = predictions.sample(min(4000, len(predictions)), random_state=42)
    axes[0].scatter(sample.popularity, sample.prediction, s=7, alpha=.18, color=BLUE)
    axes[0].plot([0, 100], [0, 100], "--", color=ORANGE)
    axes[0].set(xlabel="Popularidad real", ylabel="Predicción", title="Real vs. predicho (muestra de test)")
    axes[1].scatter(sample.prediction, sample.residual, s=7, alpha=.18, color=GREEN)
    axes[1].axhline(0, linestyle="--", color=ORANGE)
    axes[1].set(xlabel="Predicción", ylabel="Residuo: real - predicho", title="Residuos: búsqueda de errores sistemáticos")
    save(fig, "04_residuals.png", root)

    imp = pd.read_csv(out / "permutation_importance.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(imp.feature, imp.mae_increase, xerr=imp["std"], color=GREEN)
    ax.set(xlabel="Aumento del MAE al permutar (puntos)", title="Importancia predictiva en test (2.500 filas, 4 repeticiones)")
    save(fig, "05_importance.png", root)

    diag = pd.read_csv(out / "clustering_diagnostics.csv")
    conf = json.loads((out / "clustering.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(diag.k, diag.inertia, "o-", color=BLUE)
    axes[0].axvline(conf["elbow_chord_k"], color=ORANGE, linestyle="--", label="Codo orientativo")
    axes[0].set(xlabel="K", ylabel="Inercia", title="Método del codo (K = 1..10)", xticks=range(1, 11))
    axes[0].legend()
    axes[1].plot(diag.k, diag.silhouette, "o-", color=GREEN)
    axes[1].axvline(conf["selected_k"], color=ORANGE, linestyle="--", label="K elegido")
    axes[1].set(xlabel="K", ylabel="Silueta media", title="Cohesión y separación (muestra fija de train)", xticks=range(2, 11))
    axes[1].legend()
    save(fig, "06_k_selection.png", root)

    coords = pd.read_csv(out / "pca_test_coordinates.csv").sample(min(5000, len(predictions)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, group in coords.groupby("cluster"):
        ax.scatter(group.PC1, group.PC2, s=8, alpha=.35, label=f"Grupo {label}")
    variance = pd.read_csv(out / "pca_variance.csv")
    ax.set(xlabel=f"PC1 ({variance.iloc[0].explained_variance_ratio:.1%} de varianza)",
           ylabel=f"PC2 ({variance.iloc[1].explained_variance_ratio:.1%} de varianza)",
           title="PCA sobre test: proyección de los perfiles musicales")
    ax.legend(markerscale=2)
    save(fig, "07_pca_clusters.png", root)

    centers = pd.read_csv(out / "cluster_centers_standardized.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(11, max(3, len(centers) * .6)))
    im = ax.imshow(centers, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(centers.columns)), centers.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(centers)), [f"Grupo {i}" for i in centers.index])
    ax.set_title("Centroides estandarizados respecto de entrenamiento")
    for i in range(len(centers)):
        for j in range(len(centers.columns)):
            ax.text(j, i, f"{centers.iloc[i,j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(centers.iloc[i,j]) > 1.1 else "black")
    fig.colorbar(im, ax=ax, label="Desviaciones estándar", shrink=.8)
    save(fig, "08_cluster_profiles.png", root)

    errors = pd.read_csv(out / "errors_by_track_genre.csv").query("n >= 30").sort_values("mae").tail(12)
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.barh(errors.track_genre, errors.mae, color=BLUE, label="Modelo")
    ax.scatter(errors.baseline_mae, errors.track_genre, color=ORANGE, label="Referencia media", zorder=3)
    ax.set(xlabel="MAE (puntos)", title="Doce géneros con mayor error en test (n >= 30)")
    ax.legend()
    save(fig, "09_subgroups.png", root)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(variance.component, variance.cumulative_variance, "o-", color=BLUE)
    ax.axhline(.8, color=ORANGE, linestyle="--", label="80% de varianza")
    ax.set(xlabel="Componentes PCA", ylabel="Varianza acumulada", ylim=(0, 1.04),
           title="Dos componentes sirven para visualizar, con pérdida de información")
    ax.legend()
    save(fig, "10_pca_variance.png", root)
