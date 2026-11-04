"""Figuras del informe técnico, elaboradas solo desde evidencia registrada."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "hgb_metadatos_ajustado": "HGB ajustado",
    "hgb_metadatos_base": "HGB fijo",
    "mezcla_calibrada": "Mezcla calibrada",
    "mezcla_uniforme": "Mezcla uniforme",
    "extra_trees_metadatos": "ExtraTrees",
    "arbol_metadatos": "Primer árbol de ExtraTrees",
    "ridge_metadatos": "Ridge",
    "hgb_ep2": "HGB de EP2 reentrenado",
    "dummy_mediana": "Mediana global",
}

def make_plots(data: dict, out: Path, images: Path) -> None:
    images.mkdir(parents=True, exist_ok=True)
    green, blue, gold = "#087f5b", "#345c90", "#cc7300"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#243647"}):
        def save(fig, name):
            fig.savefig(images / name, dpi=160, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        clean = data["clean"]
        fig, ax = plt.subplots(figsize=(8.3, 3.9), layout="constrained")
        bins = np.arange(-0.5, 105.5, 5)
        for indices, name, color in [(data["train_idx"], "Desarrollo", blue), (data["test_idx"], "Reserva", green)]:
            ax.hist(clean.iloc[indices].popularity, bins=bins, density=True, histtype="step",
                    linewidth=2, label=name, color=color)
        ax.set(xlabel="Popularidad observada (0 a 100)", ylabel="Densidad",
               title="Distribución del objetivo por partición")
        ax.legend(frameon=False)
        save(fig, "01_objetivo.png")

        metrics = pd.read_csv(out / "cv_summary.csv").sort_values("cv_mae", ascending=False)
        fig, ax = plt.subplots(figsize=(9.4, max(4, len(metrics) * .41)), layout="constrained")
        ax.barh(metrics.model.map(LABELS), metrics.cv_mae, color=[green if i == len(metrics) - 1 else blue
                                                       for i in range(len(metrics))])
        ax.bar_label(ax.containers[0], fmt="%.2f", padding=4, fontsize=9)
        ax.set(xlim=(0, metrics.cv_mae.max() * 1.17), xlabel="MAE en validación cruzada agrupada (puntos)",
               title="Comparación en desarrollo antes de abrir la reserva")
        save(fig, "02_modelos_cv.png")

        preds = pd.read_csv(out / "test_predictions.csv")
        absolute = preds.prediction.sub(preds.popularity).abs()
        grid = np.arange(0, 31)
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
        axes[0].plot(grid, [100 * absolute.le(t).mean() for t in grid], color=green, linewidth=2)
        axes[0].axhline(80, color=gold, linestyle="--", label="Referencia de 80%")
        axes[0].axvline(15, color=blue, linestyle=":", label="Tolerancia propuesta")
        coverage = 100 * absolute.le(15).mean()
        axes[0].scatter([15], [coverage], color=green, zorder=5)
        axes[0].annotate(f"{coverage:.2f}%".replace(".", ","), (15, coverage), xytext=(8, -14),
                         textcoords="offset points", color=green, fontweight="bold")
        axes[0].set(xlabel="Tolerancia absoluta (puntos)", ylabel="Predicciones dentro del margen (%)",
                    ylim=(0, 100), title="Cobertura según tolerancia")
        axes[0].legend(frameon=False, loc="lower right", fontsize=8)
        sample = preds.sample(min(3000, len(preds)), random_state=3103)
        axes[1].scatter(sample.popularity, sample.prediction, s=7, color=blue, alpha=.22, edgecolors="none")
        axes[1].plot([0, 100], [0, 100], color=gold, linestyle="--")
        axes[1].set(xlabel="Valor observado", ylabel="Predicción", xlim=(0, 100), ylim=(0, 100),
                    title="Predicción y valor observado (reserva)")
        save(fig, "03_reserva.png")

        diag = pd.read_csv(out / "clustering_diagnostics.csv")
        coords = pd.read_csv(out / "pca_test_coordinates.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4), layout="constrained")
        axes[0].plot(diag.k, diag.silhouette, color=green, marker="o")
        axes[0].set(xlabel="Número de segmentos K", ylabel="Silueta en desarrollo", xticks=range(2, 11),
                    title="Diagnóstico de K-Means")
        sample = coords.sample(min(3000, len(coords)), random_state=3103)
        for key, group in sample.groupby("cluster"):
            axes[1].scatter(group.PC1, group.PC2, s=7, alpha=.3, label=f"Grupo {key}", edgecolors="none")
        axes[1].set(xlabel="PC1", ylabel="PC2", title="Proyección de la reserva (PCA)")
        axes[1].legend(frameon=False, markerscale=2)
        save(fig, "04_segmentos.png")
