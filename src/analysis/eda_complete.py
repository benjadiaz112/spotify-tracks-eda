"""Genera tablas y graficos del EDA completo de Spotify Tracks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES = [
    "energy",
    "danceability",
    "loudness",
    "valence",
    "acousticness",
    "tempo",
    "duration_ms",
]

LABELS = {
    "energy": "Energía",
    "danceability": "Bailabilidad",
    "loudness": "Volumen (dB)",
    "valence": "Valencia",
    "acousticness": "Acústica",
    "tempo": "Tempo (BPM)",
    "duration_ms": "Duración (ms)",
    "popularity": "Popularidad",
}


def calculate_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Resume valores extremos usando la regla del rango intercuartil."""
    rows = []
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[column] < lower) | (df[column] > upper)).sum())
        rows.append(
            {
                "variable": column,
                "limite_inferior": round(lower, 4),
                "limite_superior": round(upper, 4),
                "valores_extremos": count,
                "porcentaje": round(count / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construye las tablas principales utilizadas en el informe."""
    correlations = (
        df[FEATURES + ["popularity"]]
        .corr(numeric_only=True)[["popularity"]]
        .drop(index="popularity")
        .rename(columns={"popularity": "correlacion_popularidad"})
        .sort_values("correlacion_popularidad", key=abs, ascending=False)
        .reset_index(names="variable")
    )

    genres = (
        df.groupby("track_genre")["popularity"]
        .agg(canciones="size", promedio="mean", mediana="median")
        .round(2)
        .sort_values("mediana", ascending=False)
        .reset_index()
    )

    explicit = (
        df.groupby("explicit")["popularity"]
        .agg(canciones="size", promedio="mean", mediana="median")
        .round(2)
        .reset_index()
    )

    top_tracks = df.nlargest(10, "popularity")[
        ["track_name", "artists", "track_genre", "popularity"]
    ].reset_index(drop=True)
    bottom_tracks = df.nsmallest(10, "popularity")[
        ["track_name", "artists", "track_genre", "popularity"]
    ].reset_index(drop=True)

    return {
        "correlations": correlations,
        "genres": genres,
        "explicit": explicit,
        "top_tracks": top_tracks,
        "bottom_tracks": bottom_tracks,
        "outliers": calculate_outliers(df, FEATURES + ["popularity"]),
    }


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_plots(df: pd.DataFrame, tables: dict[str, pd.DataFrame], image_dir: Path) -> None:
    """Crea todos los graficos del EDA y los guarda como PNG."""
    image_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df, x="popularity", bins=20, color="#1DB954", ax=axes[0])
    axes[0].axvline(df["popularity"].median(), color="#191414", linestyle="--", label="Mediana")
    axes[0].set(title="Distribución de popularidad", xlabel="Popularidad", ylabel="Canciones")
    axes[0].legend()
    sns.boxplot(data=df, x="popularity", color="#1DB954", ax=axes[1])
    axes[1].set(title="Dispersión de popularidad", xlabel="Popularidad")
    fig.suptitle("Popularidad de las canciones", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, image_dir / "01_distribucion_popularidad.png")

    genres = tables["genres"]
    genre_plot = pd.concat([genres.head(10), genres.tail(10)]).drop_duplicates("track_genre")
    genre_plot = genre_plot.sort_values("mediana")
    colors = ["#6c757d" if value < 33 else "#1DB954" for value in genre_plot["mediana"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(genre_plot["track_genre"], genre_plot["mediana"], color=colors)
    ax.axvline(df["popularity"].median(), color="#191414", linestyle="--", label="Mediana general")
    ax.set(title="Géneros con mayor y menor popularidad mediana", xlabel="Popularidad mediana", ylabel="Género")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, image_dir / "02_popularidad_por_genero.png")

    sample_box = df.sample(min(20_000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=sample_box, x="explicit", y="popularity", hue="explicit", legend=False, ax=axes[0])
    axes[0].set(title="Distribución según contenido explícito", xlabel="Contenido explícito", ylabel="Popularidad")
    explicit_plot = tables["explicit"].copy()
    explicit_plot["grupo"] = explicit_plot["explicit"].map({False: "No explícita", True: "Explícita"})
    sns.barplot(data=explicit_plot, x="grupo", y="promedio", hue="grupo", legend=False, ax=axes[1])
    axes[1].set(title="Popularidad promedio", xlabel="Tipo de canción", ylabel="Popularidad promedio")
    axes[1].set_ylim(0, max(explicit_plot["promedio"]) + 8)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f")
    fig.tight_layout()
    save_plot(fig, image_dir / "03_popularidad_explicit.png")

    sample = df.sample(min(6_000, len(df)), random_state=42).copy()
    sample["duration_minutes"] = sample["duration_ms"] / 60_000
    plot_features = ["energy", "danceability", "loudness", "valence", "acousticness", "tempo", "duration_minutes"]
    plot_labels = {**LABELS, "duration_minutes": "Duración (minutos)"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, feature in zip(axes.flat, plot_features):
        sns.regplot(
            data=sample,
            x=feature,
            y="popularity",
            scatter_kws={"alpha": 0.12, "s": 9, "color": "#1DB954"},
            line_kws={"color": "#191414"},
            ci=None,
            ax=ax,
        )
        ax.set(title=plot_labels[feature], xlabel=plot_labels[feature], ylabel="Popularidad")
    axes.flat[-1].axis("off")
    fig.suptitle("Relación entre características musicales y popularidad", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, image_dir / "04_relaciones_popularidad.png")

    corr_columns = FEATURES + ["popularity"]
    corr_matrix = df[corr_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True, ax=ax)
    ax.set_title("Matriz de correlación")
    ax.set_xticklabels([LABELS.get(label, label) for label in corr_matrix.columns], rotation=45, ha="right")
    ax.set_yticklabels([LABELS.get(label, label) for label in corr_matrix.index], rotation=0)
    fig.tight_layout()
    save_plot(fig, image_dir / "05_matriz_correlacion.png")

    outlier_data = df[FEATURES].copy()
    outlier_data["duration_ms"] = outlier_data["duration_ms"] / 60_000
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, feature in zip(axes.flat, FEATURES):
        sns.boxplot(x=outlier_data[feature], color="#1DB954", ax=ax)
        label = "Duración (minutos)" if feature == "duration_ms" else LABELS[feature]
        ax.set(title=label, xlabel=label)
    axes.flat[-1].axis("off")
    fig.suptitle("Detección visual de valores extremos", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, image_dir / "06_valores_extremos.png")


def run_full_eda(project_root: Path) -> dict[str, pd.DataFrame]:
    """Ejecuta el EDA completo y guarda todas las evidencias."""
    data_path = project_root / "data" / "processed" / "spotify_tracks_clean.csv"
    image_dir = project_root / "images" / "eda"
    report_dir = project_root / "reports" / "eda"
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    tables = build_tables(df)
    create_plots(df, tables, image_dir)

    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)

    return {"data": df, **tables}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    results = run_full_eda(project_root)
    df = results["data"]
    print(f"EDA completado para {len(df):,} canciones")
    print(f"Popularidad promedio: {df['popularity'].mean():.2f}")
    print(results["correlations"].to_string(index=False))


if __name__ == "__main__":
    main()
