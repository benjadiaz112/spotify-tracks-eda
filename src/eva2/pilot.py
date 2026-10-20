"""Calcular KPIs de un piloto cuando existan observaciones reales."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["reviewer_alias", "pair_id", "method", "reviewed_tracks", "selected_tracks", "duration_seconds",
           "recommended_tracks", "accepted_recommendations", "selected_genres", "available_genres"]


def pilot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if df.empty:
        raise ValueError("Plantilla vacía: no hay piloto ejecutado ni resultados comerciales que calcular")
    if df[COLUMNS].isna().any().any():
        raise ValueError("El registro del piloto tiene valores faltantes")
    work = df[COLUMNS].copy()
    counts = COLUMNS[3:]
    for name in counts:
        work[name] = pd.to_numeric(work[name], errors="raise")
    if not np.isfinite(work[counts].to_numpy()).all():
        raise ValueError("Valores no finitos")
    if (work[counts] < 0).any().any() or not work.duration_seconds.gt(0).all():
        raise ValueError("Duración y contadores fuera de dominio")
    integer_counts = [name for name in counts if name != "duration_seconds"]
    if not work[integer_counts].mod(1).eq(0).all().all():
        raise ValueError("Los contadores deben ser enteros")
    if not work.method.isin(["manual", "assisted"]).all():
        raise ValueError("method debe ser manual o assisted")
    if work.loc[work.method.eq("manual"), ["recommended_tracks", "accepted_recommendations"]].ne(0).any().any():
        raise ValueError("La sesión manual no recibe recomendaciones del modelo")
    if not work.reviewed_tracks.gt(0).all() or not work.selected_tracks.gt(0).all():
        raise ValueError("Debe haber canciones revisadas y seleccionadas")
    if ((work.selected_tracks > work.reviewed_tracks) | (work.recommended_tracks > work.reviewed_tracks)
            | (work.accepted_recommendations > work.recommended_tracks)
            | (work.selected_genres > work.selected_tracks) | (work.selected_genres > work.available_genres)
            | (work.available_genres > work.reviewed_tracks)).any():
        raise ValueError("Contadores inconsistentes")
    if work.duplicated(["reviewer_alias", "pair_id", "method"]).any():
        raise ValueError("Sesiones duplicadas en el par")
    paired = work.groupby(["reviewer_alias", "pair_id"])
    if not paired.method.apply(lambda s: set(s) == {"manual", "assisted"}).all():
        raise ValueError("Cada par necesita una sesión manual y una asistida")
    if not paired.reviewed_tracks.nunique().eq(1).all() or not paired.selected_tracks.nunique().eq(1).all():
        raise ValueError("Cada par debe comparar igual cantidad revisada y seleccionada")
    work["minutes_per_100"] = work.duration_seconds / 60 / work.reviewed_tracks * 100
    capacity = np.minimum(work.available_genres, work.selected_tracks)
    if not (capacity > 0).all():
        raise ValueError("La capacidad de cobertura de géneros debe ser positiva")
    work["genre_coverage_pct"] = 100 * work.selected_genres / capacity
    manual = work[work.method.eq("manual")].set_index(["reviewer_alias", "pair_id"])
    assisted = work[work.method.eq("assisted")].set_index(["reviewer_alias", "pair_id"]).loc[manual.index]
    if not assisted.recommended_tracks.gt(0).all():
        raise ValueError("La sesión asistida debe registrar recomendaciones")
    return pd.DataFrame({
        "time_reduction_pct": 100 * (1 - assisted.minutes_per_100 / manual.minutes_per_100),
        "recommendation_acceptance_pct": 100 * assisted.accepted_recommendations / assisted.recommended_tracks,
        "genre_coverage_delta_pp": assisted.genre_coverage_pct - manual.genre_coverage_pct,
    }).reset_index()


def pilot_summary(df: pd.DataFrame) -> dict:
    pairs = pilot_metrics(df)
    assisted = df.loc[df.method.eq("assisted")]
    return {
        "pairs": len(pairs),
        "time_reduction_median_pct": float(pairs.time_reduction_pct.median()),
        "recommendation_acceptance_weighted_pct": float(
            100 * pd.to_numeric(assisted.accepted_recommendations).sum()
            / pd.to_numeric(assisted.recommended_tracks).sum()),
        "genre_coverage_median_delta_pp": float(pairs.genre_coverage_delta_pp.median()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observations = pd.read_csv(args.input)
    result = pilot_metrics(observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = pilot_summary(observations)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
