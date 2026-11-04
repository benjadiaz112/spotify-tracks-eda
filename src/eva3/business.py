"""Auditar el error y simular un presupuesto editorial fijo sobre la reserva."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.eva2.data import save_json

SEED = 3103
TOLERANCES = (5, 10, 15, 20)
BUDGET_FRACTION = 0.20
HIGH_QUANTILE = 0.75


def simulate_shortlist(frame: pd.DataFrame, score: str, budget: int, threshold: float) -> dict:
    """Ranking determinista; el umbral real lo define desarrollo."""
    top = frame.sort_values([score, "track_id"], ascending=[False, True]).head(budget)
    n_high = int(frame.popularity.ge(threshold).sum())
    hits = int(top.popularity.ge(threshold).sum())
    return {
        "songs": len(top), "hits": hits, "precision_at_budget": hits / len(top),
        "recall_at_budget": hits / n_high if n_high else None,
        "mean_popularity": float(top.popularity.mean()),
        "genres": int(top.track_genre.nunique()),
        "genres_total_holdout": int(frame.track_genre.nunique()),
    }


def grouped_bootstrap(frame: pd.DataFrame, *, n_boot: int = 600) -> dict:
    """Intervalos percentiles, condicionales al ajuste, re-muestreando grupos enteros."""
    abs_error = frame.prediction.sub(frame.popularity).abs()
    values = pd.DataFrame({"group": frame.audio_group, "n": 1, "mae": abs_error})
    for tolerance in TOLERANCES:
        values[f"within_{tolerance}"] = abs_error.le(tolerance).astype(int)
    if "baseline_median" in frame:
        values["mae_gain_baseline"] = frame.baseline_median.sub(frame.popularity).abs() - abs_error
    values["sse"] = abs_error.pow(2)
    values["sum_y"] = frame.popularity
    values["sum_y2"] = frame.popularity.astype(float).pow(2)
    by_group = values.groupby("group", sort=True).sum()
    sums = by_group.to_numpy(dtype=float)
    if len(sums) < 2:
        raise ValueError("Se necesitan dos o más grupos para estimar incertidumbre.")
    rng = np.random.default_rng(SEED)
    def scores(total):
        n, *_, sse, sum_y, sum_y2 = total
        sst = sum_y2 - sum_y ** 2 / n
        if sst <= 0:
            raise ValueError("La muestra bootstrap no tiene variabilidad del objetivo.")
        return np.r_[total[1:-3] / n, 1 - sse / sst]

    reps = np.empty((n_boot, sums.shape[1] - 3))
    for i in range(n_boot):
        total = sums[rng.integers(0, len(sums), size=len(sums))].sum(axis=0)
        reps[i] = scores(total)
    point = scores(sums.sum(axis=0))
    return {
        "seed": SEED, "resamples": n_boot, "groups": len(sums),
        "unit": "grupo de canciones relacionadas, no fila",
        "interpretation": "Incertidumbre de muestreo en la reserva del mismo origen, no garantía temporal.",
        "intervals": {
            col: {"point": float(point[i]),
                  "lo_95": float(np.quantile(reps[:, i], 0.025)),
                  "hi_95": float(np.quantile(reps[:, i], 0.975))}
            for i, col in enumerate([*by_group.columns[1:-3], "r2"])
        },
    }


def audit_business(data: dict, out: Path) -> dict:
    predictions = pd.read_csv(out / "test_predictions.csv", dtype={"track_id": str, "audio_group": str})
    clean, train, holdout = data["clean"], data["train_idx"], data["test_idx"]
    truth = clean.iloc[holdout].assign(_component=data["groups"].iloc[holdout].to_numpy()).set_index("track_id")
    if not predictions.track_id.is_unique or set(predictions.track_id) != set(truth.index):
        raise ValueError("Las predicciones no corresponden al conjunto reservado.")
    aligned = truth.loc[predictions.track_id]
    if not np.array_equal(predictions.popularity.to_numpy(), aligned.popularity.to_numpy()):
        raise ValueError("Los objetivos guardados difieren del conjunto reservado.")
    if not np.array_equal(predictions.audio_group.to_numpy(), aligned._component.astype(str).to_numpy()):
        raise ValueError("Los grupos de predicción difieren de la partición reservada.")
    predictions["track_genre"] = aligned.track_genre.to_numpy()
    predictions["explicit"] = aligned.explicit.to_numpy()
    train_frame = clean.iloc[train]
    if "artist_seen" not in predictions:
        predictions["artist_seen"] = aligned.artists.isin(train_frame.artists).to_numpy()
    if "album_seen" not in predictions:
        known = set(zip(train_frame.artists, train_frame.album_name))
        predictions["album_seen"] = [pair in known for pair in zip(aligned.artists, aligned.album_name)]
    if not np.isfinite(predictions.prediction).all():
        raise ValueError("Predicciones no finitas.")
    predictions["target_band"] = pd.cut(predictions.popularity, [-1, 20, 40, 60, 80, 100], include_lowest=True)
    predictions["abs_error"] = predictions.prediction.sub(predictions.popularity).abs()
    predictions["residual"] = predictions.prediction.sub(predictions.popularity)
    rows = []
    for column in ("artist_seen", "album_seen", "track_genre", "explicit", "target_band"):
        for label, group in predictions.groupby(column, observed=True, dropna=False):
            rows.append({
                "dimension": column, "group": str(label), "n": len(group),
                "mae": float(group.abs_error.mean()), "bias_pred_minus_real": float(group.residual.mean()),
                "within_15": float(group.abs_error.le(15).mean()), "small_sample": len(group) < 30,
            })
    pd.DataFrame(rows).to_csv(out / "holdout_subgroups.csv", index=False)
    bootstrap = grouped_bootstrap(predictions)
    save_json(out / "bootstrap.json", bootstrap)
    by_artist = {
        str(seen): {"rows": len(group), **grouped_bootstrap(group)}
        for seen, group in predictions.groupby("artist_seen", observed=True)
    }
    save_json(out / "bootstrap_artist_seen.json", by_artist)

    threshold = float(train_frame.popularity.quantile(HIGH_QUANTILE))
    budget = max(1, int(BUDGET_FRACTION * len(predictions)))
    model = simulate_shortlist(predictions, "prediction", budget, threshold)
    reference = {
        "precision_expected": float(predictions.popularity.ge(threshold).mean()),
        "mean_popularity_expected": float(predictions.popularity.mean()),
        "explanation": "Valor esperado de una muestra uniforme del mismo tamaño, sin ranking.",
    }
    result = {
        "scope": "Simulación retrospectiva de una lista para revisión humana; no es un piloto comercial.",
        "budget_fraction": BUDGET_FRACTION, "budget_songs": budget,
        "threshold_quantile_train": HIGH_QUANTILE, "threshold_train": threshold,
        "selected_model": model, "uniform_reference": reference,
        "precision_gain_pp": 100 * (model["precision_at_budget"] - reference["precision_expected"]),
        "mean_popularity_gain_points": model["mean_popularity"] - reference["mean_popularity_expected"],
    }
    metrics = pd.read_csv(out / "supervised_metrics.csv")
    score_columns = [name for name in metrics.model.unique() if name in predictions]
    candidate_rows = [
        {"candidate": name, **simulate_shortlist(predictions, name, budget, threshold)}
        for name in score_columns
    ]
    pd.DataFrame(candidate_rows).to_csv(out / "business_candidates.csv", index=False)
    result["candidate_rankings"] = len(candidate_rows)
    save_json(out / "business.json", result)
    return result
