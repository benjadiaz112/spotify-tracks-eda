"""Contrastes pareados de EP3: diagnóstico después de congelar la selección."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eva2.data import save_json
from .data import SEED

_HGB_FIXED = "hgb_metadatos_base"
_HGB_TUNED = "hgb_metadatos_ajustado"
_TREE = "arbol_metadatos"
_BAG = "extra_trees_metadatos"
_BLEND = "mezcla_uniforme"
_CALIBRATED = "mezcla_calibrada"
_EP2 = "hgb_ep2"
_FIXED_COMPONENTS = {
    "extra_trees_fijo": (_BAG, "bag_fixed_params"),
    "ridge_fijo": ("ridge_metadatos", "ridge_fixed_params"),
}


def _aligned_predictions(path: Path, manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Comprueba cobertura, etiquetas y grupos contra la partición declarada."""
    predictions = pd.read_csv(path, dtype={"track_id": str, "group_id": str, "audio_group": str})
    required = {"track_id", "popularity"} | ({"fold"} if split == "train" else set())
    if missing := required - set(predictions.columns):
        raise ValueError(f"Faltan columnas en {path.name}: {sorted(missing)}")
    truth = manifest.loc[manifest.split.eq(split)].set_index("track_id")
    if predictions.track_id.duplicated().any() or set(predictions.track_id) != set(truth.index):
        raise ValueError(f"IDs de {path.name} no corresponden a la partición {split}")
    truth = truth.loc[predictions.track_id]
    if not np.array_equal(predictions.popularity.to_numpy(), truth.popularity.to_numpy()):
        raise ValueError(f"Etiquetas de {path.name} no coinciden con el manifiesto")
    # La fuente de verdad es la componente unida, no la huella de audio original.
    predictions["group_id"] = truth.group_id.to_numpy()
    if split == "train":
        fold = predictions.fold.to_numpy(dtype=float)
        if not np.isfinite(fold).all() or (fold < 1).any() or (fold != np.floor(fold)).any():
            raise ValueError("Asignaciones OOF de fold inválidas")
        predictions["fold"] = fold.astype(int)
        if predictions.groupby("group_id").fold.nunique().max() != 1:
            raise ValueError("Un grupo cruza folds OOF")
    return predictions


def _errors(predictions: pd.DataFrame, model: str) -> np.ndarray:
    values = predictions[model].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError(f"Predicciones inválidas para {model}")
    return np.abs(predictions.popularity.to_numpy(dtype=float) - values)


def _fold_scores(path: Path, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    table = pd.read_csv(path)
    counts = oof.groupby("fold").size().sort_index()
    scores = table.pivot(index="fold", columns="model", values="mae").sort_index()
    if len(counts) < 2 or not counts.index.equals(scores.index) or scores.isna().any().any():
        raise ValueError("La tabla CV no usa los mismos folds completos que las predicciones OOF")
    if "val_rows" in table and not np.array_equal(table.val_rows.to_numpy(), table.fold.map(counts).to_numpy()):
        raise ValueError("Los tamaños de validación CV no coinciden con las predicciones OOF")
    for model in scores.columns:
        if model not in oof:
            raise ValueError(f"Faltan predicciones OOF de {model}")
        observed = pd.Series(_errors(oof, model)).groupby(oof.fold).mean().reindex(counts.index)
        if not np.allclose(scores[model], observed, rtol=1e-9, atol=1e-10):
            raise ValueError(f"El MAE CV de {model} no se reproduce con sus predicciones OOF")
    return scores, counts


def _add_fixed_scores(out: Path, scores: pd.DataFrame, counts: pd.Series, selection: dict) -> None:
    """Recupera referencias fijas del ensayo exacto guardado, sin reentrenar ni sustituir."""
    missing = set(selection["blend_weights"][_CALIBRATED]) - set(scores.columns)
    if not missing:
        return
    trials = pd.read_csv(out / "search_trials.csv")
    for slug in sorted(missing):
        if slug not in _FIXED_COMPONENTS:
            raise ValueError(f"No se reconoce la referencia fija {slug}")
        family, param_key = _FIXED_COMPONENTS[slug]
        family_trials = trials.loc[trials.model.eq(family)]
        exact = family_trials.loc[family_trials.params.map(json.loads).map(lambda p: p == selection[param_key])]
        rows = exact.set_index("fold").sort_index()
        if rows.index.has_duplicates or not rows.index.equals(counts.index):
            raise ValueError(f"No hay un único ensayo de parámetros fijos para {slug} por fold")
        if not np.array_equal(rows.val_rows.to_numpy(), counts.to_numpy()):
            raise ValueError(f"El ensayo fijo de {slug} utiliza otros tamaños de fold")
        if not np.isfinite(rows.mae.to_numpy(dtype=float)).all():
            raise ValueError(f"El ensayo fijo de {slug} contiene MAE no finito")
        scores[slug] = rows.mae


def _best_component(scores: pd.DataFrame, selection: dict, blend: str) -> str:
    components = selection["blend_weights"][blend]
    if not components or not set(components).issubset(scores.columns):
        raise ValueError(f"No hay componentes comparables para {blend}")
    return min(components, key=lambda name: (float(scores[name].mean()), name))


def _reference_column(test: pd.DataFrame, reference: str, selection: dict) -> tuple[str | None, str]:
    if reference in test:
        return reference, "prediccion_guardada"
    if reference in _FIXED_COMPONENTS:
        family, param_key = _FIXED_COMPONENTS[reference]
        # El entrenamiento comparte el objeto ajustado si su configuración fija
        # coincide con la ganadora de esa familia; en otro caso no son equivalentes.
        fixed_params = selection.get(param_key)
        chosen_params = selection.get("best_params", {}).get(family)
        if fixed_params is not None and chosen_params == fixed_params and family in test:
            return family, "misma_configuracion_fija_y_ajustada"
        return None, "referencia_fija_sin_prediccion_guardada"
    raise ValueError(f"Faltan predicciones de la referencia {reference}")


def paired_group_intervals(groups: pd.Series, deltas: np.ndarray, n_boot: int = 800) -> np.ndarray:
    """Percentiles del delta medio por canción re-muestreando componentes enteras."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 2 or len(groups) != deltas.shape[0] or not np.isfinite(deltas).all():
        raise ValueError("Deltas pareados inválidos")
    if groups.isna().any() or groups.nunique() < 2 or n_boot < 2:
        raise ValueError("Se requieren dos componentes válidas y dos réplicas o más")
    code, unique = pd.factorize(groups, sort=True)
    n_groups = len(unique)
    totals = np.column_stack([
        np.bincount(code, minlength=n_groups),
        *[np.bincount(code, weights=deltas[:, j], minlength=n_groups) for j in range(deltas.shape[1])],
    ])
    rng = np.random.default_rng(SEED)
    replicates = np.empty((n_boot, deltas.shape[1]))
    for i in range(n_boot):
        sample = totals[rng.integers(0, n_groups, n_groups)].sum(axis=0)
        replicates[i] = sample[1:] / sample[0]
    return np.quantile(replicates, [0.025, 0.975], axis=0).T


def audit_contrasts(out: Path, *, n_boot: int = 800) -> pd.DataFrame:
    """Compara predicciones guardadas sin entrenar ni cambiar ganador, umbrales o pesos."""
    out = Path(out)
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("La selección debe estar congelada antes de abrir la reserva")
    manifest = pd.read_csv(out / "split_manifest.csv", dtype={"track_id": str, "group_id": str})
    if manifest.track_id.isna().any() or manifest.track_id.duplicated().any() or manifest.group_id.isna().any():
        raise ValueError("El manifiesto de EP3 requiere IDs únicos y grupos válidos")
    if set(manifest.split) != {"train", "test"} or manifest.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("El manifiesto de EP3 no contiene dos conjuntos disjuntos por grupo")
    oof = _aligned_predictions(out / "oof_predictions.csv", manifest, "train")
    scores, counts = _fold_scores(out / "cv_folds.csv", oof)
    _add_fixed_scores(out, scores, counts, selection)

    # Las referencias de las mezclas se resuelven antes de cargar la reserva.
    specs = [
        ("ajuste_hgb", _HGB_TUNED, _HGB_FIXED,
         "Mismo boosting y predictores: variante ajustada frente a configuración fija de EP3."),
        ("ensamble_arboles", _BAG, _TREE,
         "ExtraTrees con configuración congelada frente a su primer árbol exacto, no un árbol de otra familia."),
        ("mezcla_uniforme_vs_componente", _BLEND, _best_component(scores, selection, _BLEND),
         "Mezcla uniforme frente al componente ajustado de menor MAE medio en CV."),
        ("mezcla_calibrada_vs_componente", _CALIBRATED, _best_component(scores, selection, _CALIBRATED),
         "Mezcla calibrada frente al componente fijo de menor MAE medio en CV."),
        ("ganador_vs_ep2", selection["selected_model"], _EP2,
         "Ganador congelado frente a la configuración de EP2 reentrenada en el mismo protocolo EP3; no aísla una sola intervención."),
    ]
    if any(model not in scores for _, candidate, reference, _ in specs for model in (candidate, reference)):
        raise ValueError("Algún contraste no cuenta con las dos referencias completas en CV")
    test = _aligned_predictions(out / "test_predictions.csv", manifest, "test")

    rows, fold_rows, bootstrap_rows, deltas = [], [], [], []
    for contrast, candidate, reference, basis in specs:
        cv_candidate, cv_reference = scores[candidate], scores[reference]
        cv_delta = cv_reference - cv_candidate
        for fold in counts.index:
            fold_rows.append({
                "contrast": contrast, "fold": int(fold), "val_rows": int(counts.loc[fold]),
                "candidate": candidate, "reference": reference,
                "mae_candidate": float(cv_candidate.loc[fold]), "mae_reference": float(cv_reference.loc[fold]),
                "delta_mae_points": float(cv_delta.loc[fold]),
            })
        candidate_error = _errors(test, candidate)
        column, status = _reference_column(test, reference, selection)
        row = {
            "contrast": contrast, "candidate": candidate, "reference": reference, "basis": basis,
            "cv_folds": len(counts), "cv_mae_candidate": float(cv_candidate.mean()),
            "cv_mae_reference": float(cv_reference.mean()), "cv_delta_mae_points": float(cv_delta.mean()),
            "cv_positive_folds": int(cv_delta.gt(0).sum()), "cv_delta_min": float(cv_delta.min()),
            "cv_delta_max": float(cv_delta.max()), "cv_delta_std": float(cv_delta.std(ddof=0)),
            "oof_delta_mae_points": float(np.average(cv_delta, weights=counts)),
            "holdout_rows": len(test), "holdout_groups": int(test.group_id.nunique()),
            "holdout_mae_candidate": float(candidate_error.mean()),
            "holdout_reference_prediction_column": column, "holdout_status": status,
            "holdout_mae_reference": None, "holdout_delta_mae_points": None,
            "holdout_relative_gain_pct": None, "holdout_delta_ci95_low": None,
            "holdout_delta_ci95_high": None, "holdout_ci_includes_zero": None,
        }
        if column is not None:
            reference_error = _errors(test, column)
            reference_mae = float(reference_error.mean())
            difference = reference_error - candidate_error
            gain = float(difference.mean())
            row.update({"holdout_mae_reference": reference_mae, "holdout_delta_mae_points": gain,
                        "holdout_relative_gain_pct": 100 * gain / reference_mae if reference_mae else None})
            deltas.append(difference)
            bootstrap_rows.append(row)
        rows.append(row)

    intervals = paired_group_intervals(test.group_id, np.column_stack(deltas), n_boot)
    for row, (lo, hi) in zip(bootstrap_rows, intervals):
        row.update({"holdout_delta_ci95_low": float(lo), "holdout_delta_ci95_high": float(hi),
                    "holdout_ci_includes_zero": bool(lo <= 0 <= hi)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "controlled_contrasts.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(out / "controlled_contrasts_folds.csv", index=False, lineterminator="\n")
    save_json(out / "controlled_contrasts.json", {
        "selected_model_unchanged": selection["selected_model"],
        "used_holdout_for_reselection": False,
        "delta_definition": "MAE_referencia - MAE_candidato; positivo favorece al candidato (puntos de popularidad)",
        "cv_aggregation": "Media simple de deltas pareados por fold; OOF pondera por filas de validación.",
        "bootstrap": {"seed": SEED, "resamples": n_boot, "unit": "componente EP3", "level": 0.95,
                      "scope": "Intervalos percentiles exploratorios condicionados a modelos ajustados y a esta reserva; sin corrección por comparaciones múltiples ni inferencia temporal."},
        "contrasts": rows,
    })
    return result
