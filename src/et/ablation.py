"""Aislar el aporte del historial de artista y álbum en el HGB congelado."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.eva2.data import save_json
from src.eva3.contrasts import _aligned_predictions, paired_group_intervals
from src.eva3.data import ROOT, prepare
from src.eva3.modeling import CatalogDesign, FOLDS, HISTORY_KEYS, MODEL_HGB, SEED, _hgb, clip, metrics
from src.features.prepare_features import MODEL_FEATURES

WITHOUT_HISTORY = "sin_historial_artista_album"
WITH_HISTORY = "con_historial_artista_album"


def _fit_pair(train, valid, y, groups, params, seed):
    design = CatalogDesign()
    train_views, valid_views = design.fit_transform(train, y, groups, valid, seed)
    start = train_views["tree"].shape[1] - 4 * len(HISTORY_KEYS)
    # Cada clave aporta media, mediana, proporción de ceros y conteo suavizados.
    genre_start = start + 4 * list(HISTORY_KEYS).index("genre")
    keep = np.r_[0:start, genre_start:genre_start + 4]
    return {
        WITHOUT_HISTORY: clip(_hgb(params).fit(train_views["tree"][:, keep], y)
                              .predict(valid_views["tree"][:, keep])),
        WITH_HISTORY: clip(_hgb(params).fit(train_views["tree"], y).predict(valid_views["tree"])),
    }


def _assert_matches(reference, frame, groups):
    if (not np.array_equal(reference.popularity, frame.popularity)
            or not np.array_equal(reference.group_id, groups)):
        raise ValueError("La ablación debe comparar las mismas etiquetas y componentes de EP3.")


def audit_ablation(data: dict, ep3: Path, out: Path, *, n_threads: int = 4) -> dict:
    """Reajusta dos HGB de parámetros fijos en las particiones publicadas."""
    selection = json.loads((ep3 / "selection.json").read_text(encoding="utf-8"))
    if selection["selected_model"] != MODEL_HGB or selection["test_used_for_selection"] is not False:
        raise ValueError("Se esperaba el HGB ajustado congelado antes de abrir la reserva.")
    params = selection["best_params"][MODEL_HGB]
    manifest = pd.read_csv(ep3 / "split_manifest.csv")
    clean = data["clean"]
    train = clean.iloc[data["train_idx"]].reset_index(drop=True)
    test = clean.iloc[data["test_idx"]].reset_index(drop=True)
    train_groups = data["groups"].iloc[data["train_idx"]].to_numpy()
    test_groups = data["groups"].iloc[data["test_idx"]].to_numpy()
    if set(train_groups) & set(test_groups):
        raise ValueError("Un componente cruza desarrollo y reserva.")
    y = train.popularity.to_numpy(dtype=float)
    y_test = test.popularity.to_numpy(dtype=float)
    oof = _aligned_predictions(ep3 / "oof_predictions.csv", manifest, "train").set_index("track_id").loc[train.track_id]
    saved_test = _aligned_predictions(ep3 / "test_predictions.csv", manifest, "test").set_index("track_id").loc[test.track_id]
    _assert_matches(oof, train, train_groups)
    _assert_matches(saved_test, test, test_groups)
    folds = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_rows = []
    with threadpool_limits(limits=n_threads):
        for fold, (fit, valid) in enumerate(folds.split(train, y, train_groups), start=1):
            if not oof.iloc[valid].fold.eq(fold).all():
                raise ValueError("Los folds reproducidos difieren del registro OOF de EP3.")
            pair = _fit_pair(train.iloc[fit], train.iloc[valid], y[fit], train_groups[fit], params, SEED + fold)
            np.testing.assert_allclose(pair[WITH_HISTORY], oof.iloc[valid][MODEL_HGB], atol=1e-8, rtol=0)
            fold_rows.append({"fold": fold, "train_rows": len(fit), "validation_rows": len(valid),
                              **{f"mae_{name}": metrics(y[valid], pred)["mae"] for name, pred in pair.items()}})
        pair = _fit_pair(train, test, y, train_groups, params, SEED)
    np.testing.assert_allclose(pair[WITH_HISTORY], saved_test[MODEL_HGB], atol=1e-8, rtol=0)

    per_fold = pd.DataFrame(fold_rows)
    per_fold["mae_reduction_with_history"] = per_fold[f"mae_{WITHOUT_HISTORY}"] - per_fold[f"mae_{WITH_HISTORY}"]
    deltas = np.abs(y_test - pair[WITHOUT_HISTORY]) - np.abs(y_test - pair[WITH_HISTORY])
    interval = paired_group_intervals(pd.Series(test_groups), deltas[:, None])[0]
    artist_seen = saved_test.artist_seen.to_numpy(dtype=bool)
    summary = {
        "protocol": "Ablación retrospectiva. Mantiene el género, su historial y todas las entradas estáticas.",
        "selected_model": MODEL_HGB,
        "used_for_selection": False,
        "reference_reproduced_at_atol": 1e-8,
        "parameters": params,
        "static_features_kept": MODEL_FEATURES,
        "history_keys_removed": [key for key in HISTORY_KEYS if key != "genre"],
        "variants": {
            name: {"cv_mae": float(per_fold[f"mae_{name}"].mean()), "reserve": metrics(y_test, pred)}
            for name, pred in pair.items()
        },
        "reserve_rows": len(test),
        "reserve_mae_reduction_with_history": float(deltas.mean()),
        "reserve_reduction_ci95": interval.tolist(),
        "reserve_bootstrap_replicates": 800,
        "by_artist_history": {
            label: {"n": int(mask.sum()), "mae": {
                name: metrics(y_test[mask], pred[mask])["mae"] for name, pred in pair.items()
            }} for label, mask in (("visto", artist_seen), ("no_visto", ~artist_seen))
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out / "ablation_folds.csv", index=False, lineterminator="\n")
    save_json(out / "ablation.json", summary)
    return summary


def main() -> None:
    out = ROOT / "reports/et"
    with TemporaryDirectory(prefix="ablation-et-") as prepared:
        data = prepare(ROOT, Path(prepared))
        summary = audit_ablation(data, ROOT / "reports/eva3", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
