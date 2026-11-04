"""Búsqueda, ensambles y prueba final agrupada para la regresión de popularidad."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.eva2.data import save_json
from src.eva2.supervised import hgb_pipeline, preprocessor
from src.features.prepare_features import CONTINUOUS_FEATURES as NUM, CATEGORICAL_FEATURES as CAT, MODEL_FEATURES

SEED = 3103
FOLDS = 5
HISTORY_FOLDS = 4
TOLERANCES = (5, 10, 15, 20)
FEATURES_META = [*MODEL_FEATURES, "artists", "album_name"]
HISTORY_KEYS = {
    "genre": 12.0, "artist_primary": 5.0, "artist_full": 5.0,
    "album": 4.0, "artist_album": 2.0, "artist_genre": 4.0,
}
RIDGE_CATEGORIES = [*CAT, "artist_primary", "artist_full", "artist_album", "album"]
MODEL_BASE = "dummy_mediana"
MODEL_EP2 = "hgb_ep2"
MODEL_HGB_BASE = "hgb_metadatos_base"
MODEL_HGB = "hgb_metadatos_ajustado"
MODEL_TREE = "arbol_metadatos"
MODEL_BAG = "extra_trees_metadatos"
MODEL_RIDGE = "ridge_metadatos"
MODEL_BLEND = "mezcla_uniforme"
MODEL_BLEND_CAL = "mezcla_calibrada"
COMPONENTS = (MODEL_HGB, MODEL_BAG, MODEL_RIDGE)
# Fijos y comunes a las comparaciones del ensamble calibrado. La calibración
# interna no puede heredar hiperparámetros elegidos con el fold externo.
CALIBRATION_COMPONENTS = (MODEL_HGB_BASE, "extra_trees_fijo", "ridge_fijo")
HGB_DEFAULT = dict(loss="squared_error", max_iter=200, learning_rate=0.08,
                   max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0)
HGB_SEARCH = [
    HGB_DEFAULT,
    dict(loss="squared_error", max_iter=330, learning_rate=0.06, max_leaf_nodes=47,
         min_samples_leaf=20, l2_regularization=1.0),
    dict(loss="squared_error", max_iter=400, learning_rate=0.07, max_leaf_nodes=63,
         min_samples_leaf=14, l2_regularization=2.0),
    dict(loss="squared_error", max_iter=420, learning_rate=0.055, max_leaf_nodes=31,
         min_samples_leaf=26, l2_regularization=8.0),
    dict(loss="absolute_error", max_iter=220, learning_rate=0.08, max_leaf_nodes=31,
         min_samples_leaf=20, l2_regularization=1.0),
    dict(loss="absolute_error", max_iter=340, learning_rate=0.07, max_leaf_nodes=63,
         min_samples_leaf=14, l2_regularization=2.0),
    dict(loss="absolute_error", max_iter=380, learning_rate=0.055, max_leaf_nodes=47,
         min_samples_leaf=24, l2_regularization=6.0),
    dict(loss="absolute_error", max_iter=420, learning_rate=0.08, max_leaf_nodes=63,
         min_samples_leaf=10, l2_regularization=0.5),
]
BAG_DEFAULT = dict(n_estimators=100, min_samples_leaf=5, max_features=0.8, max_samples=0.85)
BAG_SEARCH = [
    BAG_DEFAULT,
    dict(n_estimators=100, min_samples_leaf=3, max_features=0.85, max_samples=0.90),
    dict(n_estimators=100, min_samples_leaf=8, max_features=0.65, max_samples=0.85),
]
RIDGE_DEFAULT = dict(alpha=6.0)
RIDGE_SEARCH = [dict(alpha=2.0), RIDGE_DEFAULT, dict(alpha=18.0)]


def _normalized(series: pd.Series) -> pd.Series:
    return (series.astype("string").str.normalize("NFKC").str.casefold()
            .str.replace(r"\s+", " ", regex=True).str.strip().fillna("<sin dato>"))


def catalog_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Claves declaradas como metadatos disponibles en inferencia; nunca ID o título."""
    full = _normalized(frame.artists)
    primary = full.str.split(";", n=1).str[0].str.strip()
    album = _normalized(frame.album_name)
    genre = _normalized(frame.track_genre)
    return pd.DataFrame({
        "genre": genre, "artist_primary": primary, "artist_full": full, "album": album,
        "artist_album": primary + "\x1f" + album, "artist_genre": primary + "\x1f" + genre,
    }, index=frame.index)


class CatalogHistory:
    """Estadísticas suavizadas solo a partir de etiquetas del subconjunto de ajuste."""

    def fit(self, keys: pd.DataFrame, y):
        values = np.asarray(y, dtype=float)
        if not len(values) or len(values) != len(keys):
            raise ValueError("Historial vacío o desalineado")
        frame = keys.reset_index(drop=True).assign(y=values, zero=(values == 0).astype(float))
        self.global_ = np.array([values.mean(), np.median(values), np.mean(values == 0)])
        self.tables_ = {}
        for key, smoothing in HISTORY_KEYS.items():
            summary = frame.groupby(key, sort=False, observed=True).agg(
                n=("y", "size"), mean=("y", "mean"), median=("y", "median"), zero=("zero", "mean"),
            )
            n = summary["n"].to_numpy()[:, None]
            summary[["mean", "median", "zero"]] = (
                summary[["mean", "median", "zero"]].to_numpy() * n + smoothing * self.global_
            ) / (n + smoothing)
            self.tables_[key] = summary
        return self

    def transform(self, keys: pd.DataFrame) -> np.ndarray:
        result = []
        for key in HISTORY_KEYS:
            tab = self.tables_[key]
            labels = keys[key]
            result.extend(labels.map(tab[stat]).fillna(self.global_[i]).to_numpy(dtype=float)
                          for i, stat in enumerate(("mean", "median", "zero")))
            result.append(np.log1p(labels.map(tab["n"]).fillna(0).to_numpy(dtype=float)))
        return np.column_stack(result)


def crossfit_catalog_history(keys: pd.DataFrame, y, groups,
                             n_splits: int = HISTORY_FOLDS, seed: int = SEED):
    """Cada fila de ajuste recibe estadísticas que excluyen todo su grupo."""
    y, groups = np.asarray(y, dtype=float), np.asarray(groups)
    if len(keys) != len(groups) or len(keys) != len(y):
        raise ValueError("Grupos, claves y etiquetas desalineados")
    fold_count = min(n_splits, len(np.unique(groups)))
    if fold_count < 2:
        raise ValueError("Se requieren al menos dos grupos para codificar sin fuga")
    cv = GroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    encoded = np.full((len(y), 4 * len(HISTORY_KEYS)), np.nan)
    for train, valid in cv.split(keys, y, groups):
        if set(groups[train]) & set(groups[valid]):
            raise ValueError("Fuga en el historial cruzado")
        encoded[valid] = CatalogHistory().fit(keys.iloc[train], y[train]).transform(keys.iloc[valid])
    if not np.isfinite(encoded).all():
        raise ValueError("Codificación cruzada incompleta")
    return encoded, CatalogHistory().fit(keys, y)


class CatalogDesign:
    """Vistas para árboles y Ridge, ajustadas independientemente por fold."""

    def __init__(self):
        self.audio = preprocessor(True)
        self.ridge = ColumnTransformer([
            ("num", StandardScaler(), NUM), ("exp", "passthrough", ["explicit"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2,
                                  sparse_output=True), RIDGE_CATEGORIES),
        ], sparse_threshold=1.0)

    @staticmethod
    def ridge_input(frame, keys):
        return frame[MODEL_FEATURES].join(keys.drop(columns="genre"))

    def fit_transform(self, frame, y, groups, validation, seed):
        keys, keys_valid = catalog_keys(frame), catalog_keys(validation)
        hist, self.history = crossfit_catalog_history(keys, y, groups, seed=seed)
        x_audio = self.audio.fit_transform(frame[MODEL_FEATURES])
        x_ridge = self.ridge.fit_transform(self.ridge_input(frame, keys))
        return {"tree": np.column_stack([x_audio, hist]), "ridge": x_ridge}, self.transform(validation, keys_valid)

    def transform(self, frame, keys=None):
        keys = catalog_keys(frame) if keys is None else keys
        return {
            "tree": np.column_stack([self.audio.transform(frame[MODEL_FEATURES]), self.history.transform(keys)]),
            "ridge": self.ridge.transform(self.ridge_input(frame, keys)),
        }


class CatalogPredictor:
    """Predictor serializable. Las estadísticas de entrenamiento se reutilizan en inferencia."""

    def __init__(self, design, models, weights=None):
        self.design, self.models, self.weights = design, models, weights

    def predict(self, frame):
        matrices = self.design.transform(frame)
        predictions = np.column_stack([
            clip(model.predict(matrices[view])) for view, model in self.models
        ])
        return predictions[:, 0] if self.weights is None else predictions @ self.weights


def clip(values):
    return np.clip(np.asarray(values, dtype=float), 0, 100)


def metrics(y, prediction):
    prediction = clip(prediction)
    mse = float(mean_squared_error(y, prediction))
    error = np.abs(np.asarray(y) - prediction)
    return {
        "mae": float(error.mean()), "mse": mse, "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, prediction)),
        **{f"within_{t}": float(np.mean(error <= t)) for t in TOLERANCES},
    }


def _hgb(params, seed=SEED):
    return HistGradientBoostingRegressor(
        categorical_features=list(range(len(NUM), len(NUM) + len(CAT))),
        early_stopping=False, random_state=seed, **params,
    )


def _bag(params, threads, seed=SEED):
    return ExtraTreesRegressor(bootstrap=True, n_jobs=threads, random_state=seed, **params)


def _ridge(params):
    return Ridge(solver="lsqr", tol=1e-4, **params)


def _predict_tree_bag(train, valid, y, params, threads, seed=SEED):
    model = _bag(params, threads, seed).fit(train, y)
    # Exacto primer árbol del ensamble: mismos datos remuestreados, parámetros
    # y semilla. Aísla el efecto de promediar los demás árboles.
    return clip(model.predict(valid)), clip(model.estimators_[0].predict(valid))


def _optimize_weights(y, predictions):
    """Simplex fijo de pasos 0,1; empate favorece cercanía a pesos uniformes."""
    candidates = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
    y = np.asarray(y, dtype=float)
    return np.asarray(min(candidates, key=lambda w: (
        mean_absolute_error(y, predictions @ w), sum((v - 1 / 3) ** 2 for v in w), w,
    )))


def _calibration_fold(frame, y, groups, fold, threads):
    """Un holdback dentro de train externo: nunca toca el fold puntuado."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=SEED + fold)
    a, b = next(split.split(frame, y, groups))
    if set(groups[a]) & set(groups[b]):
        raise ValueError("Fuga en calibración de pesos")
    design = CatalogDesign()
    train, valid = design.fit_transform(frame.iloc[a], y[a], groups[a], frame.iloc[b], SEED + fold)
    hgb = _hgb(HGB_DEFAULT).fit(train["tree"], y[a])
    bag = _bag(BAG_DEFAULT, threads).fit(train["tree"], y[a])
    ridge = _ridge(RIDGE_DEFAULT).fit(train["ridge"], y[a])
    matrix = np.column_stack([clip(hgb.predict(valid["tree"])), clip(bag.predict(valid["tree"])),
                              clip(ridge.predict(valid["ridge"]))])
    return _optimize_weights(y[b], matrix), len(a), len(b)


def _summarize(folds: pd.DataFrame, oof: dict, y) -> pd.DataFrame:
    rows = []
    for model, subset in folds.groupby("model", sort=False):
        row = {"model": model, "cv_std": float(subset.mae.std(ddof=0)), "cv_oof_mae": metrics(y, oof[model])["mae"]}
        row.update({f"cv_{name}": float(subset[name].mean()) for name in metrics(y, oof[model])})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cv_mae", "model"], ignore_index=True)


def fit_models(data, root: Path, out: Path, model_dir: Path | None = None, n_threads: int = 4):
    """Entrena, selecciona y congela la decisión por CV antes de puntuar la prueba."""
    root, out = Path(root), Path(out)
    model_dir = Path(model_dir) if model_dir is not None else root / "models/eva3"
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    clean, train_index, test_index = data["clean"], data["train_idx"], data["test_idx"]
    frame = clean.iloc[train_index].reset_index(drop=True)
    y = frame.popularity.to_numpy(dtype=float)
    groups = np.asarray(data["groups"].iloc[train_index])
    test_groups = np.asarray(data["groups"].iloc[test_index])
    if set(groups) & set(test_groups):
        raise ValueError("Fuga entre entrenamiento y prueba")
    folds = list(GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(frame, y, groups))
    cv_split_rows = []
    trial_rows = []
    fold_rows = []
    fold_calibration = []
    oof = {}
    all_trial_predictions = {}
    all_tree_predictions = {}
    specs = [(MODEL_HGB, HGB_SEARCH), (MODEL_BAG, BAG_SEARCH), (MODEL_RIDGE, RIDGE_SEARCH)]
    def get_oof(name):
        return oof.setdefault(name, np.full(len(y), np.nan))
    def get_trial(name, n):
        return all_trial_predictions.setdefault((name, n), np.full(len(y), np.nan))
    def write_fold(name, i, a, b, prediction):
        get_oof(name)[b] = prediction
        fold_rows.append({"model": name, "fold": i, "train_rows": len(a), "val_rows": len(b),
                          **metrics(y[b], prediction)})
    start = time.perf_counter()
    with threadpool_limits(limits=n_threads):
        for i, (a, b) in enumerate(folds, start=1):
            if set(groups[a]) & set(groups[b]):
                raise ValueError("Fuga en validación cruzada")
            print(f"EP3 fold {i}/{FOLDS}: {len(a):,} ajuste, {len(b):,} validación", flush=True)
            x_a, x_b = frame.iloc[a], frame.iloc[b]
            cv_split_rows.append({"fold": i, "train_rows": len(a), "val_rows": len(b),
                                  "train_groups": len(set(groups[a])), "val_groups": len(set(groups[b])),
                                  "group_intersection": 0})
            dummy = DummyRegressor(strategy="median").fit(x_a[MODEL_FEATURES], y[a])
            write_fold(MODEL_BASE, i, a, b, clip(dummy.predict(x_b[MODEL_FEATURES])))
            ep2 = hgb_pipeline({"model__max_leaf_nodes": 31, "model__l2_regularization": 10.0})
            ep2.fit(x_a[MODEL_FEATURES], y[a])
            write_fold(MODEL_EP2, i, a, b, clip(ep2.predict(x_b[MODEL_FEATURES])))
            design = CatalogDesign()
            x_tr, x_va = design.fit_transform(x_a, y[a], groups[a], x_b, SEED + i)
            for family, candidates in specs:
                print(f"  {family}: {len(candidates)} configuraciones", flush=True)
                for trial, params in enumerate(candidates, start=1):
                    started = time.perf_counter()
                    if family == MODEL_HGB:
                        reg = _hgb(params).fit(x_tr["tree"], y[a])
                        prediction = clip(reg.predict(x_va["tree"]))
                    elif family == MODEL_BAG:
                        prediction, tree = _predict_tree_bag(x_tr["tree"], x_va["tree"], y[a], params, n_threads)
                        all_tree_predictions.setdefault(trial, np.full(len(y), np.nan))[b] = tree
                    else:
                        reg = _ridge(params).fit(x_tr["ridge"], y[a])
                        prediction = clip(reg.predict(x_va["ridge"]))
                    get_trial(family, trial)[b] = prediction
                    trial_rows.append({"model": family, "trial": trial, "fold": i,
                                       "params": json.dumps(params, sort_keys=True), "train_rows": len(a),
                                       "val_rows": len(b), "fit_predict_seconds": time.perf_counter() - started,
                                       **metrics(y[b], prediction)})
            fixed_predictions = np.column_stack([
                get_trial(MODEL_HGB, 1)[b], get_trial(MODEL_BAG, 1)[b], get_trial(MODEL_RIDGE, 2)[b]
            ])
            print("  calibrar pesos solo dentro de entrenamiento externo", flush=True)
            weights, n_fit, n_cal = _calibration_fold(x_a, y[a], groups[a], i, n_threads)
            fold_calibration.append({"fold": i, "weights": dict(zip(CALIBRATION_COMPONENTS, weights.tolist())),
                                     "fit_rows": n_fit, "calibration_rows": n_cal})
            write_fold(MODEL_BLEND_CAL, i, a, b, fixed_predictions @ weights)
        search = pd.DataFrame(trial_rows)
        search.to_csv(out / "search_trials.csv", index=False)
        cv_search = search.groupby(["model", "trial", "params"], sort=False).agg(
            cv_mae=("mae", "mean"), cv_std=("mae", lambda x: x.std(ddof=0)),
            cv_rmse=("rmse", "mean"), cv_r2=("r2", "mean"),
            cv_within_15=("within_15", "mean"), seconds=("fit_predict_seconds", "sum"),
        ).reset_index().sort_values(["model", "cv_mae", "trial"], ignore_index=True)
        cv_search["selected_in_family"] = ~cv_search.duplicated("model")
        cv_search.to_csv(out / "search_summary.csv", index=False)
        best = cv_search[cv_search.selected_in_family].set_index("model")
        best_trials = {family: int(best.loc[family, "trial"]) for family, _ in specs}
        best_params = {family: json.loads(best.loc[family, "params"]) for family, _ in specs}
        for i, (a, b) in enumerate(folds, start=1):
            for name, predictions in [
                (MODEL_HGB_BASE, get_trial(MODEL_HGB, 1)),
                (MODEL_HGB, get_trial(MODEL_HGB, best_trials[MODEL_HGB])),
                (MODEL_BAG, get_trial(MODEL_BAG, best_trials[MODEL_BAG])),
                (MODEL_TREE, all_tree_predictions[best_trials[MODEL_BAG]]),
                (MODEL_RIDGE, get_trial(MODEL_RIDGE, best_trials[MODEL_RIDGE])),
            ]:
                write_fold(name, i, a, b, predictions[b])
            write_fold(MODEL_BLEND, i, a, b, np.column_stack([get_oof(k)[b] for k in COMPONENTS]).mean(axis=1))
        assert all(np.isfinite(pred).all() for pred in oof.values()), "OOF incompleto"
        fold_table = pd.DataFrame(fold_rows)
        fold_table.to_csv(out / "cv_folds.csv", index=False)
        cv_summary = _summarize(fold_table, oof, y)
        cv_summary.to_csv(out / "cv_summary.csv", index=False)
        selected = str(cv_summary.iloc[0].model)
        # Para despliegue se usa el OOF de componentes de parámetros fijos del
        # entrenamiento completo. La métrica CV de esta mezcla es la calibrada
        # en holdback interno de cada fold, no la del mismo OOF usado para pesos.
        fixed_oof = np.column_stack([get_trial(MODEL_HGB, 1), get_trial(MODEL_BAG, 1),
                                     get_trial(MODEL_RIDGE, 2)])
        final_weights = _optimize_weights(y, fixed_oof)
        selection = {
            "selected_model": selected,
            "criterion": "menor MAE medio en cinco folds agrupados del entrenamiento; empate por nombre",
            "test_used_for_selection": False, "seed": SEED,
            "features": FEATURES_META if selected not in {MODEL_BASE, MODEL_EP2} else MODEL_FEATURES,
            "input_columns": FEATURES_META if selected not in {MODEL_BASE, MODEL_EP2} else MODEL_FEATURES,
            "historical_metadata": selected not in {MODEL_BASE, MODEL_EP2},
            "selection_cv_optimism": ("La búsqueda y la selección reutilizan los folds de entrenamiento; "
                                      "la cifra final del holdout es la comprobación independiente."),
            "best_params": best_params, "best_parameters": best_params, "best_trials": best_trials,
            "hgb_base_params": HGB_DEFAULT, "bag_fixed_params": BAG_DEFAULT, "ridge_fixed_params": RIDGE_DEFAULT,
            "cv_folds": cv_split_rows, "metadata_history_inner_folds": HISTORY_FOLDS,
            "history_keys_and_smoothing": HISTORY_KEYS, "tolerances": list(TOLERANCES),
            "blend_weights": {
                MODEL_BLEND: dict.fromkeys(COMPONENTS, 1 / 3),
                MODEL_BLEND_CAL: dict(zip(CALIBRATION_COMPONENTS, final_weights.tolist())),
            },
            "calibration_by_fold": fold_calibration,
            "tolerance_diagnostic": {"points": 15, "target_fraction": 0.80, "comparison": ">",
                                     "status": "meta de trabajo, no métrica confirmada por el docente"},
            "prediction_range": [0, 100],
        }
        save_json(out / "selection.json", selection)
        fold_assignment = np.full(len(frame), -1, dtype=int)
        for i, (_, b) in enumerate(folds, start=1):
            fold_assignment[b] = i
        pd.DataFrame({"track_id": frame.track_id, "audio_group": groups, "popularity": y,
                      "fold": fold_assignment, **oof}).to_csv(out / "oof_predictions.csv", index=False)

        # Holdout: no etiquetas, métricas ni predicciones antes de selection.json.
        print(f"EP3 decisión por CV congelada: {selected}. Ajustar en todo train y evaluar prueba.", flush=True)
        test = clean.iloc[test_index].reset_index(drop=True)
        y_test = test.popularity.to_numpy(dtype=float)
        design = CatalogDesign()
        x_tr, x_te = design.fit_transform(frame, y, groups, test, SEED)
        final_models = {
            MODEL_BASE: DummyRegressor(strategy="median").fit(frame[MODEL_FEATURES], y),
            MODEL_EP2: hgb_pipeline({"model__max_leaf_nodes": 31, "model__l2_regularization": 10.0}).fit(
                frame[MODEL_FEATURES], y),
            MODEL_HGB_BASE: _hgb(HGB_DEFAULT).fit(x_tr["tree"], y),
            MODEL_HGB: _hgb(best_params[MODEL_HGB]).fit(x_tr["tree"], y),
            MODEL_BAG: _bag(best_params[MODEL_BAG], n_threads).fit(x_tr["tree"], y),
            MODEL_RIDGE: _ridge(best_params[MODEL_RIDGE]).fit(x_tr["ridge"], y),
        }
        final_models[MODEL_TREE] = final_models[MODEL_BAG].estimators_[0]
        final_models["extra_trees_fijo"] = (final_models[MODEL_BAG] if best_params[MODEL_BAG] == BAG_DEFAULT
                                             else _bag(BAG_DEFAULT, n_threads).fit(x_tr["tree"], y))
        final_models["ridge_fijo"] = (final_models[MODEL_RIDGE] if best_params[MODEL_RIDGE] == RIDGE_DEFAULT
                                       else _ridge(RIDGE_DEFAULT).fit(x_tr["ridge"], y))
        catalog_views = {name: ("ridge" if name in {MODEL_RIDGE, "ridge_fijo"} else "tree")
                         for name in final_models if name not in {MODEL_BASE, MODEL_EP2}}
        pred = {name: clip(model.predict(test[MODEL_FEATURES] if name not in catalog_views
                                         else x_te[catalog_views[name]]))
                for name, model in final_models.items()}
        pred[MODEL_BLEND] = np.column_stack([pred[name] for name in COMPONENTS]).mean(axis=1)
        pred[MODEL_BLEND_CAL] = np.column_stack([pred[name] for name in CALIBRATION_COMPONENTS]) @ final_weights
        if selected in {MODEL_BASE, MODEL_EP2}:
            pipeline = final_models[selected]
        elif selected in {MODEL_BLEND, MODEL_BLEND_CAL}:
            components = COMPONENTS if selected == MODEL_BLEND else CALIBRATION_COMPONENTS
            weights = np.full(len(components), 1 / len(components)) if selected == MODEL_BLEND else final_weights
            pipeline = CatalogPredictor(design, [(catalog_views[n], final_models[n]) for n in components], weights)
        else:
            pipeline = CatalogPredictor(design, [(catalog_views[selected], final_models[selected])])
        columns = selection["features"]
        if not np.allclose(clip(pipeline.predict(test[columns])), pred[selected]):
            raise ValueError("Predicciones distintas en la interfaz del ganador")
        joblib.dump({"pipeline": pipeline, "metadata": selection, "dataset_sha256": data["audit"]["raw_sha256"],
                     "sklearn_version": version("scikit-learn")},
                    model_dir / "popularity_pipeline.joblib", compress=3)
    model_scores = pd.DataFrame([{"model": name, **metrics(y_test, pred[name])} for name in oof])
    results = cv_summary.merge(model_scores, on="model", validate="one_to_one").sort_values("cv_mae")
    results.to_csv(out / "supervised_metrics.csv", index=False)
    detail = test[["track_id", "track_genre", "explicit", "popularity"]].copy()
    train_keys, test_keys = catalog_keys(frame), catalog_keys(test)
    detail["audio_group"] = test_groups
    detail["prediction"] = pred[selected]
    detail["baseline_median"] = pred[MODEL_BASE]
    detail["baseline_ep2"] = pred[MODEL_EP2]
    detail["residual"] = y_test - pred[selected]
    detail["abs_error"] = np.abs(detail.residual)
    detail["artist_seen"] = test_keys.artist_primary.isin(train_keys.artist_primary)
    detail["album_seen"] = test_keys.artist_album.isin(train_keys.artist_album)
    detail["target_band"] = pd.cut(detail.popularity, [-1, 20, 40, 60, 80, 100],
                                    labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
    for model in (*oof, "extra_trees_fijo", "ridge_fijo"):
        detail[model] = pred[model]
    detail.to_csv(out / "test_predictions.csv", index=False)
    for attr in ("artist_seen", "album_seen", "track_genre", "explicit", "target_band"):
        slices = [{"value": str(key), "n": len(slice_), "eligible_n_ge_30": len(slice_) >= 30,
                   **metrics(slice_.popularity, slice_.prediction),
                   "baseline_ep2_mae": float(mean_absolute_error(slice_.popularity, slice_.baseline_ep2))}
                  for key, slice_ in detail.groupby(attr, observed=True, sort=False) if len(slice_) >= 2]
        pd.DataFrame(slices).to_csv(out / f"errors_by_{attr}.csv", index=False)
    metric = metrics(y_test, pred[selected])
    base, old = metrics(y_test, pred[MODEL_BASE]), metrics(y_test, pred[MODEL_EP2])
    evaluation = {
        "selected_model": selected, "metrics": metric, "baseline_median_metrics": base,
        "baseline_ep2_metrics": old, "test_rows": len(test), "train_rows": len(frame),
        "improvement_vs_median_pct": 100 * (1 - metric["mae"] / base["mae"]),
        "improvement_vs_ep2_pct": 100 * (1 - metric["mae"] / old["mae"]),
        "coverage_target_met": metric["within_15"] > 0.80,
        "coverage_target": selection["tolerance_diagnostic"],
        "residual_bias": float(detail.residual.mean()),
        "residual_p95_absolute": float(detail.abs_error.quantile(.95)),
        "elapsed_seconds": time.perf_counter() - start,
        "train_artists_seen_test_fraction": float(detail.artist_seen.mean()),
        "train_albums_seen_test_fraction": float(detail.album_seen.mean()),
    }
    save_json(out / "evaluation.json", evaluation)
    print(results[["model", "cv_mae", "mae", "rmse", "r2", "within_15", "within_20"]].to_string(index=False),
          flush=True)
    return {"selected": selected, "pipeline": pipeline, "selection": selection,
            "evaluation": evaluation, "results": results, "cv_summary": cv_summary}
