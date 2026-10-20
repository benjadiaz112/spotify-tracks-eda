"""Selección en CV agrupada y evaluación final independiente del ranking CV."""
from __future__ import annotations

import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .data import ROOT, SEED, save_json
from src.features.prepare_features import CONTINUOUS_FEATURES as NUM, CATEGORICAL_FEATURES as CAT, MODEL_FEATURES


def clip_predictions(prediction):
    """Regla fijada antes de CV: popularidad válida entre 0 y 100."""
    return np.clip(np.asarray(prediction, dtype=float), 0, 100)


def clipped_mae(y, prediction):
    return mean_absolute_error(y, clip_predictions(prediction))


def preprocessor(native_categories=False):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    encode = (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
              if native_categories else OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", encode)])
    return ColumnTransformer([
        ("numeric", numeric, NUM), ("category", categorical, CAT),
        ("binary", "passthrough", ["explicit"]),
    ], sparse_threshold=0.0 if native_categories else 1.0)


def candidates():
    """Grillas pequeñas predefinidas, iguales particiones para todos los candidatos."""
    def pipe(model, native=False):
        return Pipeline([("prepare", preprocessor(native)), ("model", model)])
    return {
        "Dummy_media": (pipe(DummyRegressor(strategy="mean")), {}),
        "Dummy_mediana": (pipe(DummyRegressor(strategy="median")), {}),
        "Regresion_lineal": (pipe(LinearRegression(tol=1e-6)), {}),
        "Arbol_decision": (pipe(DecisionTreeRegressor(random_state=SEED)), {
            "model__max_depth": [8, 16, 24], "model__min_samples_leaf": [10, 30],
        }),
        "HistGradientBoosting": (pipe(HistGradientBoostingRegressor(
            categorical_features=list(range(len(NUM), len(NUM) + len(CAT))),
            max_iter=220, learning_rate=0.08, early_stopping=False, random_state=SEED,
        ), native=True), {"model__max_leaf_nodes": [15, 31], "model__l2_regularization": [1.0, 10.0]}),
    }


def metrics(y, pred):
    pred = clip_predictions(pred)
    return {"mae": float(mean_absolute_error(y, pred)),
            "mse": float(mean_squared_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}


def bootstrap_groups(y, pred, baseline, groups, n_boot=600):
    """IC percentil por grupos de audio: respeta dependencia exacta intra-grupo.

    Condicional al modelo y al holdout observado. No mide variabilidad debida
    a otro entrenamiento ni un intervalo de predicción para una canción.
    """
    errors = pd.DataFrame({"group": np.asarray(groups), "err": np.abs(y - pred),
                           "base": np.abs(y - baseline), "n": 1})
    table = errors.groupby("group").sum().to_numpy()
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_boot):
        sample = table[rng.integers(0, len(table), len(table))].sum(axis=0)
        values.append([sample[0] / sample[2], 100 * (1 - sample[0] / sample[1])])
    bounds = np.quantile(values, [0.025, 0.975], axis=0)
    return {"resamples": n_boot, "unit": "audio_group", "mae_95_ci": bounds[:, 0].tolist(),
            "improvement_pct_95_ci": bounds[:, 1].tolist()}


def fit_supervised(data: dict, root: Path = ROOT) -> dict:
    out = root / "reports/eva2"
    model_dir = root / "models/eva2"
    model_dir.mkdir(parents=True, exist_ok=True)
    clean, tr, te = data["clean"], data["train_idx"], data["test_idx"]
    X_train, y_train = clean.iloc[tr][MODEL_FEATURES], clean.iloc[tr].popularity
    X_test, y_test = clean.iloc[te][MODEL_FEATURES], clean.iloc[te].popularity
    groups = data["groups"].iloc[tr].to_numpy()
    cv = GroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train, y_train, groups))
    for a, b in folds:
        if set(groups[a]) & set(groups[b]):
            raise ValueError("Grupos compartidos dentro de CV")
    scorer = make_scorer(clipped_mae, greater_is_better=False)
    fitted, rows, cv_tables, parameters = {}, [], [], {}
    # Limitar hilos mejora portabilidad y evita sobresuscripción en notebooks.
    with threadpool_limits(limits=4):
        for name, (estimator, grid) in candidates().items():
            print(f"CV: {name}", flush=True)
            start = time.perf_counter()
            search = GridSearchCV(estimator, grid, cv=folds, scoring=scorer,
                                  n_jobs=1, return_train_score=True, error_score="raise")
            search.fit(X_train, y_train)
            fitted[name] = search.best_estimator_
            parameters[name] = search.best_params_
            best = search.best_index_
            rows.append({"model": name, "cv_mae": -float(search.best_score_),
                         "cv_std": float(search.cv_results_["std_test_score"][best]),
                         "cv_train_mae": -float(search.cv_results_["mean_train_score"][best]),
                         "fit_seconds": time.perf_counter() - start})
            table = pd.DataFrame(search.cv_results_)
            table.insert(0, "model", name)
            cv_tables.append(table)
        cv_summary = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
        # Congelar selección ANTES de obtener cualquier predicción de prueba.
        # También se permite seleccionar un Dummy si ninguno aprende señal útil.
        selected = str(cv_summary.iloc[0]["model"])
        decision = {"selected_model": selected, "criterion": "menor MAE medio de CV agrupada de 3 folds",
                    "test_used_for_selection": False, "best_parameters": parameters,
                    "seed": SEED, "features": MODEL_FEATURES, "prediction_range": [0, 100],
                    "cv_folds": [{"train": len(a), "validation": len(b)} for a, b in folds]}
        save_json(out / "selection.json", decision)
        cv_summary.to_csv(out / "cv_summary.csv", index=False)
        pd.concat(cv_tables, ignore_index=True).to_csv(out / "cv_all_candidates.csv", index=False)
        joblib.dump({"pipeline": fitted[selected], "metadata": decision,
                     "dataset_sha256": data["audit"]["sha256"], "sklearn_version": version("scikit-learn")},
                    model_dir / "popularity_pipeline.joblib", compress=3)
        # A partir de aquí: auditoría de prueba. No se reajusta la elección.
        test_rows, all_predictions = [], {}
        for name, estimator in fitted.items():
            raw_pred = estimator.predict(X_test)
            pred = clip_predictions(raw_pred)
            all_predictions[name] = pred
            test_rows.append({"model": name, **metrics(y_test, pred),
                              "train_mae": metrics(y_train, estimator.predict(X_train))["mae"],
                              "clipped_count": int(np.sum(raw_pred != pred))})
        results = pd.DataFrame(test_rows).merge(cv_summary, on="model").sort_values("cv_mae")
        results.to_csv(out / "supervised_metrics.csv", index=False)
        pred = all_predictions[selected]
        detail = clean.iloc[te][["track_id", "track_genre", "explicit", "artists", "popularity"]].copy()
        detail["prediction"] = pred
        detail["baseline_mean"] = all_predictions["Dummy_media"]
        detail["residual"] = y_test.to_numpy() - pred
        detail["abs_error"] = np.abs(detail.residual)
        detail["artist_seen"] = detail.artists.isin(clean.iloc[tr].artists)
        detail["audio_group"] = data["groups"].iloc[te].to_numpy()
        detail["target_band"] = pd.cut(detail.popularity, [-1, 20, 40, 60, 80, 100],
                                        labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
        # Evitar exponer nombres en el reporte de errores, se conserva solo flag.
        detail.drop(columns="artists").to_csv(out / "test_predictions.csv", index=False)
        for column in ["track_genre", "explicit", "artist_seen", "target_band"]:
            table = detail.groupby(column, observed=True).agg(
                n=("abs_error", "size"), mae=("abs_error", "mean"),
                bias=("residual", "mean"), actual_mean=("popularity", "mean"),
                predicted_mean=("prediction", "mean"),
            )
            table["baseline_mae"] = detail.groupby(column, observed=True).apply(
                lambda d: np.mean(np.abs(d.popularity - d.baseline_mean)), include_groups=False)
            table["improvement_pct"] = 100 * (1 - table.mae / table.baseline_mae)
            table["eligible_n_ge_30"] = table.n >= 30
            table.to_csv(out / f"errors_by_{column}.csv")
        # Importancia de permutación solo en holdout, sin entrenar ni seleccionar variables.
        sample = X_test.sample(n=min(2500, len(X_test)), random_state=SEED)
        importance = permutation_importance(fitted[selected], sample, y_test.loc[sample.index],
                                           scoring=scorer, n_repeats=4, random_state=SEED, n_jobs=1)
        pd.DataFrame({"feature": MODEL_FEATURES, "mae_increase": importance.importances_mean,
                      "std": importance.importances_std}).sort_values("mae_increase", ascending=False).to_csv(
                          out / "permutation_importance.csv", index=False)
    confidence = bootstrap_groups(y_test.to_numpy(), pred, all_predictions["Dummy_media"], detail.audio_group)
    best = results.set_index("model").loc[selected]
    base = results.set_index("model").loc["Dummy_media"]
    improvement = 100 * (1 - best.mae / base.mae)
    evaluation = {"selected_model": selected, "metrics": metrics(y_test, pred),
                  "improvement_vs_mean_pct": float(improvement),
                  "mae_target_15_met": bool(best.mae <= 15), "improvement_target_10pct_met": bool(improvement >= 10),
                  "confidence": confidence, "test_rows": len(te),
                  "residual_p95_absolute": float(np.quantile(np.abs(y_test.to_numpy() - pred), 0.95)),
                  "residual_bias": float(np.mean(y_test.to_numpy() - pred))}
    save_json(out / "evaluation.json", evaluation)
    save_json(out / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "matplotlib", "joblib", "nbformat", "nbclient", "threadpoolctl"]}})
    print(results[["model", "cv_mae", "mae", "rmse", "r2"]].to_string(index=False), flush=True)
    return {"selected": selected, "results": results, "evaluation": evaluation, "pipeline": fitted[selected]}
