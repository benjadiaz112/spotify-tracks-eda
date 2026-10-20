"""Auditorías controladas de género y etiquetas, sin reajustar la selección."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .data import ROOT, SEED, save_json
from .supervised import clip_predictions, hgb_pipeline, metrics
from src.features.prepare_features import MODEL_FEATURES


def audit_genre(data: dict, root: Path = ROOT):
    out = root / "reports/eva2"
    clean, tr, te = data["clean"], data["train_idx"], data["test_idx"]
    train, test = clean.iloc[tr], clean.iloc[te]
    groups = data["groups"].iloc[tr].to_numpy()
    folds = list(GroupKFold(3, shuffle=True, random_state=SEED).split(train, groups=groups))
    config = json.loads((out / "selection.json").read_text(encoding="utf-8"))["best_parameters"]["HistGradientBoosting"]
    retained_features = [name for name in MODEL_FEATURES if name != "track_genre"]
    if set(MODEL_FEATURES) - set(retained_features) != {"track_genre"}:
        raise ValueError("La ablación debe retirar únicamente track_genre")
    shared_params = ["max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization", "early_stopping", "random_state"]
    full = hgb_pipeline(config)
    reduced = hgb_pipeline(config, include_genre=False)
    fixed = {p: full.named_steps["model"].get_params()[p] for p in shared_params}
    if fixed != {p: reduced.named_steps["model"].get_params()[p] for p in shared_params}:
        raise ValueError("Los hiperparámetros comunes de la ablación difieren")
    genre_cv, reduced_cv = [], []
    with threadpool_limits(limits=4):
        for a, b in folds:
            fit, val = train.iloc[a], train.iloc[b]
            mapping = fit.groupby("track_genre").popularity.mean()
            pred = val.track_genre.map(mapping).fillna(fit.popularity.mean()).to_numpy()
            genre_cv.append(metrics(val.popularity, pred)["mae"])
            model = hgb_pipeline(config, include_genre=False).fit(fit[retained_features], fit.popularity)
            reduced_cv.append(metrics(val.popularity, model.predict(val[retained_features]))["mae"])
        mapping = train.groupby("track_genre").popularity.mean()
        genre_pred = test.track_genre.map(mapping).fillna(train.popularity.mean()).to_numpy()
        reduced.fit(train[retained_features], train.popularity)
        reduced_pred = clip_predictions(reduced.predict(test[retained_features]))
    table = pd.DataFrame([
        {"model": "Media_por_genero_train", "cv_mae": np.mean(genre_cv), **metrics(test.popularity, genre_pred)},
        {"model": "HGB_sin_genero_14_variables", "cv_mae": np.mean(reduced_cv), **metrics(test.popularity, reduced_pred)},
    ])
    table.to_csv(out / "sensitivity_genre.csv", index=False)
    save_json(out / "sensitivity_protocol.json", {
        "used_to_change_selection": False, "only_feature_removed": "track_genre",
        "retained_features": retained_features, "same_grouped_folds": True,
        "fixed_shared_hyperparameters": fixed,
        "note": "Ablación corregida: retira solo género y conserva tonalidad, modo y compás. Reutiliza configuración, no optimiza ni cambia selección. Diagnóstico de asociación predictiva, no causalidad.",
        "unseen_genre_fallback": "media de entrenamiento",
    })
    print(table.to_string(index=False), flush=True)
    return table


def audit_label_policy(data: dict, root: Path = ROOT):
    """Cambiar SOLO y de ajuste y evaluar contra la misma referencia de CV.

    Se limita al train original, con parámetros congelados. Nunca cambia el
    objetivo de test ni vuelve a seleccionar el ganador por esta comparación.
    """
    out = root / "reports/eva2"
    train = data["clean"].iloc[data["train_idx"]]
    groups = data["groups"].iloc[data["train_idx"]].to_numpy()
    medians = data["raw"].groupby("track_id").popularity.median()
    alternative_y = train.track_id.map(medians)
    config = json.loads((out / "selection.json").read_text(encoding="utf-8"))["best_parameters"]["HistGradientBoosting"]
    rows = []
    with threadpool_limits(limits=4):
        for fold, (a, b) in enumerate(GroupKFold(3, shuffle=True, random_state=SEED).split(train, groups=groups), 1):
            X_fit, X_val = train.iloc[a][MODEL_FEATURES], train.iloc[b][MODEL_FEATURES]
            original = hgb_pipeline(config).fit(X_fit, train.iloc[a].popularity)
            alternative = hgb_pipeline(config).fit(X_fit, alternative_y.iloc[a])
            y_reference = train.iloc[b].popularity
            first_mae = metrics(y_reference, original.predict(X_val))["mae"]
            median_mae = metrics(y_reference, alternative.predict(X_val))["mae"]
            rows.append({"fold": fold, "validation_n": len(b),
                         "changed_fit_labels": int(train.iloc[a].popularity.ne(alternative_y.iloc[a]).sum()),
                         "mae_first_training": first_mae, "mae_median_training": median_mae,
                         "delta_median_minus_first": median_mae - first_mae})
    table = pd.DataFrame(rows)
    table.to_csv(out / "label_sensitivity_folds.csv", index=False)
    save_json(out / "label_sensitivity.json", {
        "scope": "training_only_grouped_cv", "test_used": False,
        "selection_changed": False, "primary_target_changed": False,
        "validation_reference": "first_observed_row_by_track_id",
        "cv_mae_first_training": float(table.mae_first_training.mean()),
        "cv_mae_median_training": float(table.mae_median_training.mean()),
        "delta_median_minus_first": float(table.delta_median_minus_first.mean()),
        "training_labels_changed": int(train.popularity.ne(alternative_y).sum()),
        "limitations": "La referencia first no es verdad temporal certificada. Este contraste mide sensibilidad a las etiquetas usadas para entrenar, sin demostrar superioridad de la mediana ni rendimiento futuro.",
    })
    print("Sensibilidad de etiquetas en train:", table.to_string(index=False), flush=True)
    return table
