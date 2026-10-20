"""Contrastes diagnósticos posteriores, sin cambiar el modelo seleccionado."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from .data import ROOT, SEED, save_json
from .supervised import clip_predictions, metrics
from src.features.prepare_features import CONTINUOUS_FEATURES as NUM


def audit_genre(data: dict, root: Path = ROOT):
    """Medir si el valor proviene del género o de audio, sin buscar más parámetros."""
    out = root / "reports/eva2"
    clean, tr, te = data["clean"], data["train_idx"], data["test_idx"]
    train, test = clean.iloc[tr], clean.iloc[te]
    groups = data["groups"].iloc[tr].to_numpy()
    folds = list(GroupKFold(3, shuffle=True, random_state=SEED).split(train, groups=groups))
    genre_cv, audio_cv = [], []
    config = json.loads((out / "selection.json").read_text(encoding="utf-8"))["best_parameters"]["HistGradientBoosting"]
    def build():
        return Pipeline([("prepare", ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), NUM + ["explicit"])])),
            ("model", HistGradientBoostingRegressor(
                learning_rate=.08, max_iter=220, early_stopping=False, random_state=SEED,
                **{key.removeprefix("model__"): value for key, value in config.items()}))])
    with threadpool_limits(limits=4):
        for a, b in folds:
            fit, val = train.iloc[a], train.iloc[b]
            mapping = fit.groupby("track_genre").popularity.mean()
            pred = val.track_genre.map(mapping).fillna(fit.popularity.mean()).to_numpy()
            genre_cv.append(metrics(val.popularity, pred)["mae"])
            model = build().fit(fit, fit.popularity)
            audio_cv.append(metrics(val.popularity, model.predict(val))["mae"])
        mapping = train.groupby("track_genre").popularity.mean()
        genre_pred = test.track_genre.map(mapping).fillna(train.popularity.mean()).to_numpy()
        model = build().fit(train, train.popularity)
        audio_pred = clip_predictions(model.predict(test))
    table = pd.DataFrame([
        {"model": "Media_por_genero_train", "cv_mae": np.mean(genre_cv), **metrics(test.popularity, genre_pred)},
        {"model": "HGB_solo_audio_sin_reajuste", "cv_mae": np.mean(audio_cv), **metrics(test.popularity, audio_pred)},
    ])
    table.to_csv(out / "sensitivity_genre.csv", index=False)
    save_json(out / "sensitivity_protocol.json", {
        "used_to_change_selection": False,
        "note": "Auditoría de dependencia de género posterior a la selección. Audio reutiliza parámetros HGB, no se optimiza. No prueba causalidad.",
        "same_grouped_folds": True, "unseen_genre_fallback": "media de entrenamiento",
    })
    print(table.to_string(index=False), flush=True)
    return table
