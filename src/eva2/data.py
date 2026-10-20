"""Auditoría y partición por grupos de audio, sin ajustar transformaciones."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.clean_data import clean_spotify_data, validate_clean_data
from src.features.prepare_features import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, MODEL_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
EXPECTED_SHA256 = "335e5ee0a07a3f30b20ca9860c87562c98082a032e0ecd90e8ca0a86d965fe58"
AUDIO = CONTINUOUS_FEATURES + ["key", "mode", "explicit", "time_signature"]


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def audio_groups(df: pd.DataFrame) -> pd.Series:
    """Huella de valores de audio EXACTOS, independiente de ID, target y género.

    No detecta remasterizaciones ni versiones casi iguales. La huella no es
    una identidad musical certificada, solo una regla conservadora de partición.
    """
    return pd.util.hash_pandas_object(df[AUDIO], index=False).astype(str).rename("audio_group")


def grouped_split(clean: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    groups = audio_groups(clean)
    # El target se usa solo para estratificar la partición, nunca como predictor.
    group_target = clean.groupby(groups, sort=True)["popularity"].median()
    bins = pd.qcut(group_target.rank(method="first"), q=10, labels=False)
    train_groups, test_groups = train_test_split(
        group_target.index.to_numpy(), test_size=0.2, random_state=SEED,
        stratify=bins.to_numpy(),
    )
    train_idx = np.flatnonzero(groups.isin(train_groups))
    test_idx = np.flatnonzero(groups.isin(test_groups))
    if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
        raise ValueError("Fuga: grupos de audio compartidos")
    if len(train_idx) + len(test_idx) != len(clean):
        raise ValueError("Partición incompleta")
    return train_idx, test_idx, groups


def prepare(root: Path = ROOT) -> dict:
    source = root / "data/raw/Spotify_Tracks_Dataset.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"CSV distinto del recibido: SHA256={digest}. Revisar procedencia antes de entrenar.")
    raw = pd.read_csv(source)
    clean, steps = clean_spotify_data(raw)
    validate_clean_data(clean)
    if not np.isfinite(clean[CONTINUOUS_FEATURES + ["popularity"]]).all().all():
        raise ValueError("Datos numéricos no finitos")
    if clean[MODEL_FEATURES].isna().any().any():
        raise ValueError("Valores faltantes inesperados")
    if not clean["key"].between(0, 11).all() or not clean["mode"].isin([0, 1]).all():
        raise ValueError("Tonalidad o modo fuera de dominio")
    clean["explicit"] = clean["explicit"].astype("int8")
    tr, te, groups = grouped_split(clean)
    out = root / "reports/eva2"
    out.mkdir(parents=True, exist_ok=True)
    steps.to_csv(out / "cleaning.csv", index=False)
    pd.DataFrame({"column": raw.columns, "dtype": raw.dtypes.astype(str).values,
                  "missing": raw.isna().sum().values,
                  "unique": raw.nunique().values}).to_csv(out / "schema.csv", index=False)
    clean[CONTINUOUS_FEATURES + ["popularity"]].describe().to_csv(out / "descriptive.csv")
    clean[CONTINUOUS_FEATURES + ["popularity"]].corr().to_csv(out / "correlations.csv")
    clean.groupby("track_genre")["popularity"].agg(["count", "mean", "median", "std"]).to_csv(out / "genres.csv")
    clean.groupby("explicit")["popularity"].agg(["count", "mean", "median"]).to_csv(out / "explicit.csv")
    manifest = clean[["track_id", "popularity"]].copy()
    manifest["audio_group"] = groups
    manifest["split"] = np.where(np.isin(np.arange(len(clean)), tr), "train", "test")
    manifest.to_csv(out / "split_manifest.csv", index=False)
    qa = {
        "sha256": digest, "raw_rows": len(raw), "raw_columns": len(raw.columns),
        "raw_missing_cells": int(raw.isna().sum().sum()),
        "raw_unique_tracks": int(raw.track_id.nunique()), "clean_rows": len(clean),
        "raw_retention_pct": 100 * len(clean) / len(raw),
        "unique_retention_pct": 100 * len(clean) / raw.track_id.nunique(),
        "clean_missing_cells": int(clean.isna().sum().sum()),
        "clean_genres": int(clean.track_genre.nunique()),
        "raw_genres": int(raw.track_genre.nunique()),
        "audio_groups": int(groups.nunique()),
        "repeated_audio_rows": int(groups.duplicated().sum()),
        "audio_groups_multiple_targets": int(clean.groupby(groups).popularity.nunique().gt(1).sum()),
        "train_rows": len(tr), "test_rows": len(te), "test_pct": 100 * len(te) / len(clean),
        "shared_tracks": len(set(clean.iloc[tr].track_id) & set(clean.iloc[te].track_id)),
        "shared_audio_groups": len(set(groups.iloc[tr]) & set(groups.iloc[te])),
        "test_artists_seen_pct": 100 * clean.iloc[te].artists.isin(clean.iloc[tr].artists).mean(),
        "popularity_mean": float(clean.popularity.mean()),
        "popularity_median": float(clean.popularity.median()),
        "seed": SEED,
    }
    save_json(out / "data_audit.json", qa)
    return {"raw": raw, "clean": clean, "train_idx": tr, "test_idx": te, "groups": groups, "audit": qa}


if __name__ == "__main__":
    result = prepare()
    print(json.dumps(result["audit"], indent=2))
