"""Inferencia local con validación del contrato de entrada."""
import argparse
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import ROOT
from .supervised import clip_predictions
from src.features.prepare_features import CONTINUOUS_FEATURES as NUM, MODEL_FEATURES


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(MODEL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan variables: {sorted(missing)}")
    X = df[MODEL_FEATURES].copy()
    if X.empty or X.isna().any().any():
        raise ValueError("La entrada está vacía o tiene valores faltantes")
    numeric = NUM + ["key", "mode", "time_signature"]
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="raise")
    if not np.isfinite(X[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Variables numéricas no finitas")
    # Admitir explícitamente bool, 0/1 y cadenas habituales del CSV.
    binary = X.explicit.astype(str).str.lower().map({"true": 1, "false": 0, "0": 0, "1": 1})
    if binary.isna().any():
        raise ValueError("explicit debe ser True/False o 0/1")
    X["explicit"] = binary.astype("int8")
    if not X.duration_ms.between(30_000, 1_200_000).all() or not (X.tempo > 0).all():
        raise ValueError("Duración o tempo fuera del dominio de entrenamiento")
    bounded = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
    if not X[bounded].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("Las características normalizadas deben estar entre 0 y 1")
    for col, domain in {"key": range(12), "mode": [0, 1], "time_signature": range(1, 6)}.items():
        if not X[col].isin(domain).all():
            raise ValueError(f"{col} fuera del dominio de entrenamiento")
    if X.track_genre.astype(str).str.strip().eq("").any():
        raise ValueError("Género vacío")
    return X


def predict(df: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    X = validate_input(df)
    source = root / "models/eva2/popularity_pipeline.joblib"
    if not source.is_file():
        raise FileNotFoundError("Primero ejecuta python -m src.eva2.run")
    # Solo se carga el modelo propio generado por el proyecto.
    bundle = joblib.load(source)
    if bundle["sklearn_version"] != version("scikit-learn"):
        raise RuntimeError("Versión sklearn distinta de entrenamiento. Reentrena con el entorno actual.")
    result = pd.DataFrame(index=df.index)
    if "track_id" in df:
        result["track_id"] = df.track_id
    result["predicted_popularity"] = clip_predictions(bundle["pipeline"].predict(X))
    encoded = bundle["pipeline"].named_steps["prepare"].named_transformers_["category"].named_steps["encode"]
    result["unknown_genre"] = ~X.track_genre.isin(encoded.categories_[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV con las 15 variables predictoras")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/eva2/example_predictions.csv")
    args = parser.parse_args()
    result = predict(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
