"""Estimar popularidad observada a partir de un CSV de canciones candidatas."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.eva2.data import ROOT, EXPECTED_SHA256
from .data import ARCHIVE_MEMBER_SHA256
from src.features.prepare_features import CONTINUOUS_FEATURES


def predict_frame(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    if artifact["dataset_sha256"] not in {EXPECTED_SHA256, ARCHIVE_MEMBER_SHA256} or artifact["sklearn_version"] != sklearn.__version__:
        raise ValueError("El modelo no corresponde al dataset o a la versión de scikit-learn validada.")
    required = artifact["metadata"]["input_columns"]
    if missing := sorted(set(required).difference(frame.columns)):
        raise ValueError("Columnas requeridas ausentes: " + ", ".join(missing))
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Se requiere al menos una fila y las variables de entrada deben estar completas.")
    if not np.isfinite(frame[CONTINUOUS_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Atributos de audio no finitos.")
    features = frame[required].copy()
    values = np.clip(artifact["pipeline"].predict(features), 0, 100)
    if not np.isfinite(values).all():
        raise ValueError("El modelo produjo una predicción no finita.")
    result = pd.DataFrame({"estimated_popularity": values})
    if "track_id" in frame:
        result.insert(0, "track_id", frame.track_id.to_numpy())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output/predictions_ep3.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/eva3/popularity_pipeline.joblib")
    args = parser.parse_args()
    predictions = predict_frame(pd.read_csv(args.input), args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} predicciones: {args.output}")


if __name__ == "__main__":
    main()
