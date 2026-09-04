"""Prepara los datos de Spotify para un modelo de regresión."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "popularity"
RANDOM_STATE = 42
TEST_SIZE = 0.20

IDENTIFIER_COLUMNS = ["track_id", "artists", "album_name", "track_name"]
CONTINUOUS_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
CATEGORICAL_FEATURES = ["track_genre", "key", "mode", "time_signature"]
BINARY_FEATURES = ["explicit"]
MODEL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Crea un preprocesador reproducible sin utilizar la variable objetivo."""
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numericas", numeric_pipeline, CONTINUOUS_FEATURES),
            ("categoricas", categorical_pipeline, CATEGORICAL_FEATURES),
            ("binarias", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa predictores y objetivo manteniendo una distribución similar del objetivo."""
    missing = set(MODEL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    features = df[MODEL_FEATURES].copy()
    features["explicit"] = features["explicit"].astype("int8")
    target = df[TARGET].copy()

    stratification_bins = pd.qcut(
        target.rank(method="first"), q=10, labels=False
    )

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratification_bins,
    )


def validate_preparation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> None:
    """Comprueba las condiciones necesarias para modelar sin fuga directa."""
    expected_columns = set(MODEL_FEATURES + [TARGET])
    assert set(train_df.columns) == expected_columns
    assert set(test_df.columns) == expected_columns
    assert not set(IDENTIFIER_COLUMNS).intersection(train_df.columns)
    assert not train_df.isna().any().any()
    assert not test_df.isna().any().any()
    assert set(train_df["explicit"].unique()).issubset({0, 1})
    assert set(test_df["explicit"].unique()).issubset({0, 1})
    assert len(train_df) + len(test_df) == 89_495

    transformed_train = preprocessor.transform(train_df[MODEL_FEATURES])
    transformed_test = preprocessor.transform(test_df[MODEL_FEATURES])
    train_values = transformed_train.data if hasattr(transformed_train, "data") else transformed_train
    test_values = transformed_test.data if hasattr(transformed_test, "data") else transformed_test
    assert np.isfinite(train_values).all()
    assert np.isfinite(test_values).all()
    assert transformed_train.shape[1] == transformed_test.shape[1]


def prepare_model_data(project_root: Path) -> dict[str, object]:
    """Divide los datos, ajusta el pipeline solo con train y guarda resultados."""
    input_path = project_root / "data" / "processed" / "spotify_tracks_clean.csv"
    train_path = project_root / "data" / "processed" / "train.csv"
    test_path = project_root / "data" / "processed" / "test.csv"
    pipeline_path = project_root / "models" / "preprocessing_pipeline.joblib"
    report_dir = project_root / "reports" / "preparation"

    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    x_train, x_test, y_train, y_test = split_dataset(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(x_train)

    train_df = x_train.copy()
    train_df[TARGET] = y_train
    test_df = x_test.copy()
    test_df[TARGET] = y_test
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    validate_preparation(train_df, test_df, preprocessor)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(preprocessor, pipeline_path)

    feature_names = pd.DataFrame(
        {"variable_transformada": preprocessor.get_feature_names_out()}
    )
    feature_names.to_csv(report_dir / "feature_names.csv", index=False)

    summary = pd.DataFrame(
        [
            {"conjunto": "train", "filas": len(train_df), "porcentaje": 80.0},
            {"conjunto": "test", "filas": len(test_df), "porcentaje": 20.0},
        ]
    )
    summary.to_csv(report_dir / "split_summary.csv", index=False)

    return {
        "train": train_df,
        "test": test_df,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "summary": summary,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    results = prepare_model_data(project_root)
    print(results["summary"].to_string(index=False))
    print(f"Variables originales para modelar: {len(MODEL_FEATURES)}")
    print(f"Variables después de codificar: {len(results['feature_names'])}")
    print("Pipeline ajustado únicamente con el conjunto de entrenamiento.")


if __name__ == "__main__":
    main()
