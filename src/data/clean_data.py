"""Limpieza reproducible del dataset Spotify Tracks."""

from pathlib import Path

import pandas as pd


REQUIRED_TEXT_COLUMNS = ["artists", "album_name", "track_name"]
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]


def clean_spotify_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia el dataset y devuelve los datos junto con un resumen del proceso."""
    cleaned = df.copy()
    steps: list[dict[str, int | str]] = []

    def register(step: str, before: int) -> None:
        steps.append(
            {
                "paso": step,
                "filas_antes": before,
                "filas_despues": len(cleaned),
                "filas_eliminadas": before - len(cleaned),
            }
        )

    before = len(cleaned)
    cleaned = cleaned.drop(columns=["Unnamed: 0"], errors="ignore")
    register("Eliminar columna de indice exportado", before)

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=REQUIRED_TEXT_COLUMNS)
    register("Eliminar registros sin artista, album o cancion", before)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset="track_id", keep="first")
    register("Conservar un registro por track_id", before)

    before = len(cleaned)
    cleaned = cleaned[cleaned["duration_ms"].between(30_000, 1_200_000)]
    register("Conservar duraciones entre 30 segundos y 20 minutos", before)

    before = len(cleaned)
    cleaned = cleaned[(cleaned["tempo"] > 0) & cleaned["time_signature"].between(1, 5)]
    register("Eliminar tempo o compas no validos", before)

    valid_audio = cleaned[AUDIO_FEATURES].apply(lambda column: column.between(0, 1)).all(axis=1)
    before = len(cleaned)
    cleaned = cleaned[valid_audio]
    register("Validar caracteristicas de audio entre 0 y 1", before)

    cleaned = cleaned.reset_index(drop=True)
    return cleaned, pd.DataFrame(steps)


def validate_clean_data(df: pd.DataFrame) -> None:
    """Detiene el proceso si los datos limpios no cumplen las reglas acordadas."""
    assert "Unnamed: 0" not in df.columns
    assert not df[REQUIRED_TEXT_COLUMNS].isna().any().any()
    assert not df["track_id"].duplicated().any()
    assert df["duration_ms"].between(30_000, 1_200_000).all()
    assert (df["tempo"] > 0).all()
    assert df["time_signature"].between(1, 5).all()
    assert df[AUDIO_FEATURES].apply(lambda column: column.between(0, 1)).all().all()
    assert df["popularity"].between(0, 100).all()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "Spotify_Tracks_Dataset.csv"
    output_path = project_root / "data" / "processed" / "spotify_tracks_clean.csv"
    report_path = project_root / "reports" / "data_quality" / "cleaning_summary.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    clean, summary = clean_spotify_data(raw)
    validate_clean_data(clean)

    clean.to_csv(output_path, index=False)
    summary.to_csv(report_path, index=False)

    print(f"Filas originales: {len(raw):,}")
    print(f"Filas limpias: {len(clean):,}")
    print(f"Archivo generado: {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
