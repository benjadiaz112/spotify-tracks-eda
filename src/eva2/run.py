"""Ejecutar la evaluación 2 completa desde el CSV recibido."""
import time

from .data import ROOT, prepare, save_json
from .supervised import fit_supervised
from .clustering import fit_clustering
from .sensitivity import audit_genre
from .plots import make_plots
from .reporting import build_reports
from .predict import predict
from src.features.prepare_features import MODEL_FEATURES


def main():
    start = time.perf_counter()
    data = prepare()
    fit_supervised(data)
    fit_clustering(data)
    audit_genre(data)
    make_plots(data)
    build_reports()
    dest = ROOT / "data/examples"
    dest.mkdir(exist_ok=True)
    example = data["clean"].iloc[data["test_idx"][:3]][["track_id"] + MODEL_FEATURES]
    example.to_csv(dest / "tracks_example.csv", index=False)
    predict(example).to_csv(ROOT / "reports/eva2/example_predictions.csv", index=False)
    save_json(ROOT / "reports/eva2/run.json", {
        "completed": True, "elapsed_seconds": time.perf_counter() - start,
        "source": "CSV raw validado", "entrypoint": "python -m src.eva2.run",
    })
    print("Evaluación completa. Consulta docs/informe_tecnico_eva2.md")


if __name__ == "__main__":
    main()
