"""Reproducir EP3 sin modificar las evidencias históricas de EP2."""
import argparse
import hashlib
import time
from pathlib import Path

from src.eva2.data import ROOT, save_json


def train_supervised(data: dict, root: Path, *, reuse: bool, n_threads: int = 4) -> dict:
    from .modeling import fit_models

    import json
    from threadpoolctl import threadpool_limits

    out = root / "reports/eva3"
    if not reuse:
        with threadpool_limits(limits=n_threads):
            fit_models(data, root, out, model_dir=root / "models/eva3", n_threads=n_threads)
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Selección incompatible con el protocolo.")
    return selection


def finish_run(data: dict, root: Path, selection: dict, *, refitted: bool, start: float) -> dict:
    from .business import audit_business
    from .contrasts import audit_contrasts
    from .plots import make_plots
    from .predict import predict_frame
    from .segments import fit_segments

    import numpy as np
    import pandas as pd

    out, models, images = (root / "reports/eva3", root / "models/eva3", root / "images/eva3")
    results = pd.read_csv(out / "supervised_metrics.csv")
    audit_contrasts(out)
    decision = audit_business(data, out)
    segments = fit_segments(data, root, out, models)
    make_plots(data, out, images)
    model_path = models / "popularity_pipeline.joblib"
    example = data["clean"].iloc[data["test_idx"][:5]].drop(columns=["popularity"])
    example = example.drop(columns=["Unnamed: 0"], errors="ignore")
    example.to_csv(out / "prediction_input_example.csv", index=False)
    predicted = predict_frame(example, model_path)
    expected = pd.read_csv(out / "test_predictions.csv", usecols=["track_id", "prediction"]).set_index("track_id")
    if not np.allclose(predicted.estimated_popularity, expected.loc[predicted.track_id, "prediction"],
                       atol=1e-8, rtol=0):
        raise ValueError("El modelo serializado no reproduce las predicciones guardadas.")
    predicted.to_csv(out / "prediction_example.csv", index=False)
    meta = {
        "completed": True, "supervised_refit": refitted,
        "selected_model": selection["selected_model"], "selected_k": segments["k"],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    save_json(out / "run.json", meta)
    return {"data": data["audit"], "results": results, "business": decision, **meta}


def run(root: Path = ROOT, *, reuse_supervised: bool = False, n_threads: int = 4) -> dict:
    from .data import prepare
    from .plots import make_audio_eda

    start = time.perf_counter()
    out = root / "reports/eva3"
    data = prepare(root, out)
    make_audio_eda(data["clean"].iloc[data["train_idx"]], out, root / "images/eva3")
    selection = train_supervised(data, root, reuse=reuse_supervised, n_threads=n_threads)
    return finish_run(data, root, selection, refitted=not reuse_supervised, start=start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-supervised", action="store_true",
                        help="Reutilizar el ajuste registrado; recalcular auditoría, figuras y segmentación.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads debe ser positivo")
    result = run(reuse_supervised=args.reuse_supervised, n_threads=args.threads)
    print(f"EP3 completada: {result['selected_model']}; K={result['selected_k']}")


if __name__ == "__main__":
    main()
