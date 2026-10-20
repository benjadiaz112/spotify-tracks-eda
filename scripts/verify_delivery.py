"""Verificar evidencia real de la entrega sin volver a seleccionar modelos."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eva2.data import EXPECTED_SHA256, save_json
from src.eva2.supervised import metrics
from src.eva2.integrity import source_integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verificar sin reescribir el acta JSON")
    args = parser.parse_args()
    report = ROOT / "reports/eva2"
    checks = {}
    def check(label, condition):
        checks[label] = bool(condition)
        if not condition:
            raise AssertionError(label)
    required = ["docs/informe_tecnico_eva2.md", "docs/informe_ejecutivo.md",
                "notebooks/05_proyecto_completo_eva2.ipynb", "README.md", "requirements.txt",
                "models/eva2/popularity_pipeline.joblib", "models/eva2/music_segments.joblib",
                "data/examples/tracks_example.csv", "src/eva2/run.py"]
    check("deliverables_exist", all((ROOT / p).is_file() for p in required))
    actual_hash = hashlib.sha256((ROOT / "data/raw/Spotify_Tracks_Dataset.csv").read_bytes()).hexdigest()
    check("dataset_sha256", actual_hash == EXPECTED_SHA256)
    provenance = source_integrity(ROOT, EXPECTED_SHA256)
    check("source_integrity_valid", provenance["data_hash"] == EXPECTED_SHA256)
    manifest = pd.read_csv(report / "split_manifest.csv", dtype={"audio_group": str})
    train, test = [manifest.query("split == @label") for label in ["train", "test"]]
    check("unique_track_ids", manifest.track_id.is_unique)
    check("no_shared_audio_groups", not (set(train.audio_group) & set(test.audio_group)))
    cv = pd.read_csv(report / "cv_summary.csv")
    selection = json.loads((report / "selection.json").read_text(encoding="utf-8"))
    evaluation = json.loads((report / "evaluation.json").read_text(encoding="utf-8"))
    check("selection_follows_cv", cv.sort_values("cv_mae").iloc[0].model == selection["selected_model"])
    check("test_not_used_for_selection", selection["test_used_for_selection"] is False)
    predictors = set(selection["features"])
    check("no_target_or_identifiers_in_features", not predictors & {"popularity", "track_id", "artists", "album_name", "track_name"})
    pred = pd.read_csv(report / "test_predictions.csv")
    calculated = metrics(pred.popularity, pred.prediction)
    check("metrics_match_saved_predictions", all(np.isclose(calculated[key], evaluation["metrics"][key]) for key in calculated))
    check("test_manifest_matches_predictions", set(test.track_id) == set(pred.track_id))
    check("predictions_finite_and_bounded", np.isfinite(pred.prediction).all() and pred.prediction.between(0, 100).all())
    diag = pd.read_csv(report / "clustering_diagnostics.csv")
    cluster = json.loads((report / "clustering.json").read_text(encoding="utf-8"))
    check("k_one_to_ten", diag.k.tolist() == list(range(1, 11)))
    check("clustering_excludes_target_and_genre", not set(cluster["features"]) & {"popularity", "track_genre"})
    check("cluster_minimum_size", diag.query("k == @cluster['selected_k']").smallest_cluster_pct.iloc[0] >= 3)
    variance = pd.read_csv(report / "pca_variance.csv")
    check("pca_variance_sums_one", np.isclose(variance.explained_variance_ratio.sum(), 1))
    notebook = nbformat.read(ROOT / "notebooks/05_proyecto_completo_eva2.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    check("all_notebook_code_cells_executed", bool(cells) and all(c.execution_count is not None for c in cells))
    check("notebook_has_no_error_outputs", not any(o.output_type == "error" for c in cells for o in c.outputs))
    figures = list((ROOT / "images/eva2").glob("*.png"))
    check("ten_figures_present", len(figures) == 10 and all(p.stat().st_size > 10_000 for p in figures))
    broken = []
    for name in ["README.md", "docs/informe_tecnico_eva2.md", "docs/informe_ejecutivo.md"]:
        source = ROOT / name
        for link in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            if not link.startswith(("https://", "http://", "#")) and not (source.parent / link.split('#')[0]).exists():
                broken.append(f"{name}: {link}")
    check("current_document_links_resolve", not broken)
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=ROOT, capture_output=True, text=True)
    check("unittest_passes", test.returncode == 0)
    result = {"passed": True, "checks": checks, "notebook_code_cells": len(cells),
              "test_output": test.stdout + test.stderr, "broken_links": broken,
              "measured_metrics": calculated, "dataset_sha256": actual_hash,
              "provenance": provenance}
    if not args.check_only:
        save_json(report / "verification.json", result)
    print(json.dumps({"passed": True, "checks": len(checks), "notebook_code_cells": len(cells),
                      "metrics": calculated, "source_mode": provenance["mode"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
