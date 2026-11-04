"""Comprobar evidencia, partición, selección y predicciones EP3 sin reentrenar."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eva3.data import ARCHIVE_MEMBER_SHA256, EXPECTED_SHA256, RAW_RELATIVE_PATH
from src.eva2.integrity import source_integrity
from src.features.prepare_features import MODEL_FEATURES


TEAM = ("Benjamín Palomino", "Benjamín Díaz")
TOLERANCES = (5, 10, 15, 20)
MODEL_FILE = "models/eva3/popularity_pipeline.joblib"
NOTEBOOK = "notebooks/06_evaluacion_3.ipynb"
DOCUMENTS = (
    "docs/ep3/README.md", "docs/ep3/protocolo.md", "docs/ep3/informe_tecnico.md",
    "docs/ep3/informe_ejecutivo.md", "docs/ep3/protocolo_prerregistro.md",
)
MODELS = {
    "dummy_mediana", "hgb_ep2", "hgb_metadatos_base", "hgb_metadatos_ajustado",
    "arbol_metadatos", "extra_trees_metadatos", "ridge_metadatos",
    "mezcla_uniforme", "mezcla_calibrada",
}
FIGURES = ("01_objetivo.png", "02_modelos_cv.png", "03_reserva.png", "04_segmentos.png")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(observed, predicted) -> dict:
    """Recalcular métricas desde las predicciones, sin importar código de modelado."""
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    error = np.abs(y - p)
    mse = float(np.mean(np.square(error)))
    variance = float(np.mean(np.square(y - y.mean())))
    r2 = 1 - mse / variance if variance else float(mse == 0)
    return {"mae": float(error.mean()), "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": r2, **{f"within_{d}": float(np.mean(error <= d)) for d in TOLERANCES}}


def same_metrics(measured: dict, registered, prefix: str = "") -> bool:
    return all(np.isclose(value, registered[prefix + key], rtol=1e-8, atol=1e-10)
               for key, value in measured.items())


def verify(root: Path = ROOT) -> dict:
    report = root / "reports/eva3"
    checks = {}

    def check(label: str, condition) -> None:
        checks[label] = bool(condition)
        if not condition:
            raise ValueError(f"Control EP3 fallido: {label}")

    def load(name: str) -> dict:
        return json.loads((report / name).read_text(encoding="utf-8"))

    def csv(name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(report / name, **kwargs)

    required = (*DOCUMENTS, NOTEBOOK, MODEL_FILE, "requirements-lock-windows-py314.txt",
                "src/eva3/run.py", "src/eva3/predict.py", "scripts/verify_ep3.py")
    absent = [name for name in required if not (root / name).is_file()]
    check("archivos requeridos" + (f": faltan {', '.join(absent)}" if absent else ""), not absent)
    audit = load("data_audit.json")
    raw_path = root / RAW_RELATIVE_PATH
    raw_bytes = raw_path.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    normalized = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()
    check("identidad del dataset", raw_hash in {EXPECTED_SHA256, ARCHIVE_MEMBER_SHA256}
          and normalized == ARCHIVE_MEMBER_SHA256 == audit["normalized_sha256"]
          and raw_hash == audit["raw_sha256"])
    if (root / "MANIFIESTO_ENTREGA.json").is_file():
        check("manifiesto de código e insumos", source_integrity(root, raw_hash)["data_hash"] == raw_hash)

    manifest_path = report / "split_manifest.csv"
    manifest = csv(manifest_path.name, dtype={name: str for name in
                   ("track_id", "group_id", "audio_group", "recording_key")}, keep_default_na=False)
    check("huella de partición", digest(manifest_path) == audit["split_manifest_sha256"])
    check("manifiesto de IDs únicos", manifest.track_id.is_unique and manifest.track_id.ne("").all()
          and set(manifest.split) == {"train", "test"})
    train = manifest[manifest.split.eq("train")].set_index("track_id")
    test = manifest[manifest.split.eq("test")].set_index("track_id")
    check("conteos de partición", len(train) == audit["train_rows"] and len(test) == audit["test_rows"]
          and len(manifest) == audit["clean_rows"]
          and train.group_id.nunique() == audit["train_groups"]
          and test.group_id.nunique() == audit["test_groups"])
    for key in ("group_id", "audio_group", "recording_key"):
        check(f"sin cruce de {key}", not ((set(train[key]) & set(test[key])) - {""}))
    check("partición sin cruce de ID", not set(train.index).intersection(test.index))
    for key in ("audio_group", "recording_key"):
        by_key = manifest.loc[manifest[key].ne("")].groupby(key).group_id.nunique()
        check(f"{key} no fragmentada entre componentes", by_key.eq(1).all())

    original = pd.read_csv(raw_path, usecols=["track_id", "popularity", "artists", "album_name", "track_name"])
    valid_target = original.dropna(subset=["artists", "album_name", "track_name"]).drop_duplicates("track_id")
    check("etiqueta original conservada", len(original) == audit["raw_rows"]
          and np.array_equal(manifest.popularity.to_numpy(),
                             manifest.track_id.map(valid_target.set_index("track_id").popularity).to_numpy()))
    cleaning = csv("cleaning.csv")
    check("cadena de limpieza reconciliada", int(cleaning.filas_antes.iloc[0]) == len(original)
          and int(cleaning.filas_despues.iloc[-1]) == len(manifest)
          and np.array_equal(cleaning.filas_antes.iloc[1:].to_numpy(), cleaning.filas_despues.iloc[:-1].to_numpy())
          and (cleaning.filas_antes - cleaning.filas_despues).equals(cleaning.filas_eliminadas)
          and cleaning.filas_eliminadas.ge(0).all())
    lineage_path = report / "row_lineage.csv.gz"
    lineage_meta = audit["row_lineage"]
    lineage = csv("row_lineage.csv.gz", dtype={"track_id": str, "group_id": str}, keep_default_na=False)
    check("hash y descartes por etapa del linaje", digest(lineage_path) == lineage_meta["gzip_sha256"]
          and hashlib.sha256(gzip.decompress(lineage_path.read_bytes())).hexdigest() == lineage_meta["csv_sha256"]
          and all(int(lineage.cleaning_step.eq(row["cleaning_step"]).sum()) == row["excluded_rows"]
                  for row in lineage_meta["rule_counts"]))
    check("trazabilidad de cada fila bruta", np.array_equal(lineage.raw_row.to_numpy(), np.arange(len(original)))
          and np.array_equal(lineage.track_id, original.track_id) and lineage.decision.ne("").all())
    retained = lineage.loc[lineage.split.isin(["train", "test"])].set_index("track_id")
    lineage_truth = manifest.set_index("track_id").reindex(retained.index)
    check("trazabilidad de cada fila retenida", retained.index.is_unique and len(retained) == len(manifest)
          and set(retained.index) == set(manifest.track_id)
          and np.array_equal(retained.group_id, lineage_truth.group_id)
          and np.array_equal(retained.split, lineage_truth.split))
    conflicts = csv("target_conflicts.csv")
    original_targets = original.groupby("track_id").popularity.nunique()
    check("desacuerdos de etiqueta rastreables", conflicts.track_id.is_unique
          and set(conflicts.track_id) == set(original_targets[original_targets.gt(1)].index)
          and len(conflicts) == audit["target_conflicts"]["conflicting_ids_raw"])

    # Las columnas audio_group de las predicciones EP3 llevan el componente unión.
    identity = {"track_id": str, "audio_group": str}
    oof = csv("oof_predictions.csv", dtype=identity).set_index("track_id")
    predictions = csv("test_predictions.csv", dtype=identity).set_index("track_id")
    for label, rows, split in [("OOF", oof, train), ("test", predictions, test)]:
        check(f"{label}: IDs corresponden a la partición", rows.index.is_unique and set(rows.index) == set(split.index))
        check(f"{label}: grupos y objetivo coinciden", np.array_equal(rows.audio_group, split.loc[rows.index].group_id)
              and np.array_equal(rows.popularity, split.loc[rows.index].popularity))
        values = rows[sorted(MODELS)].to_numpy(dtype=float)
        check(f"{label}: predicciones finitas y en rango", np.isfinite(values).all()
              and (values >= 0).all() and (values <= 100).all())
    check("grupo completo en cada fold OOF", set(oof.fold) == {1, 2, 3, 4, 5}
          and oof.groupby("audio_group").fold.nunique().eq(1).all())

    selection, evaluation = load("selection.json"), load("evaluation.json")
    cv_folds = csv("cv_folds.csv")
    cv_summary = csv("cv_summary.csv").set_index("model")
    final = csv("supervised_metrics.csv").set_index("model")
    selected = selection["selected_model"]
    check("mismos candidatos en CV y reserva", set(cv_summary.index) == set(final.index) == MODELS
          and cv_summary.index.is_unique and final.index.is_unique and set(cv_folds.model) == MODELS
          and len(cv_folds) == 5 * len(MODELS) and not cv_folds.duplicated(["model", "fold"]).any())
    check("criterio de selección aplicado", cv_summary.reset_index().sort_values(["cv_mae", "model"]).iloc[0].model == selected
          and selected == evaluation["selected_model"] and selection["test_used_for_selection"] is False)
    inputs = set(selection["input_columns"])
    allowed = set(MODEL_FEATURES) | ({"artists", "album_name"} if selection["historical_metadata"] else set())
    check("entradas declaradas y sin identificadores de grabación", inputs == set(selection["features"]) == allowed)
    check("tolerancias y dominio declarados", selection["tolerances"] == list(TOLERANCES)
          and selection["prediction_range"] == [0, 100])

    fold_counts_ok = True
    folds_ok = True
    for row in cv_folds.itertuples(index=False):
        fold = oof[oof.fold.eq(row.fold)]
        fold_counts_ok &= len(fold) == row.val_rows and len(oof) - len(fold) == row.train_rows
        folds_ok &= same_metrics(metrics(fold.popularity, fold[row.model]), row._asdict())
    check("conteos de folds reproducidos", fold_counts_ok)
    check("métricas de folds reproducidas desde OOF", folds_ok)
    check("estadísticas CV reproducidas", all(
        same_metrics(part[list(metrics(oof.popularity, oof[model]))].mean().to_dict(), cv_summary.loc[model], "cv_")
        and np.isclose(cv_summary.loc[model, "cv_std"], part.mae.std(ddof=0))
        and np.isclose(cv_summary.loc[model, "cv_oof_mae"], metrics(oof.popularity, oof[model])["mae"])
        for model, part in cv_folds.groupby("model")
    ))
    check("métricas de test reproducidas desde predicciones", all(
        same_metrics(metrics(predictions.popularity, predictions[model]), final.loc[model]) for model in MODELS
    ))
    winner_metrics = metrics(predictions.popularity, predictions.prediction)
    check("predicción del ganador y referencias consistentes", np.allclose(predictions.prediction, predictions[selected])
          and np.allclose(predictions.baseline_ep2, predictions.hgb_ep2)
          and np.allclose(predictions.baseline_median, float(train.popularity.median()))
          and same_metrics(winner_metrics, evaluation["metrics"]))
    check("cobertura de tolerancias y umbral propuesto", all(
        winner_metrics[f"within_{a}"] <= winner_metrics[f"within_{b}"] for a, b in zip(TOLERANCES, TOLERANCES[1:])
    ) and evaluation["coverage_target_met"] == (winner_metrics["within_15"] > 0.80))

    trials = csv("search_trials.csv")
    summaries = csv("search_summary.csv")
    per_trial = trials.groupby(["model", "trial"], sort=False)
    check("grilla completa y consistente", not trials.duplicated(["model", "trial", "fold"]).any()
          and all(set(g.fold) == {1, 2, 3, 4, 5} and g.params.nunique() == 1 for _, g in per_trial)
          and set(per_trial.groups) == set(zip(summaries.model, summaries.trial))
          and len(summaries) == len(per_trial))
    check("resúmenes de búsqueda reproducidos", all(
        np.isclose(row.cv_mae, per_trial.get_group((row.model, row.trial)).mae.mean())
        for row in summaries.itertuples(index=False)
    ))
    best = summaries.sort_values(["model", "cv_mae", "trial"]).drop_duplicates("model").set_index("model")
    check("hiperparámetros ganadores coinciden con la búsqueda", set(best.index) == set(selection["best_trials"])
          and all(int(best.loc[model, "trial"]) == trial
                  and json.loads(best.loc[model, "params"]) == selection["best_params"][model]
                  for model, trial in selection["best_trials"].items()))
    fixed = ["hgb_metadatos_ajustado", "extra_trees_metadatos", "ridge_metadatos"]
    check("mezcla uniforme coincide con sus componentes", all(
        np.allclose(frame.mezcla_uniforme, frame[fixed].mean(axis=1)) for frame in (oof, predictions)
    ))
    weights = [*selection["blend_weights"].values(),
               *(fold["weights"] for fold in selection["calibration_by_fold"])]
    check("pesos de ensambles convexos", len(weights) == 7 and all(
        len(w) == 3 and all(0 <= v <= 1 for v in w.values()) and np.isclose(sum(w.values()), 1) for w in weights
    ))
    calibrated = selection["blend_weights"]["mezcla_calibrada"]
    check("mezcla calibrada coincide con sus componentes en reserva", np.allclose(
        predictions.mezcla_calibrada, sum(predictions[name] * weight for name, weight in calibrated.items())
    ))
    check("calibración pertenece a ajuste de cada fold", all(
        row["fit_rows"] > 0 and row["calibration_rows"] > 0
        and row["fit_rows"] + row["calibration_rows"] == int(oof.fold.ne(row["fold"]).sum())
        for row in selection["calibration_by_fold"]
    ))
    contrasts = csv("controlled_contrasts.csv")
    contrast_folds = csv("controlled_contrasts_folds.csv")
    expected_contrasts = {"ajuste_hgb", "ensamble_arboles", "mezcla_uniforme_vs_componente",
                          "mezcla_calibrada_vs_componente", "ganador_vs_ep2"}
    check("contrastes controlados y pareados completos", contrasts.contrast.is_unique
          and set(contrasts.contrast) == set(contrast_folds.contrast) == expected_contrasts
          and not contrast_folds.duplicated(["contrast", "fold"]).any()
          and all(set(g.fold) == {1, 2, 3, 4, 5} for _, g in contrast_folds.groupby("contrast")))
    check("deltas de contrastes por fold reproducidos", np.allclose(contrast_folds.delta_mae_points,
          contrast_folds.mae_reference - contrast_folds.mae_candidate)
          and np.allclose(contrasts.cv_delta_mae_points,
                          contrasts.contrast.map(contrast_folds.groupby("contrast").delta_mae_points.mean())))
    contrasts_ok = True
    for row in contrasts.itertuples(index=False):
        ref = row.holdout_reference_prediction_column
        if pd.isna(ref):
            contrasts_ok &= pd.isna(row.holdout_delta_mae_points)
        else:
            ref_mae = metrics(predictions.popularity, predictions[ref])["mae"]
            cand_mae = metrics(predictions.popularity, predictions[row.candidate])["mae"]
            contrasts_ok &= (np.isclose(row.holdout_delta_mae_points, ref_mae - cand_mae)
                             and row.holdout_delta_ci95_low <= row.holdout_delta_ci95_high)
    check("deltas de contrastes en reserva reproducidos", contrasts_ok)

    bootstrap = load("bootstrap.json")
    check("bootstrap de grupos con puntos reproducidos", bootstrap["groups"] == test.group_id.nunique()
          and bootstrap["resamples"] > 0 and all(
              np.isclose(bootstrap["intervals"][metric]["point"], winner_metrics[metric])
              and bootstrap["intervals"][metric]["lo_95"] <= bootstrap["intervals"][metric]["hi_95"]
              for metric in ("mae", "r2", *(f"within_{d}" for d in TOLERANCES))
          ))
    artist_bootstrap = load("bootstrap_artist_seen.json")
    check("incertidumbre separada por disponibilidad de historial", set(artist_bootstrap) == {"True", "False"}
          and all(
              artist_bootstrap[str(seen)]["rows"] == len(part)
              and artist_bootstrap[str(seen)]["groups"] == part.audio_group.nunique()
              and all(
                  np.isclose(artist_bootstrap[str(seen)]["intervals"][name]["point"], measured[name])
                  and artist_bootstrap[str(seen)]["intervals"][name]["lo_95"]
                  <= artist_bootstrap[str(seen)]["intervals"][name]["hi_95"]
                  for name in ("r2", "mae", "within_15")
              )
              for seen, part in predictions.groupby("artist_seen", observed=True)
              for measured in [metrics(part.popularity, part.prediction)]
          ))
    subgroups = csv("holdout_subgroups.csv")
    check("cada auditoría cubre toda la reserva", set(subgroups.dimension) ==
          {"artist_seen", "album_seen", "track_genre", "explicit", "target_band"}
          and subgroups.groupby("dimension").n.sum().eq(len(predictions)).all())
    business = load("business.json")
    budget = max(1, int(business["budget_fraction"] * len(test)))
    threshold = float(train.popularity.quantile(business["threshold_quantile_train"]))
    top = predictions.reset_index().sort_values(["prediction", "track_id"], ascending=[False, True]).head(budget)
    precision = float(top.popularity.ge(threshold).mean())
    expected = float(test.popularity.ge(threshold).mean())
    check("KPI de negocio reproducido", budget == business["budget_songs"]
          and np.isclose(threshold, business["threshold_train"])
          and np.isclose(precision, business["selected_model"]["precision_at_budget"])
          and np.isclose(expected, business["uniform_reference"]["precision_expected"])
          and np.isclose((precision - expected) * 100, business["precision_gain_pp"]))

    diag, clustering, run = csv("clustering_diagnostics.csv"), load("clustering.json"), load("run.json")
    check("búsqueda y regla de K reproducidas", diag.k.tolist() == list(range(1, 11))
          and int(diag.query("k >= 2 and smallest_cluster_pct >= 3").sort_values(
              ["silhouette", "k"], ascending=[False, True]).iloc[0].k) == clustering["selected_k"] == run["selected_k"])
    for label, split in [("train", train), ("test", test)]:
        profiles = csv(f"cluster_profiles_{label}.csv")
        check(f"perfiles {label}: tamaño y participación", profiles.n.sum() == len(split)
              and len(profiles) == run["selected_k"] and np.isclose(profiles.pct.sum(), 100.0))
    check("figuras completas", all((root / "images/eva3" / f).is_file()
          and (root / "images/eva3" / f).stat().st_size > 5_000 for f in FIGURES))
    check("modelo corresponde a corrida", run["completed"] is True and run["selected_model"] == selected
          and digest(root / MODEL_FILE) == run["model_sha256"])
    example_input = csv("prediction_input_example.csv").set_index("track_id")
    example = csv("prediction_example.csv").set_index("track_id")
    check("ejemplo de inferencia reproducido en reserva", example.index.is_unique and 0 < len(example) <= len(test)
          and "popularity" not in example_input.columns and inputs.issubset(example_input.columns)
          and example.index.equals(example_input.index) and set(example.index).issubset(predictions.index)
          and np.allclose(example.estimated_popularity, predictions.loc[example.index].prediction))

    notebook = nbformat.read(root / NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    check("notebook ejecutado sin errores", bool(cells) and all(c.execution_count is not None for c in cells)
          and not any(output.output_type == "error" for cell in cells for output in cell.outputs))
    identity_text = [*[(root / n).read_text(encoding="utf-8") for n in DOCUMENTS],
                     "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")]
    check("integrantes EP3 en documentación y cuaderno", all(
        all(n in content for n in TEAM) and "Freymuth" not in content for content in identity_text
    ))
    links = [(name, link) for name in DOCUMENTS
             for link in re.findall(r"\]\(([^)]+)\)", (root / name).read_text(encoding="utf-8"))]
    broken = [(name, link) for name, link in links if not link.startswith(("https://", "http://", "#"))
              and not ((root / name).parent / link.split("#")[0]).exists()]
    check("enlaces internos de EP3" + (f": {broken}" if broken else ""), not broken)
    return {"passed": True, "checks": checks, "check_count": len(checks), "model": selected,
            "metrics": winner_metrics, "dataset_sha256": raw_hash, "test_rows": len(test),
            "notebook_code_cells": len(cells), "scope": "verificación de evidencias sin reentrenar"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="No reescribir el acta JSON")
    args = parser.parse_args()
    result = verify(ROOT)
    if not args.check_only:
        (ROOT / "reports/eva3/verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
