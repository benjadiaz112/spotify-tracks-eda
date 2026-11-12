"""Pruebas de contrastes pareados y selección de referencias sin mirar test."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.eva3.contrasts import audit_contrasts, paired_group_intervals
from src.eva3.data import ROOT


class EP3ContrastTests(unittest.TestCase):
    def make_fixture(self, out: Path, *, bad_group=False, export_fixed=False) -> None:
        train = pd.DataFrame({"track_id": list("abcd"), "popularity": [20, 30, 40, 50],
                              "fold": [1, 2 if bad_group else 1, 2, 2]})
        test = pd.DataFrame({"track_id": list("xyzw"), "popularity": [20, 30, 40, 50]})
        manifest = pd.concat([train.drop(columns="fold").assign(group_id=["g1", "g1", "g2", "g3"], split="train"),
                              test.assign(group_id=["g4", "g4", "g5", "g6"], split="test")])
        manifest.to_csv(out / "split_manifest.csv", index=False)
        errors_train = {
            "hgb_metadatos_base": [4, 4, 2, 2], "hgb_metadatos_ajustado": [2, 2, 1, 1],
            "arbol_metadatos": 5, "extra_trees_metadatos": 2, "ridge_metadatos": 0.8,
            "mezcla_uniforme": 1.4, "mezcla_calibrada": 0.9, "hgb_ep2": 3,
        }
        errors_test = {
            "hgb_metadatos_base": 3, "hgb_metadatos_ajustado": 1,
            "arbol_metadatos": 5, "extra_trees_metadatos": 0.2, "ridge_metadatos": 3,
            "mezcla_uniforme": 1.3, "mezcla_calibrada": 1.2, "hgb_ep2": 4,
        }
        fold_rows = []
        for model, errors in errors_train.items():
            train[model] = train.popularity + np.asarray(errors)
            for fold, part in train.groupby("fold"):
                fold_rows.append({"model": model, "fold": fold, "val_rows": len(part),
                                  "mae": abs(part[model] - part.popularity).mean()})
        for model, errors in errors_test.items():
            test[model] = test.popularity + errors
        if export_fixed:
            test["ridge_fijo"] = test.popularity + 2
        train.to_csv(out / "oof_predictions.csv", index=False)
        test.to_csv(out / "test_predictions.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
        trials = [{"model": family, "fold": fold, "params": json.dumps(params), "mae": mae,
                   "val_rows": int(train.fold.eq(fold).sum())}
                  for fold in [1, 2]
                  for family, params, mae in [("extra_trees_metadatos", {"leaf": 5}, 2),
                                              ("ridge_metadatos", {"alpha": 6}, 1)]]
        pd.DataFrame(trials).to_csv(out / "search_trials.csv", index=False)
        payload = {
            "test_used_for_selection": False, "selected_model": "ridge_metadatos",
            "blend_weights": {"mezcla_uniforme": dict.fromkeys(["hgb_metadatos_ajustado", "extra_trees_metadatos", "ridge_metadatos"], 1/3),
                              "mezcla_calibrada": {"hgb_metadatos_base": 0.3, "extra_trees_fijo": 0.3, "ridge_fijo": 0.4}},
            "bag_fixed_params": {"leaf": 5}, "ridge_fixed_params": {"alpha": 6},
            "best_params": {"extra_trees_metadatos": {"leaf": 4}, "ridge_metadatos": {"alpha": 2}},
        }
        (out / "selection.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_exact_paired_gain_and_frozen_component(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            out = Path(name)
            self.make_fixture(out)
            table = audit_contrasts(out, n_boot=25).set_index("contrast")
            tuned = table.loc["ajuste_hgb"]
            self.assertAlmostEqual(tuned.cv_delta_mae_points, 1.5)
            self.assertAlmostEqual(tuned.holdout_delta_mae_points, 2)
            self.assertAlmostEqual(tuned.holdout_delta_ci95_low, 2)
            self.assertAlmostEqual(tuned.holdout_delta_ci95_high, 2)
            self.assertFalse(tuned.holdout_ci_includes_zero)
            # ExtraTrees es mejor en test, pero la referencia sigue siendo Ridge elegido por CV.
            self.assertEqual(table.loc["mezcla_uniforme_vs_componente", "reference"], "ridge_metadatos")
            calibrated = table.loc["mezcla_calibrada_vs_componente"]
            self.assertEqual(calibrated.reference, "ridge_fijo")
            self.assertAlmostEqual(calibrated.cv_delta_mae_points, 0.1)
            self.assertTrue(pd.isna(calibrated.holdout_delta_mae_points))
            self.assertEqual(calibrated.holdout_status, "referencia_fija_sin_prediccion_guardada")

    def test_saved_fixed_reference_allows_holdout_delta(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            out = Path(name)
            self.make_fixture(out, export_fixed=True)
            result = audit_contrasts(out, n_boot=25).set_index("contrast")
            self.assertAlmostEqual(result.loc["mezcla_calibrada_vs_componente", "holdout_delta_mae_points"], 0.8)

    def test_group_cannot_cross_oof_folds(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            out = Path(name)
            self.make_fixture(out, bad_group=True)
            with self.assertRaisesRegex(ValueError, "cruza folds"):
                audit_contrasts(out, n_boot=25)

    def test_bootstrap_resamples_paired_components(self):
        intervals = paired_group_intervals(pd.Series(["a", "a", "b"]), np.array([[3, -2], [3, -2], [-1, -2]]), 300)
        np.testing.assert_allclose(intervals, [[-1, 3], [-2, -2]])


if __name__ == "__main__":
    unittest.main()
