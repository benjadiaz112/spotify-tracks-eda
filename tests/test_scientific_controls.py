"""Regresiones contra las debilidades metodológicas de la revisión."""
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.eva2.data import ROOT, audit_target_labels
from src.eva2.supervised import hgb_pipeline
from src.features.prepare_features import MODEL_FEATURES


class ScientificControlTests(unittest.TestCase):
    def test_target_conflicts_are_quantified_without_modifying_reference(self):
        fixture = pd.read_csv(ROOT / "data/raw/Spotify_Tracks_Dataset.csv", nrows=2)
        raw = pd.concat([fixture.iloc[[0]]] * 3 + [fixture.iloc[[1]]], ignore_index=True)
        raw["popularity"] = [10, 11, 50, 30]
        raw["track_genre"] = ["a", "b", "c", "a"]
        clean = raw.drop_duplicates("track_id", keep="first").reset_index(drop=True)
        snapshot = clean.copy(deep=True)
        audit, conflicts = audit_target_labels(raw, clean, np.array([0]), np.array([1]))
        self.assertEqual(audit["conflicting_ids_raw"], 1)
        self.assertEqual(audit["maximum_range"], 40)
        self.assertEqual(audit["first_differs_from_median_retained"], 1)
        self.assertEqual(conflicts.iloc[0]["median"], 11)
        self.assertEqual(conflicts.iloc[0]["retained_target"], 10)
        self.assertEqual(conflicts.iloc[0]["split"], "train")
        pd.testing.assert_frame_equal(clean, snapshot)

    def test_genre_ablation_retains_all_other_variables(self):
        features = [name for name in MODEL_FEATURES if name != "track_genre"]
        full = hgb_pipeline({"model__l2_regularization": 10.0, "model__max_leaf_nodes": 31})
        reduced = hgb_pipeline({"model__l2_regularization": 10.0, "model__max_leaf_nodes": 31}, include_genre=False)
        sample = pd.read_csv(ROOT / "data/raw/Spotify_Tracks_Dataset.csv", nrows=100)
        with threadpool_limits(limits=4):
            reduced.fit(sample[features], sample.popularity)
            prediction = reduced.predict(sample.iloc[:2][features])
        self.assertEqual(set(MODEL_FEATURES) - set(reduced.feature_names_in_), {"track_genre"})
        self.assertTrue({"key", "mode", "time_signature"} <= set(reduced.feature_names_in_))
        self.assertTrue(np.isfinite(prediction).all())
        for key in ["max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization", "early_stopping", "random_state"]:
            self.assertEqual(full.named_steps["model"].get_params()[key], reduced.named_steps["model"].get_params()[key])
