"""Pruebas de riesgos concretos: fuga, contrato y persistencia."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.eva2.data import AUDIO, ROOT, audio_groups, grouped_split, prepare
from src.eva2.predict import predict, validate_input
from src.eva2.supervised import candidates
from src.features.prepare_features import MODEL_FEATURES, CONTINUOUS_FEATURES
import joblib


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.read_csv(ROOT / "data/raw/Spotify_Tracks_Dataset.csv", nrows=600)
        cls.sample = raw.drop_duplicates("track_id").reset_index(drop=True)

    def test_group_ignores_target_genre_and_id(self):
        df = pd.concat([self.sample.iloc[:1]] * 2, ignore_index=True)
        df.loc[1, ["track_id", "track_genre", "popularity"]] = ["otro", "otro", 99]
        self.assertEqual(audio_groups(df).nunique(), 1)

    def test_duplicate_audio_cannot_cross_split(self):
        df = pd.concat([self.sample, self.sample.iloc[:40]], ignore_index=True)
        tr, te, groups = grouped_split(df)
        self.assertFalse(set(groups.iloc[tr]) & set(groups.iloc[te]))
        self.assertEqual(len(tr) + len(te), len(df))
        tr2, te2, _ = grouped_split(df)
        np.testing.assert_array_equal(tr, tr2)
        np.testing.assert_array_equal(te, te2)

    def test_hash_rejects_changed_dataset(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            root = Path(folder)
            p = root / "data/raw/Spotify_Tracks_Dataset.csv"
            p.parent.mkdir(parents=True)
            p.write_text("archivo modificado", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSV distinto"):
                prepare(root)

    def test_preprocessing_fits_only_training_values(self):
        pipeline, _ = candidates()["Regresion_lineal"]
        train = self.sample.iloc[:200]
        pipeline.fit(train[MODEL_FEATURES], train.popularity)
        scaler = pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"]
        means = scaler.mean_.copy()
        np.testing.assert_allclose(means, train[CONTINUOUS_FEATURES].mean())
        future = self.sample.iloc[200:202][MODEL_FEATURES].copy()
        future["duration_ms"] = 1e12
        pipeline.predict(future)
        np.testing.assert_array_equal(means, scaler.mean_)

    def test_input_rejects_missing_columns_and_invalid_audio(self):
        with self.assertRaisesRegex(ValueError, "Faltan variables"):
            validate_input(self.sample.drop(columns="energy"))
        bad = self.sample.iloc[:2].copy()
        bad.loc[bad.index[0], "energy"] = 2.0
        with self.assertRaisesRegex(ValueError, "entre 0 y 1"):
            validate_input(bad)

    def test_saved_pipeline_matches_predictions_and_handles_unknown_genre(self):
        path = ROOT / "models/eva2/popularity_pipeline.joblib"
        if not path.exists():
            self.skipTest("Entrenar primero para comprobar persistencia")
        sample = self.sample.iloc[:3].copy()
        result = predict(sample)
        bundle = joblib.load(path)
        expected = np.clip(bundle["pipeline"].predict(validate_input(sample)), 0, 100)
        np.testing.assert_allclose(result.predicted_popularity, expected)
        self.assertTrue(result.predicted_popularity.between(0, 100).all())
        sample["track_genre"] = "genero_nunca_observado"
        unknown = predict(sample)
        self.assertTrue(unknown.unknown_genre.all())
        self.assertTrue(np.isfinite(unknown.predicted_popularity).all())


if __name__ == "__main__":
    unittest.main()
