"""Pruebas de riesgos concretos: fuga, contrato y persistencia."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.eva2.data import AUDIO, ROOT, audio_groups, grouped_split, prepare


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


if __name__ == "__main__":
    unittest.main()
