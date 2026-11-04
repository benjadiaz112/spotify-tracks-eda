"""Controles de procedencia y de fuga en el holdout de EP3."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.eva3.data import ROOT, grouped_split, prepare, recording_keys, source_provenance, union_groups


# Partición fijada por el protocolo de datos antes de entrenar y comparar candidatos.
FROZEN_MANIFEST_SHA256 = "db12b3d3172ce9acd9c175b4093d9f6e14f2ec76849f4c71d9042a3e8b796130"


class EP3DataTests(unittest.TestCase):
    def test_recording_keys_normalize_without_collapsing_versions(self):
        frame = pd.DataFrame({
            "artists": [" ÁRTISTA ; Invitado", "A\u0301rtista", "Artista", "Ártista", "Ártista",
                        "Various Artists", "; Ártista", "Ártista", "Invitado;Ártista"],
            "track_name": [" Una   canción ", "UNA CANCIÓN", "Una canción", "Una canción (Live)",
                           "Una cancion", "Una canción", "Una canción", "Intro", "Una canción"],
        })
        keys = recording_keys(frame).tolist()
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(len({keys[i] for i in [0, 2, 3, 4, 8]}), 5)
        self.assertEqual(keys[5:8], ["", "", ""])

    def test_union_links_transitively_without_merging_blank_keys(self):
        table = pd.DataFrame({
            "id": ["C", "A", "B", "D", "E"],
            "audio": ["a1", "a1", "a2", "a3", "a4"],
            "key": ["r1", "r2", "r2", "", ""],
        })
        groups = union_groups(table.id, table.audio, table.key)
        self.assertEqual(groups.tolist(), ["g_A", "g_A", "g_A", "g_D", "g_E"])
        shuffled = table.sample(frac=1, random_state=9)
        result = union_groups(shuffled.id, shuffled.audio, shuffled.key)
        pd.testing.assert_series_equal(groups, result.sort_index())

    def test_provenance_rejects_altered_source(self):
        temporary = ROOT / "tmp"
        temporary.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            source = Path(folder) / "changed.csv"
            source.write_bytes(b"track_id,popularity\ntrack,101\n")
            with self.assertRaisesRegex(ValueError, "CSV distinto"):
                source_provenance(source)

    def test_frozen_holdout_is_disjoint_and_independent_of_row_order(self):
        temporary = ROOT / "tmp"
        temporary.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            out = Path(folder)
            data = prepare(ROOT, out)
            manifest = out / "split_manifest.csv"
            self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), FROZEN_MANIFEST_SHA256)
            frame = pd.read_csv(manifest, dtype="str", keep_default_na=False)
            train = frame.loc[frame.split.eq("train")]
            test = frame.loc[frame.split.eq("test")]
            self.assertEqual((len(train), len(test)), (71582, 17913))
            for column in ["track_id", "group_id", "audio_group", "recording_key"]:
                self.assertFalse((set(train[column]) & set(test[column])) - {""}, column)
            order = data["clean"].sample(frac=1, random_state=17).index
            clean_shuffled = data["clean"].loc[order]
            groups_shuffled = data["groups"].loc[order]
            _, shuffled_test_idx = grouped_split(clean_shuffled, groups_shuffled)
            self.assertEqual(set(clean_shuffled.iloc[shuffled_test_idx].track_id), set(test.track_id))


if __name__ == "__main__":
    unittest.main()
