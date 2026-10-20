"""La entrega debe funcionar sin Git y detectar alteraciones de sus fuentes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eva2.data import ROOT
from src.eva2.integrity import source_integrity


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.contents = {"data/raw/Spotify_Tracks_Dataset.csv": b"fixture,not_production\n1,2\n",
                         "src/example.py": b"value = 1\n", "requirements.txt": b"fixture\n"}
        for name, value in self.contents.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        self.digest = hashlib.sha256(self.contents["data/raw/Spotify_Tracks_Dataset.csv"]).hexdigest()
        self.manifest = {"commit": "fixture-only", "files": {
            name: {"sha256": hashlib.sha256(value).hexdigest()} for name, value in self.contents.items()}}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "MANIFIESTO_ENTREGA.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_archive_never_invokes_git(self):
        with patch("src.eva2.integrity.subprocess.check_output", side_effect=AssertionError("No invocar Git")):
            result = source_integrity(self.root, self.digest)
        self.assertEqual(result["mode"], "archive_manifest")
        self.assertEqual(result["source_files_checked"], 3)

    def test_modified_source_is_rejected(self):
        (self.root / "src/example.py").write_text("value = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Archivo de origen alterado"):
            source_integrity(self.root, self.digest)

    def test_regenerated_outputs_do_not_invalidate_source(self):
        output = self.root / "reports/eva2"
        output.mkdir(parents=True)
        (output / "run.json").write_text('{"new_execution": true}', encoding="utf-8")
        self.assertEqual(source_integrity(self.root, self.digest)["mode"], "archive_manifest")

    def test_modified_input_is_rejected(self):
        (self.root / "data/raw/Spotify_Tracks_Dataset.csv").write_bytes(b"alterado")
        with self.assertRaisesRegex(ValueError, "CSV no coincide"):
            source_integrity(self.root, self.digest)

    def test_manifest_cannot_reference_parent(self):
        self.manifest["files"]["../outside.txt"] = {"sha256": "invalid"}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "fuera de la entrega"):
            source_integrity(self.root, self.digest)
