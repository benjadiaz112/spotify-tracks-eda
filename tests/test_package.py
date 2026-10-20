"""Comprobar el contenido real exportado y sus hashes de integridad."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.package_delivery import build_archive
from src.eva2.data import ROOT


class PackageTests(unittest.TestCase):
    def test_archive_contains_academic_files_and_valid_hashes(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            root = Path(folder)
            contents = {
                "README.md": b"Academic delivery\n",
                "data/raw/example.csv": b"fixture,not_production\n",
                "docs/informe_tecnico_eva2.md": b"Technical report\n",
                "docs/defensa_individual.md": b"Personal preparation\n",
                "docs/equipo_y_git.md": b"Internal coordination\n",
                "docs/acta_verificacion.md": b"Internal record\n",
            }
            for name, data in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            output = root / "output/delivery.zip"
            manifest = build_archive(root, list(contents), output)
            expected = {"README.md", "data/raw/example.csv", "docs/informe_tecnico_eva2.md"}
            self.assertEqual(set(manifest["files"]), expected)
            self.assertEqual(set(manifest), {"format_version", "files"})
            with ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), {
                    "spotify-eva2/" + name for name in expected
                } | {"spotify-eva2/MANIFIESTO_ENTREGA.json"})
                self.assertEqual(json.loads(archive.read("spotify-eva2/MANIFIESTO_ENTREGA.json")), manifest)
                for name in expected:
                    value = archive.read("spotify-eva2/" + name)
                    self.assertEqual(value, contents[name])
                    self.assertEqual(manifest["files"][name]["sha256"], hashlib.sha256(value).hexdigest())
                    self.assertEqual(manifest["files"][name]["bytes"], len(value))
