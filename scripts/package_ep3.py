"""Empaquetar solo la EP3, conservar un manifiesto y probar la copia descomprimida."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "spotify-ep3"
FIXED = {
    "requirements.txt", "requirements-lock-windows-py314.txt",
    "notebooks/06_evaluacion_3.ipynb",
    "data/raw/Spotify_Tracks_Dataset.csv", "reports/eva2/split_manifest.csv",
    "src/__init__.py", "src/eva2/__init__.py", "src/eva2/data.py",
    "src/eva2/supervised.py", "src/eva2/clustering.py", "src/eva2/integrity.py",
    "scripts/build_notebook_ep3.py", "scripts/verify_ep3.py", "scripts/package_ep3.py",
    "scripts/validate_environment.py",
}
FOLDERS = ("src/eva3/", "src/data/", "src/features/", "docs/ep3/", "reports/eva3/",
           "images/eva3/", "models/eva3/", "tests/test_eva3", "tests/test_ep3")
REQUIRED = FIXED | {
    "docs/ep3/README.md", "docs/ep3/informe_tecnico.md", "docs/ep3/informe_ejecutivo.md",
    "docs/ep3/Spotify_EP3.pptx",
    "reports/eva3/selection.json", "reports/eva3/evaluation.json",
    "reports/eva3/test_predictions.csv", "reports/eva3/verification.json",
    "reports/eva3/prediction_input_example.csv", "models/eva3/popularity_pipeline.joblib",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_archive(root: Path, dest: Path) -> dict:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("Registrar la entrega en Git antes de empaquetarla.")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    names = sorted(p for p in paths if p in FIXED or p.startswith(FOLDERS))
    if missing := REQUIRED.difference(names):
        raise ValueError(f"Faltan archivos de entrega versionados: {sorted(missing)}")
    readme = (
        "# Spotify Tracks · Evaluación Parcial 3\n\n"
        "**Benjamín Palomino y Benjamín Díaz · MLY1101 · Caso C.**\n\n"
        "Abre la [guía de la EP3](docs/ep3/README.md), el [informe técnico](docs/ep3/informe_tecnico.md), "
        "el [informe ejecutivo](docs/ep3/informe_ejecutivo.md) o el "
        "[notebook reproducible](notebooks/06_evaluacion_3.ipynb).\n\n"
        "La [presentación de la EP3](docs/ep3/Spotify_EP3.pptx) y el CSV bruto están incluidos.\n"
    ).encode("utf-8")
    files = {"README.md": readme} | {n: (root / n).read_bytes() for n in names}
    manifest = {
        "format_version": 1, "evaluation": "EP3",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
        "files": {name: {"sha256": sha(value), "bytes": len(value)} for name, value in files.items()},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".zip.part")
    with ZipFile(part, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in files.items():
            archive.writestr(f"{PREFIX}/{name}", data)
        archive.writestr(f"{PREFIX}/MANIFIESTO_ENTREGA.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    with ZipFile(part) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP dañado al empaquetar.")
    part.replace(dest)
    return manifest


def check_archive(dest: Path) -> dict:
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-ep3-", dir=scratch) as temp:
        target = Path(temp).resolve()
        project = target / PREFIX
        with ZipFile(dest) as archive:
            manifest_data = archive.read(f"{PREFIX}/MANIFIESTO_ENTREGA.json")
            manifest = json.loads(manifest_data)
            actual = archive.namelist()
            expected = {f"{PREFIX}/{name}" for name in manifest["files"]} | {f"{PREFIX}/MANIFIESTO_ENTREGA.json"}
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("Miembros inesperados o duplicados en el ZIP.")
            for name, info in manifest["files"].items():
                path = (project / name).resolve()
                if not path.is_relative_to(project):
                    raise ValueError(f"Ruta insegura: {name}")
                data = archive.read(f"{PREFIX}/{name}")
                if sha(data) != info["sha256"] or len(data) != info["bytes"]:
                    raise ValueError(f"Archivo dañado: {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (project / "MANIFIESTO_ENTREGA.json").write_bytes(manifest_data)
        env = dict(os.environ, PYTHONPATH="", GIT_CEILING_DIRECTORIES=str(target),
                   MPLCONFIGDIR=str(project / "tmp/matplotlib"), PYTHONUTF8="1")
        paths = env.get("PATH", "").split(os.pathsep)
        while git := shutil.which("git", path=os.pathsep.join(paths)):
            parent = Path(git).resolve().parent
            shorter = [p for p in paths if Path(p or ".").resolve() != parent]
            if shorter == paths:
                raise RuntimeError("No se pudo aislar Git del entorno de prueba.")
            paths = shorter
        env["PATH"] = os.pathsep.join(paths)
        commands = [
            ["scripts/validate_environment.py", "--check-only"],
            ["-m", "unittest", "discover", "-s", "tests", "-q"],
            ["scripts/verify_ep3.py", "--check-only"],
            ["-m", "src.eva3.predict", "reports/eva3/prediction_input_example.csv"],
        ]
        for args in commands:
            subprocess.run([sys.executable, *args], cwd=project, env=env, check=True, timeout=240)
        with (project / "reports/eva3/prediction_example.csv").open(encoding="utf-8", newline="") as file:
            expected_scores = list(csv.DictReader(file))
        with (project / "output/predictions_ep3.csv").open(encoding="utf-8", newline="") as file:
            actual_scores = list(csv.DictReader(file))
        if (not expected_scores or len(expected_scores) != len(actual_scores) or
                any(a["track_id"] != b["track_id"] or
                    abs(float(a["estimated_popularity"]) - float(b["estimated_popularity"])) > 1e-8
                    for a, b in zip(expected_scores, actual_scores))):
            raise ValueError("La inferencia descomprimida difiere de la evidencia publicada.")
        return {"passed": True, "files": len(manifest["files"]), "commit": manifest["commit"],
                "manifest_sha256": sha(manifest_data), "validated_without_git": True,
                "reproduced_predictions": len(actual_scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/entrega_spotify_ep3.zip")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        build_archive(ROOT, args.output)
    result = check_archive(args.output)
    receipt = args.output.parent / "validacion_zip_ep3.json"
    receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ZIP EP3 verificado: {args.output}")


if __name__ == "__main__":
    main()
