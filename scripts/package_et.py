"""Generar y verificar la entrega final de Spotify Tracks."""
import argparse
from contextlib import ExitStack
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.package_ep3 import FIXED, FOLDERS, REQUIRED, sha

PREFIX = "spotify-et"
MANIFEST = "MANIFIESTO_ENTREGA.json"
BYTE_LIMIT = 25_000_000
ET_DOCS = ("README.md", "docs/et/README.md", "docs/et/informe_tecnico.md",
           "docs/et/informe_ejecutivo.md")
ET_NOTEBOOK = "notebooks/07_examen_transversal.ipynb"
ET_DECK = "docs/et/Spotify_ET.pptx"
ET_REQUIRED = {*ET_DOCS, ET_NOTEBOOK, ET_DECK, "scripts/package_et.py"}


def delivery_members(root: Path) -> dict[str, bytes]:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("Registra los archivos de la entrega en Git antes de empaquetarlos.")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    names = {name for name in paths if name in FIXED or name in ET_REQUIRED
             or name.startswith(FOLDERS + ("docs/et/", "reports/et/", "src/et/"))}
    if missing := (REQUIRED | ET_REQUIRED).difference(names):
        raise ValueError(f"Faltan archivos de entrega versionados: {sorted(missing)}")
    files = {name: (root / name).read_bytes() for name in sorted(names)}
    manifest = {
        "format_version": 1,
        "evaluation": "EFT",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
        "files": {name: {"sha256": sha(content), "bytes": len(content)} for name, content in files.items()},
    }
    files[MANIFEST] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    return files


def write_zip(dest: Path, names, files: dict[str, bytes]) -> None:
    with ZipFile(dest, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(names):
            archive.writestr(f"{PREFIX}/{name}", files[name])


def split_for_upload(full: Path, files: dict[str, bytes]) -> tuple[Path, ...]:
    if full.stat().st_size <= BYTE_LIMIT:
        return ()
    names = [[], []]
    sizes = [0, 0]
    with ZipFile(full) as archive:
        for member in sorted(archive.infolist(), key=lambda item: item.compress_size, reverse=True):
            part = 0 if sizes[0] <= sizes[1] else 1
            names[part].append(member.filename.removeprefix(f"{PREFIX}/"))
            sizes[part] += member.compress_size
    parts = tuple(full.with_name(f"{full.stem}_parte_{number}.zip") for number in (1, 2))
    for path, members in zip(parts, names, strict=True):
        write_zip(path, members, files)
        if path.stat().st_size > BYTE_LIMIT:
            raise ValueError(f"{path.name} supera {BYTE_LIMIT} bytes.")
    return parts


def extract_verified(paths: tuple[Path, ...], project: Path) -> dict:
    """Validar el conjunto de miembros y sus bytes antes de ejecutar código."""
    prefix = f"{PREFIX}/"
    with ExitStack() as stack:
        members = {}
        for path in paths:
            archive = stack.enter_context(ZipFile(path))
            for name in archive.namelist():
                if not name.startswith(prefix):
                    raise ValueError(f"Miembro fuera de la carpeta esperada: {name}")
                key = name.removeprefix(prefix)
                if key in members:
                    raise ValueError(f"Miembro duplicado: {name}")
                members[key] = archive

        manifest_data = members[MANIFEST].read(prefix + MANIFEST)
        manifest = json.loads(manifest_data)
        files = manifest["files"]
        if (manifest["evaluation"] != "EFT" or set(members) != set(files) | {MANIFEST}
                or not ET_REQUIRED.issubset(files)):
            raise ValueError("El ZIP no corresponde al manifiesto de la entrega EFT.")
        resolved_paths = {project / MANIFEST}
        for name, spec in files.items():
            target = (project / name).resolve()
            if not target.is_relative_to(project) or target in resolved_paths:
                raise ValueError(f"Ruta de extracción inválida: {name}")
            resolved_paths.add(target)
            content = members[name].read(prefix + name)
            if len(content) != spec["bytes"] or sha(content) != spec["sha256"]:
                raise ValueError(f"Archivo dañado: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (project / MANIFEST).write_bytes(manifest_data)
    return manifest


def verify_extracted(project: Path) -> int:
    import nbformat

    for name in ET_DOCS:
        path = project / name
        links = re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
        for link in links:
            if not link.startswith(("https://", "http://", "#")) and not (
                    path.parent / link.split("#", 1)[0]).exists():
                raise ValueError(f"{name}: enlace roto a {link}")

    notebook = nbformat.read(project / ET_NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code or any(cell.execution_count is None or any(
            output.output_type == "error" for output in cell.outputs) for cell in code):
        raise ValueError("Se esperaba un notebook con todas sus celdas ejecutadas sin errores.")
    with ZipFile(project / ET_DECK) as deck:
        if deck.testzip() is not None or not any(
                re.fullmatch(r"ppt/slides/slide\d+\.xml", name) for name in deck.namelist()):
            raise ValueError("El PowerPoint de la entrega está dañado o no contiene diapositivas.")

    env = dict(os.environ, PYTHONPATH="", GIT_CEILING_DIRECTORIES=str(project.parent),
               MPLCONFIGDIR=str(project / "tmp/matplotlib"), PYTHONUTF8="1")
    paths = env.get("PATH", "").split(os.pathsep)
    while executable := shutil.which("git", path=os.pathsep.join(paths)):
        parent = Path(executable).resolve().parent
        filtered = [item for item in paths if Path(item or ".").resolve() != parent]
        if filtered == paths:
            raise RuntimeError("No se pudo aislar Git para verificar la entrega.")
        paths = filtered
    env["PATH"] = os.pathsep.join(paths)
    for args in (
        ["scripts/validate_environment.py", "--check-only"],
        ["-m", "unittest", "discover", "-s", "tests", "-q"],
        ["scripts/verify_ep3.py", "--check-only"],
        ["-m", "src.eva3.predict", "reports/eva3/prediction_input_example.csv"],
    ):
        subprocess.run([sys.executable, *args], cwd=project, env=env, check=True, timeout=240)
    with (project / "reports/eva3/prediction_example.csv").open(encoding="utf-8", newline="") as file:
        expected = list(csv.DictReader(file))
    with (project / "output/predictions_ep3.csv").open(encoding="utf-8", newline="") as file:
        actual = list(csv.DictReader(file))
    if not expected or len(expected) != len(actual) or any(
            a["track_id"] != b["track_id"] or not math.isclose(
                float(a["estimated_popularity"]), float(b["estimated_popularity"]), rel_tol=0, abs_tol=1e-8
            ) for a, b in zip(expected, actual, strict=True)):
        raise ValueError("Las predicciones del ZIP difieren del ejemplo registrado.")
    return len(actual)


def package(root: Path, output: Path) -> dict:
    files = delivery_members(root)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="et-", dir=output) as temp:
        staging = Path(temp).resolve()
        full = staging / "entrega_spotify_et.zip"
        write_zip(full, files, files)
        parts = split_for_upload(full, files)
        manifest = extract_verified((full,), staging / "entrega_completa")
        if parts and extract_verified(parts, staging / "entrega_dividida") != manifest:
            raise ValueError("Los ZIP separados no coinciden con la entrega completa.")
        count = verify_extracted(staging / ("entrega_dividida" if parts else "entrega_completa"))
        archives = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
            for path in (full, *parts)
        }
        receipt = {"passed": True, "commit": manifest["commit"], "files": len(manifest["files"]),
                   "archives": archives, "verified_without_git": True, "reproduced_predictions": count}
        record = staging / "validacion_zip_et.json"
        record.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in (full, *parts, record):
            path.replace(output / path.name)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    result = package(ROOT, args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
