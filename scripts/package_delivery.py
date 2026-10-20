"""Empaquetar la entrega académica y comprobar su integridad."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
INTERNAL_DOCUMENTS = frozenset({
    "docs/defensa_individual.md",
    "docs/equipo_y_git.md",
    "docs/acta_verificacion.md",
})


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()


def build_archive(root, names, output):
    """Exportar archivos académicos, con hashes verificables de cada archivo."""
    files = sorted({p for p in names if p and p not in INTERNAL_DOCUMENTS})
    manifest = {
        "format_version": 1,
        "files": {name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                         "bytes": (root / name).stat().st_size} for name in files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name in files:
            z.write(root / name, "spotify-eva2/" + name)
        z.writestr("spotify-eva2/MANIFIESTO_ENTREGA.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(output) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP corrupto")
        for name, expected in manifest["files"].items():
            data = z.read("spotify-eva2/" + name)
            if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Contenido alterado: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true",
                        help="Crear una copia revisable de los archivos actuales sin crear un commit")
    args = parser.parse_args()
    if git("status", "--porcelain") and not args.snapshot:
        raise RuntimeError("Registra los cambios o usa --snapshot para revisar una copia antes del commit.")
    command = ["git", "ls-files", "--cached"]
    if args.snapshot:
        command.extend(["--others", "--exclude-standard"])
    names = subprocess.check_output([*command, "-z"], cwd=ROOT).decode("utf-8").split("\0")
    output = ROOT / "output/entrega_spotify_eva2.zip"
    manifest = build_archive(ROOT, names, output)
    (output.parent / "MANIFIESTO_ENTREGA.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ZIP verificado: {output}")
    print(f"Archivos: {len(manifest['files'])}, tamaño: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
