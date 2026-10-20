"""Empaquetar archivos versionados y comprobar la integridad del ZIP de entrega."""
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()


def main():
    if git("status", "--porcelain"):
        raise RuntimeError("Termina de registrar los cambios antes de empaquetar la entrega.")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    files = [p for p in tracked if p]
    manifest = {
        "branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
        "history": git("log", "--reverse", "--format=%h %an <%ae> %s").splitlines(),
        "files": {name: {"sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                         "bytes": (ROOT / name).stat().st_size} for name in files},
    }
    dest = ROOT / "output"
    dest.mkdir(exist_ok=True)
    output = dest / "entrega_spotify_eva2.zip"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name in files:
            z.write(ROOT / name, "spotify-eva2/" + name)
        z.writestr("spotify-eva2/MANIFIESTO_ENTREGA.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(output) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP corrupto")
        for name, expected in manifest["files"].items():
            data = z.read("spotify-eva2/" + name)
            if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Contenido alterado: {name}")
    (dest / "MANIFIESTO_ENTREGA.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ZIP verificado: {output}")
    print(f"Archivos: {len(files)}, tamaño: {output.stat().st_size:,} bytes")
    print(f"Commit: {manifest['commit']}")


if __name__ == "__main__":
    main()
