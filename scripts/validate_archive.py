"""Validar el ZIP fuera del repositorio, con un entorno aislado y sin Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Además, volver a entrenar desde el CSV extraído")
    parser.add_argument("--zip", type=Path, default=ROOT / "output/entrega_spotify_eva2.zip")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Usa el Python del entorno virtual validado")
    (ROOT / "tmp").mkdir(exist_ok=True)
    commands = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="archive-check-") as folder:
        destination = Path(folder).resolve()
        if not destination.is_relative_to((ROOT / "tmp").resolve()):
            raise ValueError("Destino de validación fuera del workspace temporal")
        with ZipFile(args.zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP corrupto")
            for member in archive.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise ValueError("Ruta del ZIP fuera del destino de validación")
            archive.extractall(destination)
        project = destination / "spotify-eva2"
        if (project / ".git").exists():
            raise ValueError("El paquete no debe depender de una copia interna de .git")
        env = os.environ.copy()
        env["GIT_CEILING_DIRECTORIES"] = str(ROOT)
        env["PYTHONPATH"] = ""
        env["MPLCONFIGDIR"] = str(project / "tmp/matplotlib")
        search_paths = env.get("PATH", "").split(os.pathsep)
        while found := shutil.which("git", path=os.pathsep.join(search_paths)):
            parent = Path(found).parent.resolve()
            search_paths = [p for p in search_paths if Path(p or ".").resolve() != parent]
        env["PATH"] = os.pathsep.join(search_paths)
        if shutil.which("git", path=env["PATH"]) is not None:
            raise ValueError("Git sigue disponible en el entorno de prueba")

        def run(arguments):
            print("Validando ZIP:", " ".join(arguments), flush=True)
            result = subprocess.run([sys.executable, *arguments], cwd=project, env=env,
                                    capture_output=True, text=True, timeout=900)
            commands.append({"arguments": arguments, "exit_code": result.returncode,
                             "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-1500:]})
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            return result.stdout

        run(["scripts/validate_environment.py", "--check-only"])
        if args.retrain:
            run(["-m", "src.eva2.run"])
        run(["-m", "src.eva2.predict", "data/examples/tracks_example.csv",
             "--output", "reports/eva2/archive_predictions.csv"])
        verification = json.loads(run(["scripts/verify_delivery.py", "--check-only"]))
        manifest_hash = hashlib.sha256((project / "MANIFIESTO_ENTREGA.json").read_bytes()).hexdigest()
        if verification["source_mode"] != "archive_manifest":
            raise ValueError("La prueba no utilizó el manifiesto del archivo")
        receipt = {"passed": True, "source_manifest_sha256": manifest_hash, "git_available": False,
                   "copy_has_git_folder": False, "retrained_from_extracted_csv": args.retrain,
                   "elapsed_seconds": time.perf_counter() - start, "commands": commands,
                   "verification": verification}
    output = ROOT / "output/validacion_zip.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "retrained": args.retrain,
                      "checks": verification["checks"], "receipt": str(output)}, indent=2))


if __name__ == "__main__":
    main()
