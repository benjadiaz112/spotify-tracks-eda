"""Comprobar versiones directas y salud de dependencias en un entorno aislado."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Ejecuta este control con el Python del entorno virtual, no el global")
    packages = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        name, expected = line.strip().split("==", 1)
        actual = version(name)
        if actual != expected:
            raise RuntimeError(f"{name}: instalado {actual}, requerido {expected}")
        packages[name] = actual
    health = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if health.returncode:
        raise RuntimeError(health.stdout + health.stderr)
    result = {"isolated_environment": True, "python": platform.python_version(),
              "platform": platform.platform(), "direct_packages": packages,
              "pip_check_passed": True, "pip_check_output": health.stdout.strip()}
    if not args.check_only:
        (ROOT / "reports/eva2/environment_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
