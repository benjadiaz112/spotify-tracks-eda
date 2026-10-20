"""Verificar procedencia tanto desde Git como desde la entrega descomprimida."""
import hashlib
import json
from pathlib import Path
import subprocess


def source_integrity(root: Path, expected_data_hash: str) -> dict:
    """Preferir el manifiesto del ZIP, incluso dentro de otro repositorio.

    Los resultados y notebooks pueden regenerarse. Se valida la instantánea de
    código, dependencias e input, no la igualdad binaria de salidas regeneradas.
    El manifiesto detecta alteraciones accidentales; no es una firma digital.
    """
    root = root.resolve()
    data_name = "data/raw/Spotify_Tracks_Dataset.csv"
    actual = hashlib.sha256((root / data_name).read_bytes()).hexdigest()
    if actual != expected_data_hash:
        raise ValueError("El CSV no coincide con el archivo original validado")
    manifest_path = root / "MANIFIESTO_ENTREGA.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        if files.get(data_name, {}).get("sha256") != expected_data_hash:
            raise ValueError("El manifiesto no identifica el CSV original")
        checked = []
        for name, spec in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Ruta del manifiesto fuera de la entrega")
            if name == data_name or name == "requirements.txt" or name.startswith(("src/", "scripts/", "tests/")):
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec.get("sha256"):
                    raise ValueError(f"Archivo de origen alterado: {name}")
                checked.append(name)
        if not any(name.startswith("src/") for name in checked):
            raise ValueError("El manifiesto no incluye código de origen")
        return {"mode": "archive_manifest", "commit": manifest.get("commit"),
                "source_files_checked": len(checked), "data_hash": actual}
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root,
                                      stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if Path(top).resolve() != root:
            raise RuntimeError("El repositorio encontrado pertenece a un directorio superior")
        original = subprocess.check_output(["git", "show", f"HEAD:{data_name}"], cwd=root,
                                           stderr=subprocess.DEVNULL)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Falta MANIFIESTO_ENTREGA.json y esta carpeta no contiene el repositorio de origen") from error
    if hashlib.sha256(original).hexdigest() != expected_data_hash:
        raise ValueError("Git no conserva los bytes originales del CSV")
    return {"mode": "git_checkout", "commit": commit, "source_files_checked": 1, "data_hash": actual}
