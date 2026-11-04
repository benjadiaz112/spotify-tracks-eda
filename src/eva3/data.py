"""Procedencia, calidad y partición de EP3 sin ajustar modelos."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.clean_data import REQUIRED_TEXT_COLUMNS, clean_spotify_data, validate_clean_data
from src.eva2.data import EXPECTED_SHA256, audio_groups, save_json
from src.features.prepare_features import CONTINUOUS_FEATURES, MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
RAW_RELATIVE_PATH = Path("data/raw/Spotify_Tracks_Dataset.csv")
ARCHIVE_MEMBER = "Spotify_Dataset/Spotify_Tracks_Dataset.csv"
ARCHIVE_MEMBER_SHA256 = "b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd"
PROVIDED_ZIP_SHA256 = "40cd5ca85461ed70d00810bfbbe2395e37109d533ea683c03a064927bdce82e6"
SEED = 3103
TEST_SIZE = 0.20
TARGET_BANDS = ["0-20", "21-40", "41-60", "61-80", "81-100"]

_PLACEHOLDERS = {"", "-", "...", "n/a", "none", "null", "nan", "unknown", "desconocido"}
_AMBIGUOUS_ARTISTS = _PLACEHOLDERS | {"various", "various artists", "unknown artist", "artista desconocido"}
_AMBIGUOUS_TITLES = _PLACEHOLDERS | {
    "unknown title", "no title", "sin título", "sin titulo", "untitled", "untitled track",
    "intro", "interlude", "outro", "skit", "bonus track",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    # El manifiesto tiene los mismos bytes al regenerarlo en Windows o Unix.
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def source_provenance(source: Path) -> dict:
    """Comprueba las dos copias verificadas; no modifica ni extrae el ZIP recibido."""
    content = source.read_bytes()
    digest = _sha256(content)
    normalized = _sha256(content.replace(b"\r\n", b"\n"))
    if digest not in {EXPECTED_SHA256, ARCHIVE_MEMBER_SHA256} or normalized != ARCHIVE_MEMBER_SHA256:
        raise ValueError(f"CSV distinto de las copias verificadas de Spotify: SHA-256={digest}")
    return {
        "raw_sha256": digest,
        "normalized_sha256": normalized,
        "raw_bytes": len(content),
        "raw_crlf_count": content.count(b"\r\n"),
        "raw_lf_count": content.count(b"\n") - content.count(b"\r\n"),
        "archive_member": ARCHIVE_MEMBER,
        "archive_member_sha256": ARCHIVE_MEMBER_SHA256,
        "provided_zip_sha256": PROVIDED_ZIP_SHA256,
        "matches_received_csv_after_newline_normalization": True,
        "exact_bytes_match_received_csv": digest == ARCHIVE_MEMBER_SHA256,
        "provenance_note": (
            "La referencia del ZIP se verificó al recibirlo. Para reproducir EP3 se valida el CSV local, "
            "que tiene el mismo contenido tras normalizar CRLF a LF. La corrida no requiere el ZIP."
        ),
    }


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", value).lower().split())


def primary_artist(value: object) -> str:
    """Primer crédito de la lista separada por ';'; no suplanta un primer crédito ausente."""
    if not isinstance(value, str):
        return ""
    normalized = _normalized_text(value.partition(";")[0])
    return "" if normalized in _AMBIGUOUS_ARTISTS else normalized


def recording_keys(clean: pd.DataFrame) -> pd.Series:
    """Clave conservadora: solo mayúsculas, espacios y equivalencia Unicode NFC.

    Se conservan tildes, puntuación y los sufijos de versión. Las claves ausentes
    y los títulos genéricos no enlazan filas. Esta señal agrupa para evaluar;
    no prueba identidad musical y no se usa como predictor.
    """
    def key(artists: object, title: object) -> str:
        artist = primary_artist(artists)
        name = _normalized_text(title)
        if not artist or name in _AMBIGUOUS_TITLES or re.fullmatch(r"(?:track|pista)\s*\d+", name):
            return ""
        return _sha256(json.dumps((artist, name), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    return pd.Series(
        (key(artists, title) for artists, title in zip(clean["artists"], clean["track_name"])),
        index=clean.index, name="recording_key", dtype="str",
    )


def union_groups(track_ids: pd.Series, exact_audio: pd.Series, text_keys: pd.Series) -> pd.Series:
    """Cierra transitivamente ambos enlaces y nombra cada componente por su menor ID."""
    if not (track_ids.index.equals(exact_audio.index) and track_ids.index.equals(text_keys.index)):
        raise ValueError("Las claves de agrupación no están alineadas")
    if track_ids.isna().any() or track_ids.duplicated().any() or exact_audio.isna().any():
        raise ValueError("Cada fila debe tener un ID único y una huella de audio")
    parents = list(range(len(track_ids)))
    sizes = [1] * len(track_ids)

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a == b:
            return
        if sizes[a] < sizes[b]:
            a, b = b, a
        parents[b] = a
        sizes[a] += sizes[b]

    for series in (exact_audio, text_keys):
        first: dict[str, int] = {}
        for i, value in enumerate(series):
            if not isinstance(value, str) or not value:
                continue
            union(i, first.setdefault(value, i))

    roots = [find(i) for i in range(len(track_ids))]
    minimum_id: dict[int, str] = {}
    for root, track_id in zip(roots, track_ids):
        minimum_id[root] = min(track_id, minimum_id.get(root, track_id))
    return pd.Series(("g_" + minimum_id[root] for root in roots), index=track_ids.index, name="group_id", dtype="str")


def grouped_split(clean: pd.DataFrame, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Separa 80/20 por componente, estratificando por mediana del objetivo en cada grupo."""
    if not groups.index.equals(clean.index) or groups.isna().any():
        raise ValueError("Cada fila necesita su grupo antes de particionar")
    group_target = clean.groupby(groups, sort=True)["popularity"].median()
    if len(group_target) < 2:
        raise ValueError("Se requieren al menos dos grupos para separar entrenamiento y prueba")
    n_bins = min(10, len(group_target) // 5)
    bins = pd.qcut(group_target.rank(method="first"), q=n_bins, labels=False) if n_bins >= 2 else None
    train_groups, test_groups = train_test_split(
        group_target.index.to_numpy(), test_size=TEST_SIZE, random_state=SEED,
        stratify=None if bins is None else bins.to_numpy(),
    )
    train_idx = np.flatnonzero(groups.isin(train_groups).to_numpy())
    test_idx = np.flatnonzero(groups.isin(test_groups).to_numpy())
    if len(train_idx) + len(test_idx) != len(clean) or set(train_groups) & set(test_groups):
        raise ValueError("Partición incompleta o con fuga de grupos")
    return train_idx, test_idx


def _shared(train: pd.DataFrame, test: pd.DataFrame, key: str) -> int:
    return len((set(train[key]) & set(test[key])) - {""})


def _target_conflicts(raw: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    by_id = raw.groupby("track_id", sort=True)["popularity"]
    table = by_id.agg(first_raw="first", median_raw="median", min_raw="min", max_raw="max",
                     observations="size", distinct_targets="nunique")
    table = table.loc[table.distinct_targets.gt(1)].copy()
    table["range"] = table.max_raw - table.min_raw
    table = table.join(manifest.set_index("track_id")[["popularity", "split"]].rename(columns={"popularity": "target_retained"}))
    counts = {
        "policy": "first_observed_valid_row_by_track_id_ep2",
        "conflicting_ids_raw": len(table),
        "conflicting_ids_pct": float(100 * len(table) / by_id.ngroups),
        "conflicting_ids_retained": int(table.target_retained.notna().sum()),
        "conflicting_ids_train": int(table["split"].eq("train").sum()),
        "conflicting_ids_test": int(table["split"].eq("test").sum()),
        "median_range_points": float(table["range"].median()) if len(table) else 0.0,
        "maximum_range_points": int(table["range"].max()) if len(table) else 0,
    }
    return table.reset_index(), counts


def _legacy_overlap(root: Path, manifest: pd.DataFrame, out: Path) -> dict:
    path = root / "reports/eva2/split_manifest.csv"
    if not path.is_file():
        return {"available": False, "note": "No se encontró el manifiesto histórico EP2"}
    legacy = pd.read_csv(path, usecols=["track_id", "split"], dtype={"track_id": "str", "split": "str"})
    if legacy.track_id.duplicated().any() or set(legacy["split"]) != {"train", "test"}:
        raise ValueError("El manifiesto EP2 no tiene IDs únicos y dos particiones válidas")
    joined = manifest.merge(legacy.rename(columns={"split": "ep2_split"}), how="left", on="track_id", validate="one_to_one")
    joined["ep2_split"] = joined["ep2_split"].fillna("not_in_ep2")
    matrix = joined.groupby(["split", "ep2_split"], sort=True).size().rename("rows").reset_index()
    _write_csv(matrix.rename(columns={"split": "ep3_split"}), out / "ep2_overlap.csv")
    test_ep3 = joined["split"].eq("test")
    in_ep2_test = joined["ep2_split"].eq("test")
    in_ep2_train = joined["ep2_split"].eq("train")
    same_test = int((test_ep3 & in_ep2_test).sum())
    ep2_train_keys = set(joined.loc[in_ep2_train, "recording_key"]) - {""}
    ep2_test_keys = set(joined.loc[in_ep2_test, "recording_key"]) - {""}
    return {
        "available": True,
        "ep2_test_rows": int(legacy["split"].eq("test").sum()),
        "ep3_test_rows_in_ep2_test": same_test,
        "ep3_test_pct_in_ep2_test": float(100 * same_test / test_ep3.sum()),
        "ep3_test_rows_in_ep2_train": int((test_ep3 & in_ep2_train).sum()),
        "ep3_test_rows_not_in_ep2": int((test_ep3 & ~(in_ep2_train | in_ep2_test)).sum()),
        "ep2_shared_recording_keys_under_ep3_rule": len(ep2_train_keys & ep2_test_keys),
        "note": "Prueba EP3 reservada para esta corrida; misma fuente que EP2. No es validación externa ni temporal.",
    }


def prepare(root: Path = ROOT, out: Path | None = None) -> dict:
    """Materializa las etapas de datos EP3 y entrega entradas alineadas al modelamiento."""
    root = Path(root)
    out = root / "reports/eva3" if out is None else Path(out)
    source = root / RAW_RELATIVE_PATH
    provenance = source_provenance(source)
    raw = pd.read_csv(source)
    required = set(MODEL_FEATURES + REQUIRED_TEXT_COLUMNS + ["track_id", "popularity"])
    if missing := required - set(raw.columns):
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if raw.track_id.isna().any() or raw.popularity.isna().any():
        raise ValueError("IDs o etiquetas originales ausentes")

    clean, steps = clean_spotify_data(raw)
    validate_clean_data(clean)
    if not isinstance(clean.index, pd.RangeIndex) or not clean.index.equals(pd.RangeIndex(len(clean))):
        raise ValueError("La limpieza debe devolver un índice posicional continuo")
    if clean[MODEL_FEATURES + ["popularity"]].isna().any().any():
        raise ValueError("Predictores o etiquetas limpios ausentes")
    if not np.isfinite(clean[CONTINUOUS_FEATURES + ["popularity"]]).all().all():
        raise ValueError("Valores numéricos no finitos")
    if not clean.key.between(0, 11).all() or not clean["mode"].isin([0, 1]).all():
        raise ValueError("Tonalidad o modo fuera del dominio")
    if not clean.explicit.isin([False, True]).all() or not clean.popularity.between(0, 100).all():
        raise ValueError("Indicador explícito o popularidad fuera de su dominio")
    clean["explicit"] = clean["explicit"].astype("int8")

    exact_audio = audio_groups(clean)
    text_keys = recording_keys(clean)
    groups = union_groups(clean.track_id, exact_audio, text_keys)
    train_idx, test_idx = grouped_split(clean, groups)
    manifest = clean[["track_id", "popularity"]].copy()
    manifest["group_id"] = groups
    manifest["audio_group"] = exact_audio
    manifest["recording_key"] = text_keys
    manifest["split"] = "train"
    manifest.loc[test_idx, "split"] = "test"
    train, test = manifest.iloc[train_idx], manifest.iloc[test_idx]
    shares = {
        "shared_tracks": _shared(train, test, "track_id"),
        "shared_groups": _shared(train, test, "group_id"),
        "shared_audio_groups": _shared(train, test, "audio_group"),
        "shared_recording_keys": _shared(train, test, "recording_key"),
    }
    if any(shares.values()):
        raise ValueError(f"La partición EP3 comparte identificadores: {shares}")

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(steps, out / "cleaning.csv")
    schema = pd.DataFrame({
        "column": raw.columns, "raw_dtype": raw.dtypes.astype(str).to_numpy(),
        "raw_missing": raw.isna().sum().to_numpy(), "raw_unique_nonmissing": raw.nunique().to_numpy(),
        "clean_dtype": [str(clean[col].dtype) if col in clean else "removed" for col in raw.columns],
        "role": ["target" if col == "popularity" else "feature" if col in MODEL_FEATURES else "audit_only" for col in raw.columns],
    })
    _write_csv(schema, out / "schema.csv")
    _write_csv(clean[CONTINUOUS_FEATURES + ["popularity"]].describe().rename_axis("statistic").reset_index(), out / "descriptive.csv")
    _write_csv(manifest, out / "split_manifest.csv")
    band_table = manifest.assign(target_band=pd.cut(manifest.popularity, [-1, 20, 40, 60, 80, 100], labels=TARGET_BANDS))
    band_table = band_table.groupby(["split", "target_band"], observed=False).size().rename("rows").reset_index()
    band_table["share_pct"] = 100 * band_table.rows / band_table.groupby("split").rows.transform("sum")
    _write_csv(band_table, out / "target_bands.csv")
    genre_table = manifest.assign(track_genre=clean.track_genre).groupby(["split", "track_genre"]).size().rename("rows").reset_index()
    _write_csv(genre_table, out / "genre_counts.csv")

    target_table, target = _target_conflicts(raw, manifest)
    _write_csv(target_table, out / "target_conflicts.csv")
    save_json(out / "target_audit.json", target)
    legacy = _legacy_overlap(root, manifest, out)
    component_sizes = groups.value_counts()
    train_artists = set(clean.iloc[train_idx].artists.map(primary_artist)) - {""}
    test_artists = clean.iloc[test_idx].artists.map(primary_artist)
    group_audio_counts = manifest.groupby("group_id").audio_group.nunique()
    audit = {
        **provenance,
        "source_path": RAW_RELATIVE_PATH.as_posix(),
        "raw_rows": len(raw), "raw_columns": len(raw.columns),
        "raw_unique_tracks": int(raw.track_id.nunique()), "raw_missing_cells": int(raw.isna().sum().sum()),
        "clean_rows": len(clean), "clean_columns": len(clean.columns),
        "clean_missing_cells": int(clean.isna().sum().sum()), "clean_genres": int(clean.track_genre.nunique()),
        "raw_retention_pct": 100 * len(clean) / len(raw),
        "unique_track_retention_pct": 100 * len(clean) / raw.track_id.nunique(),
        "split_seed": SEED, "test_group_fraction_requested": TEST_SIZE,
        "split_rule": "componentes de audio exacto y artista principal/título NFC; 80/20 estratificado por mediana de popularidad de grupo",
        "target_used_in_group_key": False,
        "target_used_only_for_split_stratification": True,
        "split_manifest_sha256": _sha256((out / "split_manifest.csv").read_bytes()),
        "train_rows": len(train_idx), "test_rows": len(test_idx), "test_rows_pct": 100 * len(test_idx) / len(clean),
        "train_groups": int(train.group_id.nunique()), "test_groups": int(test.group_id.nunique()),
        "train_target_mean": float(train.popularity.mean()), "test_target_mean": float(test.popularity.mean()),
        "audio_groups": int(exact_audio.nunique()), "repeated_audio_rows": int(exact_audio.duplicated().sum()),
        "recording_key_eligible_rows": int(text_keys.ne("").sum()),
        "recording_key_unusable_rows": int(text_keys.eq("").sum()),
        "recording_keys": int(text_keys[text_keys.ne("")].nunique()),
        "union_groups": len(component_sizes), "non_singleton_groups": int(component_sizes.gt(1).sum()),
        "largest_group_rows": int(component_sizes.max()),
        "union_groups_with_distinct_audio_hashes": int(group_audio_counts.gt(1).sum()),
        "union_groups_with_distinct_targets": int(manifest.groupby("group_id").popularity.nunique().gt(1).sum()),
        **shares,
        "test_primary_artist_eligible_rows": int(test_artists.ne("").sum()),
        "test_primary_artist_seen_rows": int(test_artists.isin(train_artists).sum()),
        "test_primary_artist_seen_pct_of_all_test_rows": float(100 * test_artists.isin(train_artists).mean()),
        "target_conflicts": target,
        "ep2_overlap": legacy,
        "assessment_scope": "Generalización interna a componentes de grabación no vistos de la misma fuente, con artistas posiblemente vistos; sin fecha confirmada ni prueba externa.",
        "eda_scope": "Descripción global y por partición; no se ajustan transformaciones ni se seleccionan modelos en esta etapa.",
    }
    save_json(out / "data_audit.json", audit)
    return {"raw": raw, "clean": clean, "groups": groups, "train_idx": train_idx, "test_idx": test_idx, "audit": audit}
