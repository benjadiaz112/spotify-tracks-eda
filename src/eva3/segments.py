"""Segmentos de audio, estabilidad y lectura de negocio en la nueva partición."""
from pathlib import Path

import pandas as pd

from src.eva2.clustering import fit_clustering
from src.eva2.data import save_json


def fit_segments(data: dict, root: Path, out: Path, model_dir: Path) -> dict:
    """Reutilizar el protocolo de EP2 sin tocar sus modelos ni resultados."""
    result = fit_clustering(data, root, out=out, model_dir=model_dir)
    train = pd.read_csv(out / "cluster_profiles_train.csv")
    test = pd.read_csv(out / "cluster_profiles_test.csv")
    table = train[["cluster", "pct", "energy", "acousticness", "tempo"]].merge(
        test[["cluster", "pct", "energy", "acousticness", "tempo"]],
        on="cluster", suffixes=("_train", "_test"), validate="one_to_one",
    )
    median_energy = data["clean"].iloc[data["train_idx"]].energy.median()
    median_acoustic = data["clean"].iloc[data["train_idx"]].acousticness.median()
    labels = []
    for row in table.itertuples(index=False):
        if row.energy_train > median_energy and row.acousticness_train < median_acoustic:
            label = "Priorizar revisión para listas activas"
        elif row.acousticness_train > median_acoustic and row.energy_train < median_energy:
            label = "Explorar listas de escucha tranquila"
        else:
            label = "Revisar manualmente el perfil de audio"
        labels.append(label)
    table["uso_propuesto"] = labels
    table["delta_participacion_pp"] = table.pct_test - table.pct_train
    table.to_csv(out / "segment_business_profiles.csv", index=False)
    save_json(out / "segment_protocol.json", {
        "features": "diez atributos de audio; no usa popularidad, género ni artista",
        "selection": "K=1..10, silueta y tamaño mínimo en desarrollo; evaluación separada en reserva",
        "business_usage": "exploración editorial; no son segmentos de usuarios ni estimaciones causales",
        "selected_k": result["k"],
        "largest_test_shift_pp": float(table.delta_participacion_pp.abs().max()),
    })
    return result
