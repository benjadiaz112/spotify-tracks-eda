"""Auditar la utilidad editorial de K-Means con predicciones de reserva."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from src.eva3.modeling import MODEL_HGB


def audit_clusters(data: dict, root: Path, *, out: Path | None = None) -> pd.DataFrame:
    """Comparar el K publicado y el codo sin ajustar con datos de reserva."""
    evidence = root / "reports/eva3"
    protocol = json.loads((evidence / "clustering.json").read_text(encoding="utf-8"))
    selected_k = protocol["selected_k"]
    elbow_k = protocol["elbow_chord_k"]
    model_archive = joblib.load(root / "models/eva3/music_segments.joblib")
    pipeline = model_archive["pipeline"]
    scaler = pipeline.named_steps["scale"]
    selected = pipeline.named_steps["kmeans"]
    if selected.n_clusters != selected_k or elbow_k == selected_k:
        raise ValueError("El modelo guardado y el diagnóstico K-Means son incompatibles.")
    features = model_archive["features"]
    train = data["clean"].iloc[data["train_idx"]]
    reserve = data["clean"].iloc[data["test_idx"]]
    predictions = pd.read_csv(evidence / "test_predictions.csv", usecols=[
        "track_id", "track_genre", "popularity", MODEL_HGB,
    ])
    reserve = reserve.merge(predictions, how="outer", on="track_id", validate="one_to_one",
                            indicator=True, suffixes=("", "_recorded"))
    if (not reserve["_merge"].eq("both").all()
            or not reserve["popularity"].eq(reserve["popularity_recorded"]).all()
            or not reserve["track_genre"].eq(reserve["track_genre_recorded"]).all()):
        raise ValueError("La reserva y las predicciones guardadas no contienen las mismas canciones y etiquetas.")
    # Recuperar el orden original de la reserva para reproducir la muestra de silueta.
    reserve = reserve.set_index("track_id").loc[data["clean"].iloc[data["test_idx"]].track_id]
    reserve["abs_error"] = reserve[MODEL_HGB].sub(reserve["popularity"]).abs()
    scaled_train = scaler.transform(train[features])
    scaled_reserve = scaler.transform(reserve[features])
    diagnostics = pd.read_csv(evidence / "clustering_diagnostics.csv").set_index("k")
    stored_labels = pd.read_csv(evidence / "pca_test_coordinates.csv").set_index("track_id")["cluster"]
    sample_size = min(2500, len(reserve))
    sample_train = np.sort(np.random.default_rng(protocol["seed"]).choice(
        len(train), size=min(2500, len(train)), replace=False,
    ))
    tables = []
    with threadpool_limits(limits=4):
        elbow = clone(selected).set_params(n_clusters=elbow_k).fit(scaled_train)
        for model, profile_file in (
            (selected, "cluster_profiles_test.csv"), (elbow, "cluster_profiles_elbow_test.csv")
        ):
            k = model.n_clusters
            labels = model.predict(scaled_reserve)
            if k == selected_k:
                if set(stored_labels.index) != set(reserve.index):
                    raise ValueError("Los ID de segmentación difieren de la reserva.")
                np.testing.assert_array_equal(labels, stored_labels.loc[reserve.index].to_numpy())
            train_silhouette = silhouette_score(
                scaled_train[sample_train], model.predict(scaled_train[sample_train]),
            )
            reserve_silhouette = silhouette_score(
                scaled_reserve, labels, sample_size=sample_size, random_state=protocol["seed"],
            )
            np.testing.assert_allclose(train_silhouette, diagnostics.loc[k, "silhouette"], rtol=0, atol=1e-8)
            if k == selected_k:
                np.testing.assert_allclose(
                    reserve_silhouette, protocol["silhouette_test_sample"], rtol=0, atol=1e-8,
                )
            profiles = reserve.assign(cluster=labels).groupby("cluster", sort=True)
            table = profiles.agg(
                n=("popularity", "size"),
                popularity_mean=("popularity", "mean"),
                popularity_p75=("popularity", lambda values: values.quantile(.75)),
                mae_hgb=("abs_error", "mean"),
                energy_mean=("energy", "mean"),
                acousticness_mean=("acousticness", "mean"),
                liveness_mean=("liveness", "mean"),
            )
            genres = profiles["track_genre"].value_counts().unstack(fill_value=0).sort_index(axis=1)
            table["top_genre"] = genres.idxmax(axis=1)
            table["top_genre_n"] = genres.max(axis=1)
            table["top_genre_pct"] = 100 * table["top_genre_n"] / table["n"]
            table.insert(1, "pct", 100 * table["n"] / len(reserve))
            table.insert(0, "k", k)
            table["silhouette_train"] = train_silhouette
            table["silhouette_reserve"] = reserve_silhouette
            published = pd.read_csv(evidence / profile_file).set_index("cluster")
            np.testing.assert_array_equal(table.index, published.index)
            np.testing.assert_array_equal(table["n"], published["n"])
            np.testing.assert_allclose(table["popularity_mean"], published["popularity"], rtol=0, atol=1e-8)
            np.testing.assert_allclose(
                np.average(table["mae_hgb"], weights=table["n"]), reserve["abs_error"].mean(),
                rtol=0, atol=1e-8,
            )
            tables.append(table.reset_index())

    result = pd.concat(tables, ignore_index=True)
    destination = out if out is not None else root / "reports/et"
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "cluster_business.csv", index=False, lineterminator="\n")
    return result
