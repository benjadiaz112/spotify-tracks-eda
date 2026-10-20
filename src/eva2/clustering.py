"""K-Means escalado, diagnóstico K=1..10, estabilidad y PCA explicativo."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import ROOT, SEED, save_json
from src.features.prepare_features import CONTINUOUS_FEATURES as AUDIO_NUM


def fit_clustering(data: dict, root: Path = ROOT) -> dict:
    out = root / "reports/eva2"
    clean, tr, te = data["clean"], data["train_idx"], data["test_idx"]
    # Popularidad y género se excluyen: no definir grupos por el resultado ni por etiquetas.
    train = clean.iloc[tr][AUDIO_NUM]
    scaler = StandardScaler().fit(train)
    Z = scaler.transform(train)
    Z_test = scaler.transform(clean.iloc[te][AUDIO_NUM])
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(len(Z), min(2500, len(Z)), replace=False))
    models, rows = {}, []
    with threadpool_limits(limits=4):
        for k in range(1, 11):
            model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=SEED).fit(Z)
            models[k] = model
            labels = model.labels_[sample_idx]
            score = float(silhouette_score(Z[sample_idx], labels)) if k > 1 else None
            rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": score,
                         "smallest_cluster_pct": 100 * np.bincount(model.labels_).min() / len(Z),
                         "iterations": int(model.n_iter_)})
            print(f"K-Means K={k}: silhouette={score}", flush=True)
        diagnostics = pd.DataFrame(rows)
        valid = diagnostics.query("k >= 2 and smallest_cluster_pct >= 3")
        if valid.empty:
            raise ValueError("Ningún K cumple el mínimo de 3% predefinido; revisar segmentación.")
        k = int(valid.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
        model = models[k]
        # Codo orientativo por máxima distancia a la cuerda normalizada.
        normalized = (diagnostics.inertia - diagnostics.inertia.min()) / (diagnostics.inertia.max() - diagnostics.inertia.min())
        chord = 1 - (diagnostics.k - 1) / 9
        elbow = int(diagnostics.loc[(chord - normalized).idxmax(), "k"])
        stability = []
        for seed in [7, 2026]:
            alternative = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed).fit(Z)
            stability.append({"seed": seed, "ari_vs_42": float(adjusted_rand_score(model.labels_, alternative.labels_))})
        pca = PCA().fit(Z)
        coords = pca.transform(Z_test)
        test_labels = model.predict(Z_test)
        test_silhouette = float(silhouette_score(Z_test, test_labels, sample_size=min(2500, len(Z_test)), random_state=SEED))
    diagnostics.to_csv(out / "clustering_diagnostics.csv", index=False)
    pd.DataFrame(stability).to_csv(out / "cluster_stability.csv", index=False)
    variance = pd.DataFrame({"component": np.arange(1, len(AUDIO_NUM) + 1),
                            "explained_variance_ratio": pca.explained_variance_ratio_,
                            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(out / "pca_variance.csv", index=False)
    pd.DataFrame(pca.components_.T, index=AUDIO_NUM,
                 columns=[f"PC{i+1}" for i in range(len(AUDIO_NUM))]).to_csv(out / "pca_loadings.csv")
    all_labels = model.predict(scaler.transform(clean[AUDIO_NUM]))
    profile = clean[AUDIO_NUM + ["popularity"]].copy()
    profile["cluster"] = all_labels
    profile["split"] = np.where(np.isin(np.arange(len(clean)), tr), "train", "test")
    # El perfil principal usa train. Test se muestra separado para contrastar.
    for split in ["train", "test"]:
        part = profile[profile.split == split]
        means = part.groupby("cluster")[AUDIO_NUM + ["popularity"]].mean()
        means.insert(0, "n", part.groupby("cluster").size())
        means.insert(1, "pct", means.n / len(part) * 100)
        means.to_csv(out / f"cluster_profiles_{split}.csv")
    # Contrastar la alternativa sugerida por el codo sin cambiar la regla de K.
    elbow_profile = clean[AUDIO_NUM + ["popularity"]].copy()
    elbow_profile["cluster"] = models[elbow].predict(scaler.transform(clean[AUDIO_NUM]))
    for split, indices in [("train", tr), ("test", te)]:
        part = elbow_profile.iloc[indices]
        means = part.groupby("cluster")[AUDIO_NUM + ["popularity"]].mean()
        means.insert(0, "n", part.groupby("cluster").size())
        means.insert(1, "pct", means.n / len(part) * 100)
        means.to_csv(out / f"cluster_profiles_elbow_{split}.csv")
    standardized_centers = pd.DataFrame(model.cluster_centers_, columns=AUDIO_NUM)
    standardized_centers.index.name = "cluster"
    standardized_centers.to_csv(out / "cluster_centers_standardized.csv")
    cluster_genres = pd.crosstab(all_labels[tr], clean.iloc[tr].track_genre)
    cluster_genres.index.name = "cluster"
    cluster_genres.to_csv(out / "cluster_genres.csv")
    pd.DataFrame({"track_id": clean.iloc[te].track_id.to_numpy(), "cluster": test_labels,
                  "PC1": coords[:, 0], "PC2": coords[:, 1]}).to_csv(out / "pca_test_coordinates.csv", index=False)
    save_json(out / "clustering.json", {
        "selected_k": k, "elbow_chord_k": elbow,
        "selection_rule": "mayor silueta en muestra fija de 2500 train entre K=2..10 con todos los grupos >=3% de train; desempate menor K",
        "silhouette_train_sample": float(diagnostics.loc[diagnostics.k == k, "silhouette"].iloc[0]),
        "silhouette_test_sample": test_silhouette,
        "features": AUDIO_NUM, "target_used_in_fit": False, "genre_used_in_fit": False,
        "pca_two_components_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "pca_components_for_80pct": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .8) + 1),
        "stability": stability, "seed": SEED,
    })
    model_dir = root / "models/eva2"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": Pipeline([("scale", scaler), ("kmeans", model)]),
                 "pca": pca, "features": AUDIO_NUM}, model_dir / "music_segments.joblib", compress=3)
    return {"k": k, "diagnostics": diagnostics, "profiles": profile}
