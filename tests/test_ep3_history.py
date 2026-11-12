"""Controles de no fuga de etiquetas al construir historiales para EP3."""
import unittest

import numpy as np
import pandas as pd

from src.eva3.modeling import CatalogHistory, HISTORY_KEYS, catalog_keys, crossfit_catalog_history


class EP3CatalogHistoryTests(unittest.TestCase):
    def test_exclude_entire_audio_variant_group(self):
        # Mismo artista y álbum en ambas particiones: si entra cualquier fila
        # del mismo grupo al codificar, las medias esperadas dejan de ser 0/95.
        frame = pd.DataFrame({"artists": ["Artista"] * 4, "album_name": ["Disco"] * 4,
                              "track_genre": ["pop"] * 4})
        keys = catalog_keys(frame)
        y = np.array([90.0, 100.0, 0.0, 0.0])
        encoded, inference_history = crossfit_catalog_history(keys, y, ["a", "a", "b", "b"], n_splits=2)
        for offset in range(len(HISTORY_KEYS)):
            np.testing.assert_allclose(encoded[:, 4 * offset], [0.0, 0.0, 95.0, 95.0])
            np.testing.assert_allclose(encoded[:, 4 * offset + 3], np.log1p(2))
        np.testing.assert_allclose(inference_history.transform(keys)[:, 0], y.mean())

    def test_unknown_artist_and_album_use_only_training_prior(self):
        train = pd.DataFrame({"artists": ["Ártista", "ÁRTISTA"], "album_name": ["Disco", "Disco"],
                              "track_genre": ["pop", "pop"]})
        unseen = pd.DataFrame({"artists": ["Otro"], "album_name": ["Nuevo"], "track_genre": ["jazz"]})
        history = CatalogHistory().fit(catalog_keys(train), [20.0, 60.0])
        expected = np.tile([40.0, 40.0, 0.0, 0.0], len(HISTORY_KEYS))
        np.testing.assert_allclose(history.transform(catalog_keys(unseen))[0], expected)
        np.testing.assert_allclose(history.transform(catalog_keys(train))[0, 4:8], [40.0, 40.0, 0.0, np.log1p(2)])
