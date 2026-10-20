"""Ejemplos sintéticos exclusivos de pruebas, no resultados de un piloto."""
import unittest

import pandas as pd

from src.eva2.pilot import COLUMNS, pilot_metrics, pilot_summary


class PilotTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            ["test_reviewer", "test_pair", "manual", 100, 20, 600, 0, 0, 10, 30],
            ["test_reviewer", "test_pair", "assisted", 100, 20, 480, 20, 15, 9, 30],
        ], columns=COLUMNS)

    def test_paired_formulas(self):
        result = pilot_metrics(self.fixture()).iloc[0]
        self.assertAlmostEqual(result.time_reduction_pct, 20)
        self.assertAlmostEqual(result.recommendation_acceptance_pct, 75)
        self.assertAlmostEqual(result.genre_coverage_delta_pp, -5)

    def test_empty_template_does_not_invent_results(self):
        with self.assertRaisesRegex(ValueError, "no hay piloto"):
            pilot_metrics(pd.DataFrame(columns=COLUMNS))

    def test_acceptance_aggregates_counts_instead_of_unweighted_percentages(self):
        first = self.fixture()
        second = self.fixture()
        second["pair_id"] = "another_test_pair"
        second.loc[1, ["recommended_tracks", "accepted_recommendations"]] = [10, 0]
        summary = pilot_summary(pd.concat([first, second], ignore_index=True))
        self.assertEqual(summary["pairs"], 2)
        self.assertAlmostEqual(summary["recommendation_acceptance_weighted_pct"], 50)

    def test_unpaired_or_inconsistent_sessions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cada par necesita"):
            pilot_metrics(self.fixture().iloc[:1])
        df = self.fixture()
        df.loc[1, "accepted_recommendations"] = 21
        with self.assertRaisesRegex(ValueError, "inconsistentes"):
            pilot_metrics(df)
