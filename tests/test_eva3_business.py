import unittest

import pandas as pd

from src.eva3.business import grouped_bootstrap, simulate_shortlist


class BusinessEvidenceTests(unittest.TestCase):
    def test_shortlist_uses_fixed_budget_and_stable_tiebreak(self):
        rows = pd.DataFrame({
            "track_id": ["d", "c", "a", "b"],
            "prediction": [80, 10, 80, 80],
            "popularity": [90, 20, 70, 30],
            "track_genre": ["rock", "folk", "folk", "pop"],
        })
        result = simulate_shortlist(rows, "prediction", budget=2, threshold=60)
        self.assertEqual(result["songs"], 2)
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["precision_at_budget"], 0.5)
        self.assertEqual(result["recall_at_budget"], 0.5)
        self.assertEqual(result["mean_popularity"], 50)

    def test_paired_bootstrap_keeps_constant_difference_across_groups(self):
        rows = pd.DataFrame({
            "audio_group": ["A", "A", "B"], "popularity": [20, 20, 20],
            "prediction": [22, 25, 30], "baseline_median": [26, 29, 34],
        })
        result = grouped_bootstrap(rows, n_boot=40)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["intervals"]["mae_gain_baseline"], {"point": 4.0, "lo_95": 4.0, "hi_95": 4.0})
        self.assertEqual(result["intervals"]["within_10"]["point"], 1.0)


if __name__ == "__main__":
    unittest.main()
