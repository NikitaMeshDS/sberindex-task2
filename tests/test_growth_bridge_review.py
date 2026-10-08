import unittest
import numpy as np
import pandas as pd
from sberindex.forecasting.growth_bridge_review import (
    year_bridges,
    growth_for_origin,
    bridge_prediction,
)


class GrowthBridgeTests(unittest.TestCase):
    def test_wrap_count_includes_equal_month_h12(self):
        self.assertEqual(year_bridges("2023-12", 1), 1)
        self.assertEqual(year_bridges("2023-10", 5), 1)
        self.assertEqual(year_bridges("2023-12", 12), 1)
        self.assertEqual(year_bridges("2023-01", 11), 0)
        self.assertEqual(year_bridges("2023-02", 24), 2)

    def test_unpublished_macro_record_rejected(self):
        record = {"available_from": "2023-12-28", "growth_pct": 17.2}
        with self.assertRaises(ValueError):
            growth_for_origin("2023-11", record)
        self.assertAlmostEqual(growth_for_origin("2023-12", record), 0.172)

    def test_recovers_constant_growth_confounded_in_raw_one_year_profile(self):
        rate = 0.172
        t = np.arange(36)
        season = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
        values = 100 * (1 + rate) ** (t / 12) * season
        profile = values[:12] / values[:12].mean()
        for origin, horizon in [(11, 1), (11, 3), (11, 6), (11, 12), (14, 6)]:
            pred = bridge_prediction(values[origin], profile, origin, horizon, rate)
            self.assertAlmostEqual(pred, values[origin + horizon], places=9)

    def test_zero_growth_preserves_original_and_invalid_rate_fails(self):
        profile = np.arange(1, 13, dtype=float)
        self.assertEqual(bridge_prediction(100, profile, 11, 1, 0), 100 / 12)
        with self.assertRaises(ValueError):
            bridge_prediction(100, profile, 11, 1, -1)


class SelectionCausalityTests(unittest.TestCase):
    def test_future_error_mutation_does_not_change_selected_model(self):
        from sberindex.forecasting.growth_bridge_review import select_forecaster

        data = pd.DataFrame(
            [
                {"horizon": h, "model": m, "target": t, "ae": error}
                for h in [1, 3, 6]
                for m, error in [
                    ("seasonal_pooled", 10),
                    ("candidate_a", 5),
                    ("candidate_b", 15),
                ]
                for t in ["2024-06", "2024-07"]
            ]
        )
        before = select_forecaster(data)
        data.loc[(data.target == "2024-07") & (data.model == "candidate_a"), "ae"] = 1e9
        self.assertEqual(before, select_forecaster(data))
