import unittest

import numpy as np
import pandas as pd

from sberindex.forecasting.selected_model_holdout import forecast_selected


class SelectedHoldoutTests(unittest.TestCase):
    def panel(self):
        dates = pd.period_range("2023-01", "2025-12", freq="M").astype(str)
        values = np.ones((12, 36)) * 100
        values[:, 23] = 200
        return pd.DataFrame(values, index=range(12), columns=dates)

    def test_unseen_future_cannot_change_frozen_forecasts(self):
        p = self.panel()
        regions = pd.Series(1, index=p.index)
        a = forecast_selected(p, regions, "2024-12", [1, 3, 6, 12], 0.172)
        p.iloc[:, 24:] = np.nan
        b = forecast_selected(p, regions, "2024-12", [1, 3, 6, 12], 0.172)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a.target), {"2025-01", "2025-03", "2025-06", "2025-12"})
        np.testing.assert_allclose(a.predicted, 234.4)
        self.assertNotIn("actual", a.columns)

    def test_missing_anchor_excluded_and_eligibility_uses_2023(self):
        p = self.panel()
        p.loc[0, "2024-12"] = np.nan
        p.loc[1, "2023-05"] = np.nan
        p.loc[2, "2024-04"] = np.nan
        a = forecast_selected(p, pd.Series(1, index=p.index), "2024-12", [1], 0.172)
        self.assertEqual(set(a.territory_id), set(range(2, 12)))

    def test_no_anchor_date_is_rejected(self):
        with self.assertRaises(ValueError):
            forecast_selected(
                self.panel(), pd.Series(1, index=range(12)), "2026-12", [1], 0.172
            )
