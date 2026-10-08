import unittest

import numpy as np
import pandas as pd

from sberindex.forecasting.adaptive_growth_holdout import regional_growth_rate


class AdaptiveRateTests(unittest.TestCase):
    def panel(self):
        periods = pd.period_range("2023-01", "2025-12", freq="M").astype(str)
        p = pd.DataFrame(100.0, index=range(12), columns=periods)
        p.loc[:, "2024-10"] = 120.0
        p.loc[:, "2024-11"] = 110.0
        p.loc[:, "2024-12"] = 115.0
        return p

    def test_three_month_regional_rule_ignores_future_and_outlier(self):
        p = self.panel()
        p.loc[0, "2024-12"] = 10000
        reg = pd.Series(1, index=p.index)
        a = regional_growth_rate(p, reg, "2024-12", 0.172, minimum_n=10)
        p.loc[:, "2025-01":] = np.nan
        b = regional_growth_rate(p, reg, "2024-12", 0.172, minimum_n=10)
        pd.testing.assert_frame_equal(a, b)
        self.assertAlmostEqual(a.iloc[0].growth_rate, 0.15)
        self.assertEqual(a.iloc[0].growth_source, "regional_three_month_yoy")

    def test_no_prior_year_uses_documented_fallback(self):
        p = self.panel()
        reg = pd.Series(1, index=p.index)
        a = regional_growth_rate(p, reg, "2023-12", 0.172, minimum_n=10)
        self.assertEqual(a.iloc[0].growth_rate, 0.172)
        self.assertEqual(a.iloc[0].growth_source, "documented_macro_fallback")

    def test_insufficient_peers_in_any_month_uses_fallback(self):
        p = self.panel()
        p.loc[0:3, "2024-11"] = np.nan
        a = regional_growth_rate(
            p, pd.Series(1, index=p.index), "2024-12", 0.172, minimum_n=10
        )
        self.assertEqual(a.iloc[0].growth_rate, 0.172)

    def test_missing_region_does_not_borrow_foreign_region(self):
        p = self.panel()
        r = pd.Series(1.0, index=p.index)
        r.loc[0] = np.nan
        a = regional_growth_rate(p, r, "2024-12", 0.172, minimum_n=10)
        self.assertEqual(a[a.territory_id == 0].iloc[0].growth_rate, 0.172)
