import unittest

import numpy as np
import pandas as pd

from sberindex.forecasting.forecast_uncertainty_review import (
    cluster_bootstrap,
    date_balanced_metric,
    dm_hac,
    holm,
    paired_losses,
)


class ReviewTests(unittest.TestCase):
    def test_date_balanced_recomputes_unbalanced_draw_denominators(self):
        f = pd.DataFrame(
            {
                "target": ["a", "a", "b"],
                "loss_a": [10.0, 30.0, 100.0],
                "loss_b": [5.0, 15.0, 50.0],
                "territory_id": [1, 2, 1],
            }
        )
        np.testing.assert_allclose(date_balanced_metric(f), [-30.0, 50.0])
        # Sample MO1 twice and MO2 once: dateA denominator3, dateB denominator2.
        expected_a = ((2 * 10 + 30) / 3 + 100) / 2
        expected_b = ((2 * 5 + 15) / 3 + 50) / 2
        np.testing.assert_allclose(
            date_balanced_metric(f, [2, 1, 2]),
            [expected_b - expected_a, 100 * (1 - expected_b / expected_a)],
        )
        assert not np.isclose(
            expected_b - expected_a, np.average(f.loss_b - f.loss_a, weights=[2, 1, 2])
        )

    def test_whole_mo_bootstrap_preserves_dates_and_paired_percent(self):
        f = pd.DataFrame(
            {
                "territory_id": np.repeat(range(5), 2),
                "target": ["a", "b"] * 5,
                "loss_a": np.arange(1, 11.0),
                "loss_b": np.arange(1, 11.0) * 0.5,
            }
        )
        draws, n = cluster_bootstrap(f, "territory_id", 2000, 44)
        assert n == 5 and draws.shape == (2000, 2)
        np.testing.assert_allclose(draws[:, 1], 50)
        np.testing.assert_array_equal(
            draws, cluster_bootstrap(f, "territory_id", 2000, 44)[0]
        )

    def test_regional_dependence_sensitivity_has_wider_interval(self):
        # Twenty duplicate MO signals in each region are not twenty independent shocks.
        f = pd.DataFrame(
            {
                "territory_id": range(80),
                "region": np.repeat(range(4), 20),
                "target": "a",
                "loss_a": 10.0,
                "loss_b": np.repeat([1.0, 4.0, 7.0, 9.0], 20),
            }
        )
        mo, _ = cluster_bootstrap(f, "territory_id", 2500, 9)
        region, n = cluster_bootstrap(f, "region", 2500, 9)
        assert n == 4
        assert np.ptp(np.quantile(region[:, 0], [0.025, 0.975])) > 2 * np.ptp(
            np.quantile(mo[:, 0], [0.025, 0.975])
        )

    def test_h12_dm_is_undefined_not_fabricated_temporal_interval(self):
        result = dm_hac([100.0], 12)
        assert result["T"] == 1 and result["hac_lag"] == 11
        assert result["dm_status"].startswith("undefined")
        assert np.isnan(result["p_value"]) and np.isnan(result["temporal_ci_low"])
        assert dm_hac(range(7), 6)["dm_status"] == "not_reported_minimum_dates_policy"
        assert dm_hac([1.0] * 12, 1)["dm_status"] == "undefined_degenerate_hac"

    def test_two_horizon_holm_uses_step_down_not_double_each_p(self):
        np.testing.assert_allclose(
            holm([0.017030729933666684, 0.3068374109189995]),
            [0.03406145986733337, 0.3068374109189995],
        )

    def test_holm_family_excludes_unestimable_tests(self):
        adjusted = holm([0.01, 0.03, 0.2, np.nan])
        np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.2])
        assert np.isnan(adjusted[3])

    def test_paired_losses_requires_exact_common_pairs_and_actuals(self):
        f = pd.DataFrame(
            {
                "territory_id": [1, 1],
                "origin": ["2023-12"] * 2,
                "target": ["2024-01"] * 2,
                "horizon": [1, 1],
                "model": ["a", "b"],
                "actual": [10.0, 10.0],
                "predicted": [8.0, 9.0],
            }
        )
        p = paired_losses(f, "a", "b")
        assert date_balanced_metric(p)[0] == -1
        f.loc[1, "actual"] = 11
        with self.assertRaises(ValueError):
            paired_losses(f, "a", "b")
        with self.assertRaises(ValueError):
            paired_losses(f.iloc[:1], "a", "b")


if __name__ == "__main__":
    unittest.main()
