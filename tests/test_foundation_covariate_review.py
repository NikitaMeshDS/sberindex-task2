"""Leakage checks: future mutations must not change eligibility, profiles or inputs."""

import unittest
import numpy as np
import pandas as pd

try:
    from sberindex.forecasting import foundation_covariate_review as audit
except ImportError:
    audit = None


class FoundationCovariateReviewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, "new audit module missing")
        self.values = np.arange(2 * 6 * 24, dtype=float).reshape(2, 6, 24) + 1

    def test_future_values_cannot_enter_profile_or_origin_inputs(self):
        panels = {
            str(k): pd.DataFrame(self.values[:, k], index=[1, 2]) for k in range(6)
        }
        profiles, _ = audit.pooled_profiles(panels)
        changed = {k: v.copy() for k, v in panels.items()}
        for p in changed.values():
            p.iloc[:, 12:] = 1e20
        p2, _ = audit.pooled_profiles(changed)
        np.testing.assert_array_equal(profiles, p2)
        a = audit.make_inputs(
            self.values, 11, 12, profiles, grouped=True, covariates=True
        )
        perturbed = self.values.copy()
        perturbed[:, :, 12:] = np.nan
        b = audit.make_inputs(perturbed, 11, 12, p2, grouped=True, covariates=True)
        for original, new in zip(a, b):
            np.testing.assert_array_equal(original["target"], new["target"])
            for section in ["past_covariates", "future_covariates"]:
                self.assertEqual(set(original[section]), set(new[section]))
                for k in original[section]:
                    np.testing.assert_array_equal(original[section][k], new[section][k])

    def test_groups_keep_exactly_six_categories_and_known_future_profile(self):
        profiles = np.tile(np.arange(1, 13), (6, 1))
        grouped = audit.make_inputs(self.values, 13, 6, profiles, True, True)
        single = audit.make_inputs(self.values, 13, 6, profiles, False, True)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(single), 12)
        self.assertEqual(grouped[1]["target"].shape, (6, 14))
        np.testing.assert_array_equal(grouped[1]["target"][3], single[9]["target"])
        for item in grouped:
            self.assertEqual(len(item["future_covariates"]), 6)
            for vals in item["future_covariates"].values():
                np.testing.assert_array_equal(vals, [3, 4, 5, 6, 7, 8])

    def test_peak_rss_platform_units_produce_same_gib(self):
        convert = getattr(audit, "rss_gib", None)
        self.assertIsNotNone(convert, "portable RSS conversion missing")
        self.assertEqual(convert(1024**3, "darwin"), 1.0)
        self.assertEqual(convert(1024**2, "linux"), 1.0)
        with self.assertRaises(ValueError):
            convert(1024, "unknown")

    def test_gap_in_any_category_excludes_mo_using_history_only(self):
        future_changed = self.values.copy()
        future_changed[0, :, 20:] = np.nan
        np.testing.assert_array_equal(
            audit.active_mask(self.values, 18), audit.active_mask(future_changed, 18)
        )
        future_changed[1, 4, 15] = np.nan
        np.testing.assert_array_equal(
            audit.active_mask(future_changed, 18), [True, False]
        )


if __name__ == "__main__":
    unittest.main()
