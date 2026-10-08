import unittest
import numpy as np

try:
    from sberindex.reporting.review_real_cases import seasonal_errors
except ImportError:
    seasonal_errors = None


class RealCaseTests(unittest.TestCase):
    def test_case_control_uses_only_known_origin_and_frozen_profile(self):
        self.assertIsNotNone(seasonal_errors)
        values = np.arange(1.0, 25.0)
        profile = np.arange(1.0, 13.0)
        actual, predicted = seasonal_errors(values, profile)
        changed = values.copy()
        changed[19:] = 1e12
        _, altered = seasonal_errors(changed, profile)
        np.testing.assert_array_equal(predicted[:8], altered[:8])
        self.assertEqual(predicted[0], values[11] * profile[0] / profile[11])

    def test_missing_origin_is_not_imputed(self):
        self.assertIsNotNone(seasonal_errors)
        values = np.ones(24)
        values[11] = np.nan
        _, prediction = seasonal_errors(values, np.ones(12))
        self.assertTrue(np.isnan(prediction[0]))
