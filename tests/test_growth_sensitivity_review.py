import unittest
import numpy as np
import pandas as pd
from sberindex.forecasting.growth_sensitivity_review import apply_growth, available_at_origin, metric_rows


class GrowthSensitivityTests(unittest.TestCase):
    def test_no_wrap_is_exactly_unchanged(self):
        base = np.array([12.123456789, 1700.5])
        np.testing.assert_array_equal(apply_growth(base, ['2024-01','2024-02'], [1,6], .172), base)

    def test_one_wrap_multiplies_once(self):
        np.testing.assert_allclose(apply_growth([100,200], ['2023-12','2023-12'], [1,12], .172), [117.2,234.4])

    def test_prediction_does_not_depend_on_target_actual(self):
        f = pd.DataFrame({'predicted':[100.], 'origin':['2023-12'], 'horizon':[12], 'actual':[20.]})
        before = apply_growth(f.predicted, f.origin, f.horizon, .074)
        f['actual'] = 1e10
        np.testing.assert_array_equal(before, apply_growth(f.predicted, f.origin, f.horizon, .074))

    def test_cpi_release_after_origin_is_not_available(self):
        self.assertFalse(available_at_origin('2023-12', {'available_from':'2024-01-13'}))
        self.assertTrue(available_at_origin('2023-12', {'available_from':'2023-12-09'}))

    def test_mae_is_date_balanced_and_counts_single_origin(self):
        f = pd.DataFrame({'horizon':[1]*3,'target':['2024-01','2024-02','2024-02'],
                          'origin':['2023-12']*3,'territory_id':[1,1,2], 'ae':[10,0,0], 'year_bridges':[1]*3})
        rows = metric_rows(f, 'fixed', .1, 'diagnostic')
        self.assertEqual(rows[0]['MAE'], 5)
        self.assertEqual(rows[1]['origins'], 1)

    def test_invalid_growth_rejected(self):
        for rate in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                apply_growth([100], ['2023-12'], [1], rate)
