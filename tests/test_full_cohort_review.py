import unittest
import numpy as np
import pandas as pd

try:
    from sberindex.forecasting.full_cohort_review import eligible_ids, evaluation_pairs
except ImportError:
    eligible_ids = evaluation_pairs = None


class FullCohortTests(unittest.TestCase):
    def test_cohort_ignores_future_missingness(self):
        self.assertIsNotNone(eligible_ids)
        panel = pd.DataFrame(np.ones((3, 24)), index=[1, 2, 3])
        panel.loc[1, 20] = np.nan
        panel.loc[2, 5] = np.nan
        self.assertEqual(list(eligible_ids(panel)), [1, 3])

    def test_pairs_require_prefix_not_future_intermediates(self):
        self.assertIsNotNone(evaluation_pairs)
        values = np.ones(24)
        values[13] = np.nan
        pairs, reasons = evaluation_pairs(values, [1, 3, 6, 12])
        self.assertIn((11, 3), pairs)
        self.assertIn((11, 12), pairs)
        self.assertFalse(any(origin >= 13 for origin, _ in pairs))
        self.assertGreater(reasons["missing_history"], 0)
        self.assertGreater(reasons["missing_target"], 0)
