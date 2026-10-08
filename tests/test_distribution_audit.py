import unittest
import numpy as np
import pandas as pd

try:
    from sberindex.forecasting import asof_distribution_audit as audit
except ImportError:
    audit = None


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, 'distribution audit is missing')

    def test_equal_date_weight_with_unequal_counts(self):
        frame = pd.DataFrame({'territory_id':[1,2,3,1], 'target':['a','a','a','b'],
                              'actual':[100]*4, 'predicted':[100,100,100,0],
                              'expense_2023':[100]*4})
        result = audit.metrics(frame)
        self.assertEqual(result['MAE_date_balanced'], 50)
        self.assertEqual(result['MAE_pooled'], 25)
        self.assertEqual(result['NMAE_2023_pct'], 50)

    def test_training_strata_ignore_future_missingness_and_levels(self):
        dates = pd.period_range('2023-01', periods=24, freq='M').astype(str)
        frame = pd.DataFrame(np.repeat(np.arange(1,9)[:,None],24,axis=1),columns=dates)
        frame.iloc[-1,13:] = np.nan
        before, cuts = audit.training_metadata(frame)
        frame.iloc[:,12:] = 1e9
        after, new_cuts = audit.training_metadata(frame)
        pd.testing.assert_frame_equal(before,after)
        np.testing.assert_array_equal(cuts,new_cuts)
        self.assertEqual(len(before),8)

    def test_paired_comparison_rejects_missing_key(self):
        base = pd.DataFrame({'territory_id':[1,2], 'origin':['2024-06']*2,
                             'target':['2024-07']*2, 'horizon':[1]*2,
                             'actual':[100,200], 'predicted':[90,180]})
        with self.assertRaises(AssertionError):
            audit.match_errors(base.iloc[:1],base)

    def test_missing_source_actual_is_rejected(self):
        validator = getattr(audit, 'validate_actuals', None)
        self.assertIsNotNone(validator, 'source completeness validator is missing')
        predictions = pd.DataFrame({'territory_id':[1,2], 'target':['2024-07']*2,
                                    'actual':[100,200]})
        source = pd.DataFrame({'territory_id':[1], 'date':['2024-07'], 'value':[100]})
        with self.assertRaises(AssertionError):
            validator(predictions,source)

    def test_paired_wins_preserve_ties_and_direction(self):
        base = pd.DataFrame({'territory_id':[1,2,3], 'origin':['2024-06']*3,
                             'target':['2024-07']*3, 'horizon':[1]*3,
                             'actual':[100]*3, 'predicted':[90]*3})
        candidate=base.copy();candidate['predicted']=[95,85,90]
        paired = audit.paired_metrics(audit.match_errors(candidate,base))
        self.assertEqual(paired['municipalities_better'],1)
        self.assertEqual(paired['municipalities_worse'],1)
        self.assertEqual(paired['municipalities_tied'],1)
        self.assertEqual(paired['MAE_difference_rub'],0)


if __name__ == '__main__':
    unittest.main()
