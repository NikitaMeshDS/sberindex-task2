import unittest
import numpy as np
import pandas as pd

try:
    from sberindex.forecasting import foundation_delay_audit as audit
except ImportError:
    audit = None


class DelayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, 'foundation delay audit is missing')

    def test_calendar_horizon_counts_reporting_gap(self):
        self.assertEqual(audit.timing('2024-09',1), ('2024-08','2024-07',2))
        self.assertEqual(audit.timing('2024-09',2), ('2024-08','2024-06',3))
        self.assertEqual(audit.timing('2024-09',0), ('2024-08','2024-08',1))

    def test_future_target_missingness_does_not_select_inference_ids(self):
        months=pd.period_range('2023-01',periods=24,freq='M').astype(str)
        panel=pd.DataFrame(np.ones((3,24)),index=[1,2,3],columns=months)
        panel.loc[2,'2024-09']=np.nan
        panel.loc[3,'2024-03']=np.nan
        before=audit.history_context(panel,'2024-07',np.ones(12))
        self.assertEqual(before.index.tolist(),[1,2])
        panel.loc[:,panel.columns>'2024-07']=1e9
        pd.testing.assert_frame_equal(before,audit.history_context(panel,'2024-07',np.ones(12)))

    def test_restore_two_steps_uses_target_season_not_decision_month(self):
        profile=np.arange(1,13,dtype=float)
        restored=audit.restore_target(np.array([[2.,3.]]),'2024-07',profile)
        np.testing.assert_array_equal(restored,[[16.,27.]])

    def test_intersection_is_strict_across_models_and_delays(self):
        rows=[]
        for model in ['a','b']:
            for delay in [0,1,2]:
                for city in [1,2]:
                    if (model,delay,city)==('b',2,2):continue
                    rows.append({'territory_id':city,'target':'2024-09','model':model,
                                 'reporting_delay_months':delay,'actual':100.,'predicted':90.})
        scored=audit.common_pairs(pd.DataFrame(rows),['a','b'])
        self.assertEqual(scored.territory_id.unique().tolist(),[1])
        self.assertEqual(len(scored),6)


if __name__=='__main__':
    unittest.main()
