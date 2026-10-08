import unittest
import numpy as np
import pandas as pd
try:
    from sberindex.forecasting import grouped_intervals as m
except ImportError:
    m=None

class GroupedIntervalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(m,'grouped interval module is missing')
        self.cfg=dict(calibration_start='2024-07',evaluation_targets=['2024-09','2024-10'],
            levels=[.8],strategies=['raw_3m','scaled_forecast_3m'],window_months=3,
            gamma=.05,alpha_min=.005,alpha_max=.5,normalizer_floor_rub=1.)
        self.frame=pd.DataFrame([dict(territory_id=i,target=t,model='blend_75',predicted=100.*i,
            actual=100.*i+(j+1)*i) for j,t in enumerate(['2024-07','2024-08','2024-09','2024-10']) for i in range(1,17)])
        self.scale=pd.Series({i:100.*i for i in range(1,17)})
        self.groups=pd.Series({i:'Q1' if i<=8 else 'Q2' for i in range(1,17)})

    def test_other_group_cannot_change_group_bounds(self):
        a,_=m.calibrate_by_group(self.frame,self.scale,self.groups,self.cfg)
        f=self.frame.copy();f.loc[f.territory_id>8,'actual']+=10000
        b,_=m.calibrate_by_group(f,self.scale,self.groups,self.cfg)
        pd.testing.assert_frame_equal(a[a.expense_quartile=='Q1'][['lower','upper']],b[b.expense_quartile=='Q1'][['lower','upper']])

    def test_current_and_future_actuals_cannot_change_issued_group_bounds(self):
        a,_=m.calibrate_by_group(self.frame,self.scale,self.groups,self.cfg)
        f=self.frame.copy();f.loc[f.target>='2024-09','actual']+=10000
        b,_=m.calibrate_by_group(f,self.scale,self.groups,self.cfg)
        pd.testing.assert_frame_equal(a[a.target=='2024-09'][['lower','upper']],b[b.target=='2024-09'][['lower','upper']])

    def test_missing_group_refuses(self):
        with self.assertRaises(ValueError):m.calibrate_by_group(self.frame,self.scale,self.groups.drop(1),self.cfg)
