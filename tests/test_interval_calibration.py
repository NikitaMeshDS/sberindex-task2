import unittest
import numpy as np
import pandas as pd
try:
    from sberindex.forecasting import interval_calibration as m
except ImportError:
    m = None

class IntervalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(m, 'new interval calibration module is missing')
        self.cfg=dict(calibration_start='2024-07',evaluation_targets=['2024-09','2024-10','2024-11','2024-12'],
            levels=[.8,.95],strategies=['raw_3m','scaled_2023_3m','scaled_forecast_3m','adaptive_scaled_2023_3m'],
            window_months=3,gamma=.05,alpha_min=.005,alpha_max=.5,normalizer_floor_rub=1.)
        self.frame=pd.DataFrame([dict(territory_id=i,target=t,model='blend_75',predicted=100.*i,
            actual=100.*i+float((j+1)*i)) for j,t in enumerate(pd.period_range('2024-07',periods=6,freq='M').astype(str)) for i in range(1,17)])
        self.scale=pd.Series({i:100.*i for i in range(1,17)})

    def test_current_and_future_actuals_cannot_change_issued_bounds(self):
        a,_=m.calibrate_panel(self.frame,self.scale,self.cfg)
        f=self.frame.copy();f.loc[f.target>='2024-09','actual']+=10000
        b,_=m.calibrate_panel(f,self.scale,self.cfg)
        pd.testing.assert_frame_equal(a[a.target=='2024-09'][['lower','upper']],b[b.target=='2024-09'][['lower','upper']])
        f=self.frame.copy();f.loc[f.target>='2024-11','actual']+=10000
        b,_=m.calibrate_panel(f,self.scale,self.cfg)
        pd.testing.assert_frame_equal(a[a.target<='2024-10'][['lower','upper']],b[b.target<='2024-10'][['lower','upper']])

    def test_raw_control_matches_existing_finite_quantile_and_clip(self):
        a,_=m.calibrate_panel(self.frame,self.scale,self.cfg)
        from sberindex.forecasting.operational_audit import finite_quantile
        cal=self.frame[self.frame.target<'2024-09']
        q=finite_quantile((cal.actual-cal.predicted).abs(),.8)
        part=a[(a.target=='2024-09')&(a.strategy=='raw_3m')&(a.nominal_coverage==.8)]
        np.testing.assert_allclose(part.lower,np.maximum(0,part.predicted-q))
        np.testing.assert_allclose(part.upper,part.predicted+q)

    def test_missing_scale_and_duplicate_keys_refuse(self):
        with self.assertRaises(ValueError):m.calibrate_panel(self.frame,self.scale.drop(1),self.cfg)
        with self.assertRaises(ValueError):m.calibrate_panel(pd.concat([self.frame,self.frame.iloc[:1]]),self.scale,self.cfg)

    def test_adaptation_moves_after_feedback_not_before(self):
        f=self.frame.copy();f.loc[f.target=='2024-09','actual']+=10000
        _,audit=m.calibrate_panel(f,self.scale,self.cfg)
        part=audit[(audit.strategy=='adaptive_scaled_2023_3m')&(audit.nominal_coverage==.95)].set_index('target')
        self.assertAlmostEqual(part.loc['2024-09','alpha_before'],.05)
        self.assertLess(part.loc['2024-10','alpha_before'],.05)
        self.assertEqual(part.loc['2024-09','alpha_after'],part.loc['2024-10','alpha_before'])

    def test_missing_calibration_month_refuses_instead_of_shortening_window(self):
        f=self.frame[self.frame.target!='2024-08']
        with self.assertRaises(ValueError):m.calibrate_panel(f,self.scale,self.cfg)
