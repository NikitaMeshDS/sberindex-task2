import unittest
import numpy as np
try:
    from sberindex.forecasting import prophet_seasonality as m
except ImportError:
    m=None

class ProphetSeasonalityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(m,'seasonal Prophet adapter is missing')
        self.cfg=dict(yearly_fourier_order=3,yearly_period_days=365.25,seasonality_prior_scale=.1,
            regressor_prior_scale=.1,seasonality_mode='multiplicative',seed=20261006)

    def test_yearly_component_present_and_low_order(self):
        model=m.build_model('prophet_yearly3',self.cfg)
        self.assertEqual(model.seasonalities['yearly']['fourier_order'],3)
        self.assertEqual(model.seasonalities['yearly']['prior_scale'],.1)
        self.assertEqual(model.seasonalities['yearly']['mode'],'multiplicative')

    def test_profile_regressor_not_standardized(self):
        model=m.build_model('prophet_pooled_profile',self.cfg)
        reg=model.extra_regressors['pooled_profile']
        self.assertFalse(reg['standardize']);self.assertEqual(reg['mode'],'multiplicative')

    def test_regressor_uses_fixed_calendar_profile_and_rejects_future_history(self):
        profile=np.arange(1.,13.)
        train,future=m.design_frames(np.arange(1.,14.),profile,3)
        np.testing.assert_array_equal(future.pooled_profile,[1.,2.,3.])
        self.assertEqual(train.ds.max().strftime('%Y-%m'),'2024-01')
        self.assertEqual(future.ds.min().strftime('%Y-%m'),'2024-02')
        with self.assertRaises(ValueError):m.design_frames([1,np.nan],profile,3)
