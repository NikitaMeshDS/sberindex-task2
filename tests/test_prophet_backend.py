import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

try:
    from sberindex.forecasting.prophet_backend import legacy_predict
except ImportError:
    legacy_predict = None


class ProphetBackendTests(unittest.TestCase):
    def test_legacy_configuration_and_raw_output_preserved(self):
        self.assertIsNotNone(legacy_predict)
        dates = pd.date_range("2023-01", periods=14, freq="MS")
        with patch("sberindex.forecasting.prophet_backend.Prophet") as constructor:
            model = constructor.return_value
            model.predict.return_value = pd.DataFrame({"yhat": [-3.0, 4.0]})
            result = legacy_predict(np.arange(12.0), dates[:12], dates[12:])
            constructor.assert_called_once_with(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                uncertainty_samples=0,
            )
            self.assertEqual(model.fit.call_args.kwargs, {})
            np.testing.assert_array_equal(result, [-3.0, 4.0])
            np.testing.assert_array_equal(
                model.fit.call_args.args[0].y, np.arange(12.0)
            )
            np.testing.assert_array_equal(
                model.predict.call_args.args[0].ds, dates[12:]
            )

    def test_rejects_future_dates_overlapping_history(self):
        self.assertIsNotNone(legacy_predict)
        dates = pd.date_range("2023-01", periods=12, freq="MS")
        with self.assertRaises(ValueError):
            legacy_predict(np.ones(12), dates, dates[-1:])
