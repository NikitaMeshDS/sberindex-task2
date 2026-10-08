import unittest

import numpy as np
import pandas as pd

from sberindex.external.news_forecast_diagnostic_review import known_at_origin
from sberindex.external.news_interval_review import (
    event_intervals,
    interval_score,
    validate_calibrations,
)


class NewsIntervalTests(unittest.TestCase):
    def test_widening_only_known_events_and_preserves_center(self):
        lower, upper = event_intervals([100, 100], [10, 10], [True, False], 1.5)
        np.testing.assert_array_equal(lower, [85, 90])
        np.testing.assert_array_equal(upper, [115, 110])
        np.testing.assert_array_equal((lower + upper) / 2, [100, 100])
        actual = np.array([112, 112])
        np.testing.assert_array_equal(
            (lower <= actual) & (actual <= upper), [True, False]
        )

    def test_interval_score_penalizes_misses_and_excess_width(self):
        np.testing.assert_array_equal(
            interval_score([-1, 5, 11], [0] * 3, [10] * 3, 0.2), [20, 10, 20]
        )
        self.assertGreater(interval_score([5], [0], [20], 0.2)[0], 10)

    def test_future_news_does_not_expand_past_forecast(self):
        events = [{"territory_id": 1, "available_from": "2024-06-01"}]
        known = known_at_origin(events, 1, "2024-05")
        self.assertFalse(known)
        lower, upper = event_intervals([100], [10], [known], 1.5)
        np.testing.assert_array_equal(lower, [90])
        np.testing.assert_array_equal(upper, [110])

    def test_rejects_future_calibration(self):
        c = pd.DataFrame(
            [
                {
                    "origin": "2024-04",
                    "max_calibration_target": "2024-05",
                    "calibration_end": "2024-05",
                    "nominal_coverage": 0.8,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "future"):
            validate_calibrations(c, 0.8)
        c[["max_calibration_target", "calibration_end"]] = "2024-04"
        validate_calibrations(c, 0.8)


if __name__ == "__main__":
    unittest.main()
