"""Control series must not be described as detected changepoints."""

import unittest

import numpy as np

from sberindex.detection.yoy_shift_review import synthetic_summary


class ControlLabelTests(unittest.TestCase):
    def test_seasonal_and_null_controls_have_no_change_detection_metric(self):
        score = np.ones((4, 12))
        for shape in ("null", "seasonal"):
            row = synthetic_summary(shape, "ewma", score, 0.5)
            self.assertIsNone(row["detected_by_second_postchange"])
            self.assertEqual(row["reference_window_any_alert"], 1.0)

    def test_injected_change_has_bounded_postchange_detection_window(self):
        score = np.zeros((4, 12))
        score[:, 10] = 1
        row = synthetic_summary("sustained", "cusum", score, 0.5)
        self.assertEqual(row["any_alert"], 1.0)
        self.assertEqual(row["detected_by_second_postchange"], 0.0)
