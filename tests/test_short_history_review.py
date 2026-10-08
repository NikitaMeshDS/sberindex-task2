import unittest
import numpy as np
import pandas as pd
from sberindex.detection import short_history_review as study


class ShortHistoryTests(unittest.TestCase):
    def test_short_pulse_boundaries_are_observable(self):
        panel = study.make_panel(123, 20)
        for item in panel:
            self.assertEqual(len(item["log_values"]), 24)
            self.assertEqual(len(item["residual"]), 12)
            self.assertTrue(all(0 <= t < 12 for t in item["truth"]))
            if item["shape"] == "pulse":
                self.assertEqual(len(item["truth"]), 2)
                self.assertLess(item["truth"][0], item["truth"][1])

    def test_return_is_neutral_in_onset_metric_but_early_false_alarm_is_not(self):
        item = {"truth": [4, 8]}
        score = study.onset_counts(item, [1, 4, 8])
        self.assertEqual(score, (1, 1, 0))
        self.assertEqual(study.onset_counts(item, [8]), (0, 0, 1))

    def test_family_calibration_does_not_read_evaluation(self):
        cal = np.random.default_rng(44).normal(0, 0.03, (200, 12))
        parameters = study.calibrate(cal)
        self.assertIn("cusum_spike", parameters)
        x = np.zeros((1, 12))
        y = x.copy()
        y[:, 6:] = 0.5
        a = study.predict_online(x, "cusum_spike", parameters)
        b = study.predict_online(y, "cusum_spike", parameters)
        self.assertEqual([t for t in a[0] if t < 6], [t for t in b[0] if t < 6])

    def test_choice_ignores_evaluation_and_enforces_null_budget(self):
        data = pd.DataFrame(
            [
                dict(
                    split="selection",
                    method="cusum",
                    shape="all",
                    onset_f1=0.6,
                    null_false_alarms_per100_mo_month=0.1,
                ),
                dict(
                    split="selection",
                    method="spike",
                    shape="all",
                    onset_f1=0.9,
                    null_false_alarms_per100_mo_month=0.3,
                ),
                dict(
                    split="evaluation",
                    method="spike",
                    shape="all",
                    onset_f1=1.0,
                    null_false_alarms_per100_mo_month=0.0,
                ),
            ]
        )
        self.assertEqual(study.choose(data), "cusum")
        data.loc[data.split == "evaluation", "onset_f1"] = 0
        self.assertEqual(study.choose(data), "cusum")
        data.loc[data.split == "selection", "null_false_alarms_per100_mo_month"] = 1
        self.assertIsNone(study.choose(data))
