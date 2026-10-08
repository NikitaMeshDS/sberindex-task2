import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "presentation_cases",
    Path(__file__).resolve().parents[1] / "tools/presentation_comparison_cases.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PresentationEvidenceTests(unittest.TestCase):
    def test_interval_ignores_future_facts(self):
        frame = pd.DataFrame(
            {
                "target": ["2024-01", "2024-02", "2024-03", "2024-04"],
                "territory_id": [1] * 4,
                "actual": [101.0, 98.0, 103.0, 999.0],
                "predicted": [100.0] * 4,
            }
        )
        scales = pd.Series({1: 100.0})
        expected = m.calibration_for_origin(frame, "2024-03", scales)
        frame.loc[3, "actual"] = 1e9
        self.assertEqual(expected, m.calibration_for_origin(frame, "2024-03", scales))
        self.assertEqual(expected["max_calibration_target"], "2024-03")
        self.assertIsNone(m.calibration_for_origin(frame, "2024-02", scales))

    def test_quantile_finite_sample_rank(self):
        self.assertEqual(m.finite_sample_quantile(np.arange(1, 11), 0.8), 9.0)
        with self.assertRaises(ValueError):
            m.finite_sample_quantile([np.nan])

    def test_single_target_within_r2_is_undefined(self):
        frame = pd.DataFrame(
            {
                "actual": [100.0, 200.0],
                "predicted": [90.0, 210.0],
                "territory_id": [1, 2],
                "year_ago": [80.0, 160.0],
                "target": ["2024-12"] * 2,
                "origin": ["2023-12"] * 2,
            }
        )
        result = m.metrics(frame)
        self.assertIsNone(result["R2_within_MO"])
        self.assertEqual(result["within_MO_undefined_count"], 2)

    def test_matching_rejects_disagreeing_actuals(self):
        a = pd.DataFrame(
            {
                "territory_id": [1],
                "origin": ["2023-12"],
                "target": ["2024-01"],
                "horizon": [1],
                "actual": [100.0],
                "predicted": [90.0],
            }
        )
        b = a.copy()
        b["actual"] = 101.0
        with self.assertRaises(ValueError):
            m.matched_rows({"one": a, "two": b})
