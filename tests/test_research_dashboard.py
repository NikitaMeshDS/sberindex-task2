"""Scientific safeguards of the offline evidence explorer."""

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "tools/build_research_dashboard.py"
spec = importlib.util.spec_from_file_location("dashboard_builder", PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class DashboardEvidenceTests(unittest.TestCase):
    def test_news_future_and_wrong_geography_are_hidden(self):
        item = {
            "region_code": 56,
            "municipality_ids": [1673],
            "available_from": "2024-04-07",
        }
        self.assertFalse(builder.news_visible(item, 56, 1673, "2024-04-06"))
        self.assertFalse(builder.news_visible(item, 56, 1665, "2024-04-07"))
        self.assertFalse(builder.news_visible(item, 45, 1673, "2024-04-07"))
        self.assertTrue(builder.news_visible(item, 56, 1673, "2024-04-07"))

    def test_regional_context_is_not_fabricated_municipal_evidence(self):
        item = {
            "region_code": 56,
            "municipality_ids": [],
            "available_from": "2024-04-07",
        }
        self.assertTrue(builder.news_visible(item, 56, 1665, "2024-04-07"))
        self.assertFalse(builder.news_visible(item, 45, 1665, "2024-04-07"))
        self.assertEqual(item["municipality_ids"], [])

    def test_missing_values_stay_null_and_payload_is_strict_json(self):
        values = builder.clean(
            {"x": [np.nan, np.inf, -np.inf, np.int64(3), np.float64(1.25)]}
        )
        self.assertEqual(values, {"x": [None, None, None, 3, 1.25]})
        self.assertEqual(json.loads(json.dumps(values, allow_nan=False)), values)
