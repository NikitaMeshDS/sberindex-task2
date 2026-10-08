"""Operational forecast invariants, independent of archived experiment outputs."""

import unittest

import pandas as pd

from sberindex.forecasting.predict import make_forecast


class ForecastCommandTests(unittest.TestCase):
    def fixture(self):
        rows = [
            {
                "territory_id": i,
                "category": "Все категории",
                "date": f"2023-{m:02d}",
                "value": float((100 + i) * m),
            }
            for i in [1, 2]
            for m in range(1, 13)
        ]
        rows += [
            {
                "territory_id": i,
                "category": "Все категории",
                "date": "2024-12",
                "value": 2400.0 + i,
            }
            for i in [1, 2]
        ]
        data = pd.DataFrame(rows)
        lookup = pd.DataFrame({"territory_id": [1, 2], "region_code": [10, 10]})
        cfg = {
            "origin": "2024-12",
            "category": "Все категории",
            "profile_year": 2023,
            "horizons": [1, 3, 6, 12],
            "growth_rate": 0.172,
            "region_weight": 0.5,
            "minimum_region_municipalities": 1,
            "as_of_date": "2025-01-01",
        }
        return data, lookup, cfg

    def test_future_facts_do_not_change_forecast(self):
        raw, lookup, cfg = self.fixture()
        expected, _ = make_forecast(raw, lookup, cfg)
        future = pd.DataFrame(
            [
                {
                    "territory_id": 1,
                    "category": "Все категории",
                    "date": "2025-01",
                    "value": 9999999,
                }
            ]
        )
        actual, audit = make_forecast(pd.concat([raw, future]), lookup, cfg)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(audit["future_rows_excluded"], 1)

    def test_formula_and_year_boundary(self):
        raw, lookup, cfg = self.fixture()
        result, _ = make_forecast(raw, lookup, cfg)
        row = result[(result.territory_id == 1) & (result.horizon == 1)].iloc[0]
        self.assertAlmostEqual(row.predicted, 2401 / 12 * 1.172)
        self.assertEqual(row.target, "2025-01")

    def test_new_profile_year_is_supported(self):
        raw, lookup, cfg = self.fixture()
        raw = raw[raw.date.str.startswith("2023")].copy()
        raw["date"] = raw.date.str.replace("2023", "2024")
        cfg.update(profile_year=2024, origin="2024-12")
        result, _ = make_forecast(raw, lookup, cfg)
        self.assertEqual(len(result), 8)

    def test_partial_profile_is_rejected(self):
        raw, lookup, cfg = self.fixture()
        raw = raw[raw.date != "2023-06"]
        with self.assertRaises(ValueError):
            make_forecast(raw, lookup, cfg)

    def test_duplicate_month_is_rejected(self):
        raw, lookup, cfg = self.fixture()
        with self.assertRaises(ValueError):
            make_forecast(pd.concat([raw, raw.iloc[:1]]), lookup, cfg)

    def test_publication_after_as_of_is_excluded(self):
        raw, lookup, cfg = self.fixture()
        raw["available_from"] = "2024-01-01"
        raw.loc[raw.date == "2024-12", "available_from"] = "2025-02-01"
        with self.assertRaises(ValueError):
            make_forecast(raw, lookup, cfg)

    def test_growth_source_must_be_available(self):
        raw, lookup, cfg = self.fixture()
        cfg["growth_available_from"] = "2025-02-01"
        with self.assertRaises(ValueError):
            make_forecast(raw, lookup, cfg)

    def test_invalid_parameters_are_rejected(self):
        raw, lookup, cfg = self.fixture()
        for patch in [
            {"region_weight": 2},
            {"horizons": [0]},
            {"horizons": [1.5]},
            {"growth_rate": -1},
            {"profile_year": 2025},
        ]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                make_forecast(raw, lookup, dict(cfg, **patch))
