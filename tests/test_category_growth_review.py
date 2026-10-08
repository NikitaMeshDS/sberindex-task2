import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sberindex.forecasting.category_growth_review import (
    MONTHS,
    category_panel,
    checkpoint_manifest,
    fixed_cohort,
)
from sberindex.forecasting.full_cohort_review import eligible_ids, evaluation_pairs
from sberindex.forecasting.growth_bridge_review import growth_for_origin
from sberindex.forecasting.hierarchical_review import fit_profiles


class ReviewTests(unittest.TestCase):
    def test_fixed_category_cohort_does_not_select2024_survivors(self):
        rows = []
        for cat in ["Все категории", "Здоровье"]:
            for city in [1, 2, 3]:
                for i, date in enumerate(MONTHS):
                    value = 10.0 + i
                    if city == 1 and i >= 12:
                        value = np.nan
                    if city == 2 and i == 2:
                        value = np.nan
                    if cat == "Здоровье" and city == 3 and i == 3:
                        value = 0
                    rows.append(
                        {
                            "category": cat,
                            "territory_id": city,
                            "date": date,
                            "value": value,
                        }
                    )
        raw = pd.DataFrame(rows)
        assert list(fixed_cohort(raw)) == [1, 3]
        panel = category_panel(raw, "Здоровье")
        assert list(fixed_cohort(raw).intersection(eligible_ids(panel))) == [1]
        panel.iloc[:, 12:] = 1e9
        assert list(eligible_ids(panel)) == [1]
        assert (
            evaluation_pairs(
                category_panel(raw, "Все категории").loc[1].to_numpy(), [1, 3, 6, 12]
            )[0]
            == []
        )

    def test_category_profiles_ignore_future_and_growth_requires_asof(self):
        panel = pd.DataFrame(np.arange(1.0, 73.0).reshape(3, 24), index=[1, 2, 3])
        regions = pd.Series([10, 10, 20], index=panel.index)
        a, ar = fit_profiles(panel, regions, 1)
        panel.iloc[:, 12:] = np.nan
        b, br = fit_profiles(panel, regions, 1)
        np.testing.assert_array_equal(a, b)
        for k in ar:
            np.testing.assert_array_equal(ar[k], br[k])
        source = {"available_from": "2023-12-28", "growth_pct": 17.2}
        assert growth_for_origin("2023-12", source) == 0.172
        with self.assertRaises(ValueError):
            growth_for_origin("2023-11", source)

    def test_restart_rejects_changed_backend_or_missing_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            checkpoint_manifest(directory, {"seed": 10, "source_sha": "a"})
            checkpoint_manifest(directory, {"seed": 10, "source_sha": "a"})
            with self.assertRaises(ValueError):
                checkpoint_manifest(directory, {"seed": 11, "source_sha": "a"})
            (directory / "manifest.json").unlink()
            (directory / "one.parquet").touch()
            with self.assertRaises(ValueError):
                checkpoint_manifest(directory, {"seed": 10})


if __name__ == "__main__":
    unittest.main()
