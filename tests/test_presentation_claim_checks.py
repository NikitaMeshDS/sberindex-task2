"""Substantive checks; integration writes only to a temporary directory."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "presentation_claim_checks", ROOT / "tools/presentation_claim_checks.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class PresentationClaimChecks(unittest.TestCase):
    def test_convex_endpoint_bound_and_date_weighting(self):
        frame = pd.DataFrame(
            {
                "actual": [100.0, 120.0, 300.0],
                "predicted": [100.0, 100.0, 100.0],
                "year_bridges": [1, 1, 0],
                "target": ["a", "a", "b"],
                "territory_id": [1, 2, 1],
                "origin": ["x", "x", "y"],
                "horizon": [1, 1, 1],
            }
        )
        comp = pd.DataFrame(
            {"horizon": [1], "model": ["prophet_saved"], "MAE": [110.0]}
        )
        result = checks.continuous_growth_check(frame, comp)[0]
        self.assertEqual(result["lower_MAE"], 105.0)
        self.assertEqual(result["upper_MAE"], 105.0)
        self.assertTrue(result["all_rates_strictly_better"])
        for rate in np.linspace(0.07, 0.17, 101):
            self.assertLessEqual(
                checks.date_balanced_mae(frame, rate),
                result["continuous_upper_bound_MAE"] + 1e-12,
            )

    def test_reject_non_affine_bridge_and_duplicates(self):
        frame = pd.DataFrame(
            {
                "actual": [1.0],
                "predicted": [1.0],
                "year_bridges": [2],
                "target": ["a"],
                "territory_id": [1],
                "origin": ["x"],
                "horizon": [1],
            }
        )
        comp = pd.DataFrame({"horizon": [1], "model": ["p"], "MAE": [2.0]})
        with self.assertRaisesRegex(ValueError, "k in"):
            checks.continuous_growth_check(frame, comp)
        frame.year_bridges = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checks.continuous_growth_check(pd.concat([frame, frame]), comp)

    def test_rank_sign_population_and_ties(self):
        rows = []
        for id_, value in [(1, 0.1), (2, -0.2), (3, 0.1)]:
            for month in range(1, 13):
                rows.append(
                    {
                        "territory_id": id_,
                        "target": f"2024-{month:02d}",
                        "peer_residual_log": value,
                    }
                )
        rows.append({"territory_id": 4, "target": "2024-05", "peer_residual_log": 0.3})
        result = checks.rank_panel(pd.DataFrame(rows))
        one = result[
            (result.territory_id == 1) & (result.population == "may_available")
        ].iloc[0]
        self.assertEqual(one.population_n, 4)
        self.assertEqual(one.signed_desc_rank, 2)
        self.assertEqual(one.absolute_desc_rank, 3)
        self.assertEqual(
            result[result.population == "complete_2024_peer_residual"]
            .population_n.unique()
            .tolist(),
            [3],
        )

    @unittest.skipIf(
        (ROOT / "compact_distribution.json").exists()
        and not all((ROOT / name).is_file() for name in checks.INPUTS),
        "Нужен исследовательский архив из Releases: см. docs/GETTING_STARTED.md",
    )
    def test_saved_evidence_and_source_preservation(self):
        before = {p: checks.sha(ROOT / p) for p in checks.INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            evidence = checks.run(out=Path(tmp))
            self.assertTrue(evidence["growth"]["continuous_verified"])
            self.assertEqual(
                [r["horizon"] for r in evidence["growth"]["horizons"]], [1, 3, 6, 12]
            )
            self.assertFalse(evidence["independent_validation"])
            panel = pd.read_parquet(
                ROOT / "reports/peer_residual_review/residual_panel.parquet"
            )
            observed = panel.dropna(subset=["own_residual_log"])
            np.testing.assert_allclose(
                observed.own_residual_log,
                np.log(observed.actual / observed.predicted),
                rtol=0,
                atol=1e-14,
            )
            self.assertIn("log(actual/prediction)", evidence["orsk"]["reference"])
            self.assertNotIn("log1p", evidence["orsk"]["reference"])
            self.assertFalse(evidence["orsk"]["claimed_rank39_of1998_verified"])
            self.assertAlmostEqual(
                evidence["orsk"]["residual_ordinary_percent"],
                100 * np.expm1(evidence["orsk"]["residual_log"]),
            )
            self.assertAlmostEqual(
                evidence["orsk"]["may_july_orsk_minus_orenburg_mean_pp"],
                7.178227552311249,
            )
            self.assertEqual(
                evidence["orsk"]["alerts_both_cities_all_frozen_methods"], 0
            )
            evaluation = {
                r["method"]: r
                for r in evidence["detectors"]["rows"]
                if r["split"] == "evaluation"
            }
            ewma = evaluation["ewma"]
            self.assertEqual(ewma["true_onsets"], 720)
            self.assertEqual(ewma["onset_delay_denominator"], 420)
            self.assertEqual(ewma["missed_onsets_excluded_from_delay"], 300)
            self.assertEqual(ewma["null_mo_months"], 4320)
            self.assertEqual(ewma["null_false_alarm_count"], 2)
            self.assertGreater(evaluation["pelt"]["onset_f1"], ewma["onset_f1"])
            self.assertIsNone(evaluation["pelt"]["mean_onset_delay_months"])
            self.assertLess(
                evaluation["spike"]["mean_onset_delay_months"],
                ewma["mean_onset_delay_months"],
            )
            audit = json.loads((Path(tmp) / "audit.json").read_text())
            self.assertEqual(
                audit["source_code_sha256"],
                checks.sha(ROOT / "tools/presentation_claim_checks.py"),
            )
        self.assertEqual(before, {p: checks.sha(ROOT / p) for p in checks.INPUTS})

    @unittest.skipIf(
        (ROOT / "compact_distribution.json").exists()
        and not (ROOT / "reports/short_history_review/series_metrics.csv").is_file(),
        "Нужен исследовательский архив из Releases: см. docs/GETTING_STARTED.md",
    )
    def test_corrupt_onset_summary_is_detected(self):
        summary = pd.read_csv(ROOT / "reports/short_history_review/comparison.csv")
        series = pd.read_csv(ROOT / "reports/short_history_review/series_metrics.csv")
        mask = (
            (summary.split == "evaluation")
            & (summary.method == "ewma")
            & (summary["shape"] == "all")
        )
        summary.loc[mask, "onset_f1"] = 0.99
        with self.assertRaises(AssertionError):
            checks.detector_checks(summary, series)


if __name__ == "__main__":
    unittest.main()
