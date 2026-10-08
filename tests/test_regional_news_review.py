"""Source provenance, geographic ambiguity and temporal leakage regressions."""

import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from sberindex.external.regional_news_review import DATA, OUT, ROOT, asof_features


class TemporalFeatures(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame(
            [
                dict(
                    available_from="2024-04-01",
                    region_code=56,
                    territory_ids="[1673]",
                    episode_id="flood",
                    historical_vintage_verified=False,
                ),
                dict(
                    available_from="2024-04-07",
                    region_code=56,
                    territory_ids="[1673]",
                    episode_id="flood",
                    historical_vintage_verified=False,
                ),
                dict(
                    available_from="2024-05-01",
                    region_code=56,
                    territory_ids="[1673]",
                    episode_id="future",
                    historical_vintage_verified=False,
                ),
                dict(
                    available_from="2024-04-05",
                    region_code=31,
                    territory_ids="[931]",
                    episode_id="other_region",
                    historical_vintage_verified=False,
                ),
            ]
        )

    def test_next_day_release_does_not_leak_into_previous_month(self):
        x = asof_features(self.rows, 56, 1673, "2024-03-01")
        self.assertTrue(np.isnan(x[[0, 2]]).all())
        np.testing.assert_array_equal(x[[1, 3]], [0, 0])

    def test_deduplicates_episode_excludes_future_and_other_region(self):
        np.testing.assert_array_equal(
            asof_features(self.rows, 56, 1673, "2024-04-01"), [1, 1, 1, 1]
        )

    def test_region_context_is_not_municipal_exposure(self):
        x = asof_features(self.rows, 56, 999999, "2024-04-01")
        self.assertEqual(x[0], 1)
        self.assertTrue(np.isnan(x[2]))
        self.assertEqual(x[3], 0)

    def test_strict_historical_vintage_excludes_current_snapshots(self):
        x = asof_features(self.rows, 56, 1673, "2024-04-01", strict=True)
        self.assertTrue(np.isnan(x[[0, 2]]).all())

    def test_old_records_expire(self):
        x = asof_features(self.rows, 56, 1673, "2024-09-01")
        self.assertTrue(np.isnan(x[[0, 2]]).all())


@unittest.skipIf(
    (ROOT / "compact_distribution.json").exists()
    and not (DATA / "registry.csv").is_file(),
    "Нужен исследовательский архив из Releases: см. docs/GETTING_STARTED.md",
)
class AuditArtifacts(unittest.TestCase):
    def test_captured_evidence_integrity(self):
        manifest = json.loads((DATA / "capture_manifest.json").read_text())
        for item in manifest:
            body = (ROOT / item["path"]).read_bytes()
            self.assertEqual(len(body), item["bytes"])
            self.assertEqual(hashlib.sha256(body).hexdigest(), item["sha256"])
        reg = pd.read_csv(DATA / "registry.csv")
        self.assertTrue(reg.claim_id.is_unique)
        failed = {
            x["path"] for x in manifest if x["capture_status"] == "failed_excluded"
        }
        self.assertFalse(failed & set(reg.snapshot_path))

    def test_publication_delay_and_unknown_coverage(self):
        reg = pd.read_csv(DATA / "registry.csv")
        delay = pd.to_datetime(reg.available_from) - pd.to_datetime(reg.published_date)
        self.assertTrue(delay.eq(pd.Timedelta(days=1)).all())
        self.assertFalse(reg.historical_vintage_verified.any())
        self.assertTrue(
            reg.allowed_role.eq(
                "external_context_not_spending_shift_ground_truth"
            ).all()
        )
        f = pd.read_csv(OUT / "asof_features.csv")
        self.assertTrue(
            f.loc[f.selected_region_evidence == 0, "regional_episodes"].isna().all()
        )
        self.assertTrue(
            f.loc[f.selected_local_evidence == 0, "local_episodes"].isna().all()
        )
        self.assertFalse(f.source_coverage_known.any())

    def test_ambiguity_is_audited(self):
        geo = pd.read_csv(OUT / "geography_audit.csv")
        x = geo[
            (geo.source_number.astype(str) == "7") & (geo.alias == "Дальнереченский")
        ]
        self.assertEqual(x.territory_id.item(), 780)
        self.assertEqual(set(json.loads(x.candidates.item())), {780, 788})
        unresolved = pd.read_csv(OUT / "unresolved_mentions.csv")
        self.assertTrue((unresolved.decision == "exclude_ambiguous_place").any())

    def test_saved_batch_features_match_temporal_reference(self):
        reg = pd.read_csv(DATA / "registry.csv")
        features = pd.read_csv(OUT / "asof_features.csv")
        covered = features[features.selected_region_evidence == 1].sample(
            n=15, random_state=7
        )
        unknown = features[features.selected_region_evidence == 0].sample(
            n=10, random_state=7
        )
        for row in pd.concat([covered, unknown]).itertuples():
            expected = asof_features(reg, row.region_code, row.territory_id, row.origin)
            actual = [
                row.regional_episodes,
                row.selected_region_evidence,
                row.local_episodes,
                row.selected_local_evidence,
            ]
            np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_paired_ablation_and_no_real_event_labels(self):
        p = pd.read_parquet(OUT / "predictions.parquet")
        counts = p.groupby(["territory_id", "origin", "target"]).model.nunique()
        self.assertTrue(counts.eq(2).all())
        self.assertEqual(p.target.nunique(), 11)
        alignment = pd.read_csv(OUT / "event_alignment.csv")
        self.assertFalse(alignment.spending_shift_ground_truth.any())
        self.assertTrue(alignment.lead_days_to_spending_publication.isna().all())
        unknown = alignment.event_start.isna()
        self.assertTrue(
            alignment.loc[unknown, "lead_days_to_external_onset_assumed"].isna().all()
        )
        audit = json.loads((OUT / "ablation_audit.json").read_text())
        self.assertEqual(audit["cohort_ids"], 2075)
        self.assertFalse(audit["independent_holdout"])
        self.assertFalse(audit["primary_model_changed"])
        self.assertEqual(
            audit["protocol_sha256"],
            hashlib.sha256((DATA / "protocol.json").read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
