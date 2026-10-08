import unittest

import numpy as np
import pandas as pd

from sberindex.detection.event_conditioned_review import event_available
from sberindex.detection.yoy_shift_review import regional_growth, sustained_scores
from sberindex.external.news_event_expanded_review import reliable_page
from sberindex.external.news_event_extraction_review import resolve_geo, validate_output
from sberindex.external.news_event_field_review import quoted_geography, validate_fields
from sberindex.external.news_forecast_diagnostic_review import (
    date_balanced_mae,
    known_at_origin,
)


def dictionary():
    return pd.DataFrame(
        {
            "municipal_district_name_short": [
                "Орск",
                "Дальнереченский",
                "Дальнереченский",
            ],
            "territory_id": [1, 2, 3],
            "region_code": [56, 25, 25],
            "year": [2024] * 3,
        }
    )


def output():
    return {
        "publication_date": "2024-04-06",
        "event_date": None,
        "available_from": "2024-04-07",
        "region": 56,
        "municipality_id": None,
        "municipality_name": "Орск",
        "event_type": "flood",
        "severity": "unknown",
        "confidence": 0.8,
        "evidence_quote": "Орск: подтопление домов",
    }


class NewsEventReviewTests(unittest.TestCase):
    def test_ambiguous_geography_abstains(self):
        self.assertEqual(
            resolve_geo("Дальнереченский", 25, 2024, dictionary()),
            (None, "ambiguous_exact"),
        )
        self.assertIsNone(resolve_geo("Орск", 56, 2023, dictionary())[0])

    def test_unsupported_quote_cannot_be_claimed(self):
        with self.assertRaisesRegex(ValueError, "unsupported_quote"):
            validate_output(output(), "Орск: пожары", "2024-04-06", 56, dictionary())

    def test_availability_and_date_not_model_controlled(self):
        x = output()
        x["event_date"] = "2024-04-09"
        with self.assertRaisesRegex(ValueError, "future_event_date"):
            validate_output(x, x["evidence_quote"], "2024-04-06", 56, dictionary())
        x = output()
        x["available_from"] = "2024-04-06"
        with self.assertRaisesRegex(ValueError, "availability_mismatch"):
            validate_output(x, x["evidence_quote"], "2024-04-06", 56, dictionary())

    def test_exact_quote_geo_supported_and_unknown_confidence_nullable(self):
        x = output()
        x["confidence"] = "unknown"
        r = validate_output(x, x["evidence_quote"], "2024-04-06", 56, dictionary())
        self.assertEqual(r["municipality_id"], 1)
        self.assertIsNone(r["confidence"])

    def test_asof_events_do_not_relabel_earlier_targets_or_other_ids(self):
        events = [{"territory_id": 1, "available_from": "2024-04-07"}]
        self.assertFalse(event_available(events, 1, "2024-03", 2))
        self.assertTrue(event_available(events, 1, "2024-04", 0))
        self.assertFalse(event_available(events, 2, "2024-04", 0))
        self.assertFalse(event_available(events, 1, "2024-06", 0))
        self.assertFalse(event_available([], 1, "2024-04", 0))

    def test_future_perturbation_cannot_change_monitoring_history(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 0.02, (20, 12))
        y = x.copy()
        y[:, 8:] += 10
        for left, right in zip(sustained_scores(x), sustained_scores(y)):
            np.testing.assert_allclose(left[:, :8], right[:, :8])

    def test_regional_common_growth_is_removed_without_future_information(self):
        values = np.ones((7, 24)) * 100
        values[:, 12:] = 110
        regions = np.ones(7)
        self.assertTrue(np.allclose(regional_growth(values, regions), 0))
        changed = values.copy()
        changed[0, 15] = 150
        residual = regional_growth(changed, regions)
        self.assertGreater(residual[0, 3], 0)
        np.testing.assert_allclose(
            residual[:, :3], regional_growth(values, regions)[:, :3]
        )

    def test_field_replay_abstains_onset_and_preserves_quoted_orsk_exposure(self):
        x = output()
        x["municipality_name"] = None
        x["event_date"] = "2024-04-06"
        x["evidence_quote"] = "В Орске спасатели продолжают эвакуировать жителей"
        x["event_type"] = "evacuation"
        r = validate_fields(x, x["evidence_quote"], "2024-04-06", 56, dictionary())
        self.assertIsNone(r["event_date"])
        self.assertIn("event_date", r["field_errors"])
        self.assertEqual(r["municipality_id"], 1)
        self.assertEqual(r["geography_evidence_token"], "орске")

    def test_field_replay_cannot_salvage_unsupported_quote(self):
        with self.assertRaisesRegex(ValueError, "unsupported_quote"):
            validate_fields(output(), "Иная новость", "2024-04-06", 56, dictionary())

    def test_quoted_ner_abstains_when_official_alias_is_ambiguous(self):
        tid, _, _, status = quoted_geography(
            "Дальнереченского района", 25, 2024, dictionary()
        )
        self.assertIsNone(tid)
        self.assertEqual(status, "ambiguous_quote")

    def test_news_diagnostic_equal_date_weighting(self):
        self.assertEqual(
            date_balanced_mae([0, 100, 100], ["2024-01", "2024-02", "2024-02"]), 50
        )

    def test_search_snippet_is_not_a_saved_article_body(self):
        body = "6 апреля 2024 " + ("Подтоплены дома. " * 30)
        self.assertFalse(reliable_page("Search query result", body, "2024-04-06")[0])
        self.assertTrue(
            reliable_page("Source: open Content type: text/html", body, "2024-04-06")[0]
        )
        self.assertFalse(
            reliable_page("Source: open Content type: text/html", body, "2024-04-07")[0]
        )

    def test_forecast_news_available_only_before_origin(self):
        events = [{"territory_id": 1, "available_from": "2024-04-07"}]
        self.assertFalse(known_at_origin(events, 1, "2024-03"))
        self.assertTrue(known_at_origin(events, 1, "2024-04"))
        self.assertFalse(known_at_origin(events, 2, "2024-04"))


if __name__ == "__main__":
    unittest.main()
