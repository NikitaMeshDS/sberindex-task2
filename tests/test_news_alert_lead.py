import unittest
import pandas as pd
from sberindex.reporting.news_alert_lead import alert_lead


class NewsAlertLeadTests(unittest.TestCase):
    def test_april_news_precedes_april_observation_available_in_may(self):
        row = alert_lead("2024-04-07", "2024-04-05", ["2024-04"], 0, "2024-12")
        self.assertEqual(row["status"], "subsequent_alert")
        self.assertEqual(row["news_to_alert_days"], 24)
        self.assertEqual(row["event_to_news_days"], 2)
        late = alert_lead("2024-04-07", "2024-04-05", ["2024-04"], 1, "2024-12")
        self.assertEqual(late["news_to_alert_days"], 55)

    def test_no_alarm_is_censored_and_prior_alarm_not_reused(self):
        row = alert_lead("2024-04-07", None, ["2024-03"], 0, "2024-12")
        self.assertEqual(row["status"], "right_censored_no_subsequent_alert")
        self.assertIsNone(row["news_to_alert_days"])

    def test_future_news_outside_window_is_not_censored_as_observed(self):
        row = alert_lead("2025-01-10", None, [], 0, "2024-12")
        self.assertEqual(row["status"], "news_outside_observation_window")

    def test_release_on_news_day_remains_observed(self):
        row = alert_lead("2025-01-01", None, ["2024-12"], 0, "2024-12")
        self.assertEqual(row["status"], "subsequent_alert")
        self.assertEqual(row["news_to_alert_days"], 0)

    def test_series_exists_but_forecast_is_ineligible_is_distinguished(self):
        from sberindex.reporting.news_alert_lead import coverage_status

        self.assertEqual(
            coverage_status(1, {2}, {1, 2}),
            "no_eligible_h1_forecast_for_this_territory",
        )
        self.assertEqual(
            coverage_status(3, {2}, {1, 2}), "no_comparable_expense_series"
        )
        self.assertEqual(coverage_status(2, {2}, {1, 2}), "forecast_available")
