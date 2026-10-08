import unittest
from sberindex.external.mchs_archive import parse_capture, forecast_date, monthly_early


class ArchiveTests(unittest.TestCase):
    def capture(
        self,
        date,
        title="Прогноз на 2 апреля 2024 года",
        url="https://56.mchs.gov.ru/news/123",
    ):
        return (
            f"{title} ({url})\nsearch metadata\n\n{date}\n\n100\n\n# {title}\n\nТекст"
        )

    def test_current_publication_never_backdated_by_forecast_title(self):
        rows, excluded = parse_capture(self.capture("1 апреля 2026, 12:00"), "x")
        self.assertEqual(rows, [])
        self.assertEqual(excluded[0]["reason"], "publication_outside_history")

    def test_requires_header_date_not_later_body_date(self):
        s = "Title (https://56.mchs.gov.ru/news/123)\n# Title\n1 апреля 2024, 12:00"
        self.assertEqual(parse_capture(s, "x")[0], [])

    def test_rejects_unofficial_host(self):
        self.assertEqual(
            parse_capture(
                self.capture("1 апреля 2024, 12:00", url="https://evil.example/a"), "x"
            )[0],
            [],
        )

    def test_availability_and_validity_are_separate(self):
        rows, _ = parse_capture(self.capture("1 апреля 2024, 12:00"), "x")
        self.assertEqual(rows[0]["available_from_scenario"], "2024-04-02")
        self.assertEqual(rows[0]["forecast_date"], "2024-04-02")
        self.assertFalse(rows[0]["historical_vintage_verified"])

    def test_title_date_variants_and_invalid_dates(self):
        self.assertEqual(forecast_date("Прогноз на 26.12.2024"), "2024-12-26")
        self.assertIsNone(forecast_date("Прогноз на 31 февраля 2024 года"))
        self.assertIsNone(forecast_date("Прогноз за 2024 год"))

    def test_monthly_forecast_keeps_explicit_precision(self):
        rows, _ = parse_capture(
            self.capture("28 мая 2024, 17:22", title="Прогноз на июнь месяц 2024 года"),
            "x",
        )
        self.assertEqual(rows[0]["forecast_date"], "2024-06-01")
        self.assertEqual(rows[0]["forecast_time_precision"], "month")
        self.assertTrue(rows[0]["before_forecast_month_scenario"])

    def test_monthly_anticipation_is_strict_and_missing_is_unknown(self):
        self.assertTrue(monthly_early("2024-03-31", "2024-04-02"))
        self.assertFalse(monthly_early("2024-04-01", "2024-04-02"))
        self.assertIsNone(monthly_early("2024-03-31", None))
