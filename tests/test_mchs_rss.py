import unittest
from sberindex.external.mchs_rss import parse_feed, discover_feeds


class RssTests(unittest.TestCase):
    def test_historical_dates_are_explicit_and_current_items_not_backdated(self):
        xml = b"""<rss><channel><item><title>Flood</title><link>https://56.mchs.gov.ru/one</link><pubDate>Sat, 06 Apr 2024 10:00:00 +0300</pubDate></item><item><title>2023 in title</title><link>https://56.mchs.gov.ru/two</link><pubDate>Tue, 06 Oct 2026 10:00:00 +0300</pubDate></item><item><title>Missing</title><link>https://56.mchs.gov.ru/three</link></item></channel></rss>"""
        records = parse_feed(
            xml, "https://56.mchs.gov.ru/news/rss", "2023-01-01", "2024-12-31"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["published_date"], "2024-04-06")
        self.assertFalse(records[0]["historical_vintage_verified"])

    def test_feed_discovery_rejects_other_hosts(self):
        html = '<a href="/news/rss">RSS подписка</a><link type="application/rss+xml" href="https://evil.example/feed">'
        self.assertEqual(
            discover_feeds(html, "https://56.mchs.gov.ru/news"),
            ["https://56.mchs.gov.ru/news/rss"],
        )

    def test_atom_feed_parses_and_deduplicates(self):
        xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>One</title><link href="https://56.mchs.gov.ru/one"/><published>2024-04-06T10:00:00+03:00</published></entry><entry><title>Again</title><link href="https://56.mchs.gov.ru/one"/><published>2024-04-06T10:00:00+03:00</published></entry></feed>"""
        self.assertEqual(
            len(
                parse_feed(
                    xml, "https://56.mchs.gov.ru/rss", "2023-01-01", "2024-12-31"
                )
            ),
            1,
        )
