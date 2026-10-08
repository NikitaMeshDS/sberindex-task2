import unittest

from sberindex.external.news_body_recovery import (
    article_excerpt,
    require_complete_outputs,
)


class BodyRecoveryTests(unittest.TestCase):
    def test_publication_date_cuts_navigation_and_footer(self):
        text = "Content type: text/html; Source: open({})\nL0: Все регионы: Орск Москва\nL217: 6 апреля 2024, 08:42\nL218: # Паводок\nL219: В Орске продолжается эвакуация.\nL220: Оцените материал\nL221: Контакты Москва"
        body = article_excerpt(text, "2024-04-06")
        self.assertIn("В Орске продолжается эвакуация.", body)
        self.assertNotIn("Все регионы", body)
        self.assertNotIn("Контакты Москва", body)

    def test_search_snippet_never_authorizes_body(self):
        with self.assertRaises(ValueError):
            article_excerpt(
                "Source: search; 6 апреля 2024 В Орске паводок", "2024-04-06"
            )

    def test_wrong_page_date_is_rejected(self):
        with self.assertRaises(ValueError):
            article_excerpt(
                "Content type: text/html; Source: open({})\nL217: 6 апреля 2026\nL218: Орск паводок",
                "2024-04-06",
            )

    def test_partial_or_duplicate_checkpoint_cannot_be_complete(self):
        inputs = [{"source_url": "a"}, {"source_url": "b"}]
        with self.assertRaises(ValueError):
            require_complete_outputs(inputs, [{"source_url": "a"}])
        with self.assertRaises(ValueError):
            require_complete_outputs(
                inputs, [{"source_url": "a"}, {"source_url": "a"}, {"source_url": "b"}]
            )
        require_complete_outputs(inputs, [{"source_url": "b"}, {"source_url": "a"}])
