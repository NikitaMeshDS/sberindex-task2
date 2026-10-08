import unittest

from tools.clean_report_prose import clean_report_prose, count_prose_adjacencies


class ReportProseCleanerTests(unittest.TestCase):
    def test_digit_boundaries_preserve_numbers_and_horizon_identifiers(self):
        self.assertEqual(clean_report_prose('в2023не h1пары 251/252МО 1,25руб SHA256графа'),
                         'в 2023 не h1 пары 251/252 МО 1,25 руб SHA256 графа')

    def test_code_links_urls_identifiers_and_math_are_verbatim(self):
        protected = ('`МО2` ``МО2``\n```python\nМО2 = 10\n```\n'
                     '~~~sh\nМО2=10\n~~~\n'
                     '[в2023](https://example.ru/МО2_(тест3)) '
                     'https://example.ru/МО2 $МО2+1$ \\(МО2+1\\) '
                     'код_МО2\n[ref]: https://example.ru/МО2 "МО2"\n')
        expected = protected.replace('[в2023]', '[в 2023]')
        self.assertEqual(clean_report_prose(protected), expected)
        self.assertEqual(count_prose_adjacencies(clean_report_prose(protected)), 0)

    def test_idempotence(self):
        text = '12фактам2023:2075МО наh1=2165,56 против979,14уseasonal_pooled'
        cleaned = clean_report_prose(text)
        self.assertIn('979,14 уseasonal_pooled', cleaned)
        self.assertEqual(count_prose_adjacencies(cleaned), 0)
        self.assertEqual(clean_report_prose(cleaned), cleaned)

    def test_optional_protected_fstring_expression(self):
        text = '{lookup["МО2"]}МО'
        self.assertEqual(clean_report_prose(text, ((0, text.index('}') + 1),)), text)
