"""Publication allowlist and local-link boundaries."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pages", ROOT / "tools/prepare_pages_site.py"
)
pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pages)


class PagesStagingTests(unittest.TestCase):
    def test_unreviewed_file_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "private-notes.txt").write_text("not an approved website asset")
            with self.assertRaises(ValueError):
                pages.main(output)

    def test_report_links_resolve_to_public_project_and_cannot_escape(self):
        source = ROOT / "docs/research/SUBMISSION_REPORT.md"
        self.assertEqual(
            pages.public_link("../../reports/example/REPORT.md", source),
            pages.REPOSITORY + "reports/example/REPORT.md",
        )
        self.assertEqual(
            pages.public_link("https://rosstat.gov.ru", source),
            "https://rosstat.gov.ru",
        )
        with self.assertRaises(ValueError):
            pages.public_link("../../../../private.txt", source)
