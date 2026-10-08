import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "review_runner", Path(__file__).resolve().parents[1] / "run_review.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FrozenReviewTests(unittest.TestCase):
    def test_changed_or_missing_frozen_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.csv"
            source.write_text("past-only\n")
            manifest = {"sha256": {"source.csv": m.sha(source)}}
            self.assertEqual(m.verify(manifest, root), 1)
            source.write_text("future inserted\n")
            with self.assertRaises(ValueError):
                m.verify(manifest, root)
            source.unlink()
            with self.assertRaises(ValueError):
                m.verify(manifest, root)


class CompletionAndDependenciesTests(unittest.TestCase):
    def test_partial_block_cannot_be_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "reports/full_cohort_review"
            directory.mkdir(parents=True)
            (directory / "coverage.csv").write_text("only partial progress\n")
            with self.assertRaises(ValueError):
                m.require_complete_blocks(
                    root, {"full_cohort_review": ["coverage.csv", "audit.json"]}
                )

    def test_declared_transitive_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.csv"
            source.write_text("dated source\n")
            audit = {"input_sha256": {"source.csv": m.sha(source)}}
            self.assertEqual(m.declared_files(audit, root), {source})
            source.write_text("changed future vintage\n")
            with self.assertRaises(ValueError):
                m.declared_files(audit, root)
