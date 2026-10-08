"""Safety checks for the optional research snapshot downloader."""

import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "snapshot",
    Path(__file__).resolve().parents[1] / "tools/download_research_snapshot.py",
)
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


class SnapshotTests(unittest.TestCase):
    def test_checksum_rejects_modified_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            archive.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                snapshot.unpack(archive, Path(directory) / "output")
            self.assertFalse((Path(directory) / "output").exists())

    def test_path_traversal_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "unsafe")
            with patch.object(
                snapshot, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
            ), self.assertRaises(ValueError):
                snapshot.unpack(archive, Path(directory) / "output")
            self.assertFalse((Path(directory) / "output").exists())

    def test_valid_archive_unpacks_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "good.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data/input.txt", "saved")
            with patch.object(
                snapshot, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
            ):
                destination = Path(directory) / "output"
                snapshot.unpack(archive, destination)
                self.assertEqual((destination / "data/input.txt").read_text(), "saved")
                with self.assertRaises(ValueError):
                    snapshot.unpack(archive, destination)
