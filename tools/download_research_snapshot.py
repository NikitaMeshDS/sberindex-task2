"""Download and safely unpack the immutable full research snapshot separately."""

import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/NikitaMeshDS/sberindex-task2/releases/download/research-snapshot-20261008-clean/sberindex_task2_research_clean.zip"
SHA256 = "70ef850099decb5ab263c92d1e32cb84636672190af65f3ddfc0436106d16df5"
ROOT = Path(__file__).resolve().parents[1]


def unpack(archive, destination):
    """Validate the entire archive before creating the destination."""
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("Research archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains a symbolic link")
        if bundle.testzip() is not None:
            raise ValueError("Corrupt research archive")
        bundle.extractall(destination)


def main():
    destination = ROOT / "research_workspace"
    if destination.exists():
        raise SystemExit(
            "research_workspace already exists; use that copy or move it first"
        )
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "research.zip"
        print("Downloading full research snapshot (241 MB)...", flush=True)
        urllib.request.urlretrieve(URL, archive)
        unpack(archive, destination)
    print("SHA256 checked. Full snapshot: research_workspace/")
    print("cd research_workspace && python3.12 run_review.py --mode verify")
    print("Before recompute, run git init inside research_workspace (isolates new provenance).")


if __name__ == "__main__":
    main()
