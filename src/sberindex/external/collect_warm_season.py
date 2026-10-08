"""Download missing frozen warm-season source PDFs, or verify existing copies.

Run explicitly; the ordinary refresh pipeline remains offline. A changed remote
PDF is rejected and requires a separately documented source vintage.
"""

import argparse
import hashlib
import json
from urllib.request import Request, urlopen

from sberindex.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "data/external/warm_season/protocol.json").read_text())
    for source in protocol["sources"]:
        path = ROOT / source["path"]
        if not path.exists():
            if args.check_only:
                raise FileNotFoundError(path)
            request = Request(source["source_url"], headers={"User-Agent": "SberIndex research source collector"})
            with urlopen(request, timeout=120) as response:
                content = response.read()
            if not content.startswith(b"%PDF"):
                raise ValueError(f"Source did not return a PDF: {source['source_url']}")
            if hashlib.sha256(content).hexdigest() != source["sha256"]:
                raise ValueError(f"Remote PDF differs from the registered vintage: {source['source_url']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Existing source differs from the registered vintage: {path}")
        print(f"Verified {source['bulletin']}: {source['path']}")


if __name__ == "__main__":
    main()
