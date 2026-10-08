"""Build the current editable deck and export its PDF in the supplied runtime.

Scientific reproduction does not require the optional presentation runtime.
See docs/PRESENTATION_BUILD.md for environment variables.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from sberindex.paths import ROOT


def main():
    required = (
        "PRESENTATION_SKILL_DIR",
        "ARTIFACT_TOOL_ROOT",
        "RUNTIME_PYTHON",
        "RUNTIME_BIN_DIR",
    )
    missing = [name for name in required if not os.environ.get(name)]
    node = os.environ.get("RUNTIME_NODE") or shutil.which("node")
    if missing or not node:
        raise RuntimeError(
            "Presentation rebuild requires the supplied Codex runtime. "
            "See docs/PRESENTATION_BUILD.md. Existing PDF/PPTX remain available. "
            f"Missing: {missing}, Node available: {bool(node)}"
        )
    binary = Path(os.environ["RUNTIME_BIN_DIR"]) / "soffice"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    subprocess.run(
        [sys.executable, str(ROOT / "tools/prepare_presentation_data.py")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [node, str(ROOT / "tools/build_final_presentation.mjs")], cwd=ROOT, check=True
    )
    destination = ROOT / ".presentation-build/pdf"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "presentation.pdf").unlink(missing_ok=True)
    subprocess.run(
        [
            str(binary),
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(destination),
            str(ROOT / "artifacts/presentation.pptx"),
        ],
        cwd=ROOT,
        check=True,
    )
    shutil.copyfile(
        destination / "presentation.pdf", ROOT / "artifacts/presentation.pdf"
    )


if __name__ == "__main__":
    main()
