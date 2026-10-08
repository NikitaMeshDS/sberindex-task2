"""Compare a detached full refit with a frozen Git baseline, byte for byte."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def compare_directory(baseline, full, pattern):
    left = {p.name: p for p in baseline.glob(pattern) if p.is_file()}
    right = {p.name: p for p in full.glob(pattern) if p.is_file()}
    missing = sorted(left.keys() - right.keys())
    extras = sorted(right.keys() - left.keys())
    different = sorted(name for name in left.keys() & right.keys()
                       if digest(left[name]) != digest(right[name]))
    return {"baseline_count": len(left), "full_count": len(right),
            "byte_identical": len(left) - len(missing) - len(different),
            "missing": missing, "extra": extras, "different": different}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/full_retrain_report.json"))
    args = parser.parse_args()
    baseline, full = args.baseline.resolve(), args.full.resolve()
    metadata_path = full / "reports/run_metadata.json"
    if not metadata_path.exists():
        metadata_path = full / "run_metadata.json"  # frozen pre-reorganization runs
    metadata = json.loads(metadata_path.read_text())
    assert metadata["mode"] == "full"
    failures = [step["script"] for step in metadata["steps"] if step["exit_code"]]
    assert not failures, failures
    results = compare_directory(baseline / "results", full / "results", "*")
    png = compare_directory(baseline / "figures", full / "figures", "*.png")
    svg = compare_directory(baseline / "figures", full / "figures", "*.svg")
    assert not results["missing"] and not results["different"], results
    assert not png["missing"] and not png["different"], png
    report = {
        "baseline_git_commit": revision(baseline),
        "full_retrain_git_commit": revision(full),
        "started_at_utc": metadata["started_at"],
        "mode": metadata["mode"],
        "step_count": len(metadata["steps"]),
        "step_failures": failures,
        "total_step_seconds": round(sum(step["seconds"] for step in metadata["steps"]), 2),
        "results": results,
        "png": png,
        "svg": svg,
        "limits": "Same Mac and cached pinned model weights; no new target months or another OS. SVG at the frozen baseline has volatile Matplotlib metadata and IDs.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"steps": report["step_count"],
                      "results_identical": results["byte_identical"],
                      "png_identical": png["byte_identical"],
                      "svg_identical": svg["byte_identical"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
