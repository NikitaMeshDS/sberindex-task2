"""Verify frozen review evidence or explicitly recompute the separate extensions."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reports/review_evidence_manifest.json"
STAGES = [
    "sberindex.forecasting.prophet_seasonality",
    "sberindex.forecasting.full_cohort_review",
    "sberindex.external.regional_news_review",
    "sberindex.detection.event_metric_review",
    "sberindex.forecasting.foundation_covariate_review",
    "sberindex.forecasting.hierarchical_review",
    "sberindex.forecasting.direct_horizon_review",
    "sberindex.forecasting.spatial_review",
    "sberindex.reporting.review_real_cases",
    "sberindex.forecasting.growth_bridge_review",
    "sberindex.detection.short_history_review",
    "sberindex.reporting.news_alert_lead",
    "sberindex.reporting.final_review_figures",
    "sberindex.external.mchs_archive",
    "sberindex.forecasting.growth_sensitivity_review",
    "sberindex.detection.peer_residual_review",
    "tools.presentation_comparison_cases",
    "tools.presentation_claim_checks",
    "sberindex.forecasting.forecast_uncertainty_review",
    "sberindex.forecasting.category_growth_review",
    ("sberindex.forecasting.foundation_expansion_review", "--report"),
    "sberindex.forecasting.foundation_expansion_reporting",
    "sberindex.external.news_event_extraction_review",
    "sberindex.external.news_forecast_diagnostic_review",
    "sberindex.external.news_interval_review",
    "sberindex.detection.event_conditioned_review",
    "sberindex.detection.yoy_shift_review",
    "sberindex.external.news_event_field_review",
    "sberindex.external.news_event_expanded_review",
    "tools.build_research_dashboard",
]
REPORTS = [
    "prophet_seasonality",
    "full_cohort_review",
    "regional_news_review",
    "event_metric_review",
    "foundation_covariate_review",
    "hierarchical_review",
    "prophet_backend_review",
    "direct_horizon_review",
    "spatial_review",
    "review_real_cases",
    "code_style_review",
    "growth_bridge_review",
    "short_history_review",
    "news_alert_lead",
    "mchs_rss",
    "mchs_archive_review",
    "growth_sensitivity_review",
    "peer_residual_review",
    "prose_cleanup_review",
    "prose_cleanup_followup",
    "presentation_comparison_cases",
    "presentation_claim_checks",
    "forecast_uncertainty_review",
    "category_growth_review",
    "foundation_expansion_review",
    "news_event_extraction_review",
    "news_event_extraction_review/field_validation_variant",
    "news_event_extraction_review/expanded_batch",
    "event_conditioned_review",
    "yoy_shift_review",
    "research_dashboard",
    "selected_model_holdout_20261007",
    "marketplace_growth_review",
    "adaptive_growth_holdout_20261007",
    "news_body_recovery_20261007",
    "news_interval_review",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(manifest, root=ROOT):
    checked = 0
    for name, digest in manifest["sha256"].items():
        path = root / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Missing or changed frozen review input/output: {name}")
        checked += 1
    return checked


REQUIRED_OUTPUTS = {
    "short_history_review": [
        "comparison.csv",
        "series_metrics.csv",
        "choice.json",
        "audit.json",
        "REPORT.md",
        "comparison.png",
        "comparison.svg",
    ],
    "news_alert_lead": [
        "claim_lead_times.csv",
        "real_alarms.csv",
        "coverage.csv",
        "audit.json",
        "REPORT.md",
        "availability.png",
        "availability.svg",
    ],
    "mchs_rss": ["REPORT.md"],
    "growth_sensitivity_review": [
        "REPORT.md",
        "summary.csv",
        "audit.json",
        "predictions.parquet",
        "sensitivity.png",
        "sensitivity.svg",
    ],
    "peer_residual_review": [
        "REPORT.md",
        "audit.json",
        "residual_panel.parquet",
        "detector_panel.parquet",
        "monitoring_burden.csv",
        "case_yoy.csv",
        "news_lead_sensitivity.csv",
        "orsk_case.png",
        "orsk_case.svg",
    ],
    "prose_cleanup_review": ["audit.json"],
    "prose_cleanup_followup": ["audit.json"],
    "presentation_comparison_cases": [
        "REPORT.md",
        "audit.json",
        "evidence.json",
        "comparison.csv",
        "case_months.csv",
        "calibration.csv",
        "common_pairs.csv",
    ],
    "presentation_claim_checks": [
        "REPORT.md",
        "audit.json",
        "evidence.json",
        "grid_check.csv",
        "may_peer_residual_ranks.csv",
    ],
    "mchs_archive_review": [
        "REPORT.md",
        "audit.json",
        "coverage.csv",
        "forecast_expense_alignment.csv",
        "warning_timing.csv",
        "exclusions.csv",
        "coverage.png",
        "coverage.svg",
    ],
    "growth_bridge_review": [
        "predictions.parquet",
        "summary.csv",
        "audit.json",
        "REPORT.md",
    ],
    "spatial_review": [
        "predictions.parquet",
        "summary.csv",
        "protocol.json",
        "REPORT.md",
    ],
    "review_real_cases": [
        "observed_series.csv",
        "coverage.csv",
        "audit.json",
        "REPORT.md",
        "cases.png",
        "cases.svg",
    ],
    "prophet_seasonality": ["predictions.parquet", "summary.csv", "audit.json"],
    "full_cohort_review": [
        "predictions.parquet",
        "summary.csv",
        "coverage.csv",
        "audit.json",
    ],
    "regional_news_review": [
        "predictions.parquet",
        "ablation_summary.csv",
        "asof_features.csv",
        "provenance.json",
        "REPORT.md",
    ],
    "event_metric_review": [
        "comparison.csv",
        "series_metrics.csv",
        "protocol.json",
        "choice.json",
        "REPORT.md",
    ],
    "foundation_covariate_review": [
        "predictions.parquet",
        "summary.csv",
        "coverage.csv",
        "audit.json",
        "REPORT.md",
    ],
    "hierarchical_review": [
        "predictions.parquet",
        "summary.csv",
        "protocol.json",
        "REPORT.md",
    ],
    "prophet_backend_review": ["regression.json"],
    "direct_horizon_review": [
        "predictions.parquet",
        "summary.csv",
        "protocol.json",
        "REPORT.md",
    ],
}


REQUIRED_OUTPUTS.update(
    {
        "news_event_extraction_review/expanded_batch": [
            "REPORT.md",
            "audit.json",
            "structured_events.csv",
            "monitoring_burden.csv",
            "capture_manifest.json",
            "coverage_exclusions.csv",
        ],
        "forecast_uncertainty_review": [
            "REPORT.md",
            "audit.json",
            "cluster_intervals.csv",
            "temporal_tests.csv",
            "conditional_intervals.png",
        ],
        "category_growth_review": [
            "REPORT.md",
            "audit.json",
            "full_bridge_predictions.parquet",
            "full_pooled_predictions.parquet",
            "full_pooled_summary.csv",
        ],
        "foundation_expansion_review": [
            "REPORT.md",
            "FULLPANEL_REPORT.md",
            "final_audit.json",
            "summary.csv",
            "fullpanel_summary.csv",
            "fullpanel_matched_predictions.parquet",
        ],
        "news_interval_review": [
            "REPORT.md",
            "audit.json",
            "summary.csv",
            "news_pairs.csv",
            "interval_panel.parquet",
        ],
        "news_event_extraction_review": [
            "REPORT.md",
            "audit.json",
            "structured_events.csv",
            "fixed_news_forecast_diagnostic.csv",
        ],
        "news_event_extraction_review/field_validation_variant": [
            "REPORT.md",
            "audit.json",
            "structured_events.csv",
            "monitoring_burden.csv",
        ],
        "event_conditioned_review": [
            "REPORT.md",
            "audit.json",
            "monitoring_burden.csv",
            "official_exposure_registry.csv",
        ],
        "yoy_shift_review": [
            "REPORT.md",
            "audit.json",
            "monitoring_burden.csv",
            "cases.csv",
        ],
        "research_dashboard": ["REPORT.md", "audit.json"],
    }
)


REQUIRED_OUTPUTS.update(
    {
        "adaptive_growth_holdout_20261007": [
            "REPORT.md",
            "freeze_manifest.json",
            "predictions.csv",
            "growth_rates.csv",
            "diagnostic_2024_summary.csv",
            "change_from_fixed.csv",
        ],
        "news_body_recovery_20261007": [
            "REPORT.md",
            "audit.json",
            "coverage.csv",
            "structured_events.csv",
            "quote_exposure_registry.csv",
            "monitoring_burden.csv",
            "detector_panel.parquet",
        ],
    }
)


def require_complete_blocks(root=ROOT, contracts=None):
    contracts = REQUIRED_OUTPUTS if contracts is None else contracts
    for block, names in contracts.items():
        for name in names:
            path = root / "reports" / block / name
            if not path.is_file():
                raise ValueError(f"Incomplete review block: {block}/{name}")
            if name in {"audit.json", "regression.json"}:
                status = json.loads(path.read_text()).get("status")
                if status not in {"passed", "complete"}:
                    raise ValueError(
                        f"Incomplete review audit: {block}/{name}: {status}"
                    )


def declared_files(node, root=ROOT):
    """Validate and include local dependencies declared by per-block provenance."""
    found = set()
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            path = root / node["path"]
            if not path.is_file() or sha(path) != node["sha256"]:
                raise ValueError(f"Changed declared provenance file: {node['path']}")
            found.add(path)
        for key, value in node.items():
            if key in {"input_sha256", "source_sha256", "code_sha256"} and isinstance(
                value, dict
            ):
                for name, digest in value.items():
                    path = root / name
                    if not path.is_file() or sha(path) != digest:
                        raise ValueError(f"Changed declared source: {name}")
                    found.add(path)
            else:
                found |= declared_files(value, root)
    elif isinstance(node, list):
        for value in node:
            found |= declared_files(value, root)
    return found


def record():
    require_complete_blocks()
    paths = [
        ROOT / "run_review.py",
        ROOT / "run_pipeline.py",
        ROOT / "requirements-lock.txt",
        ROOT / "reports/final_review_reproduction.json",
        ROOT / "reports/presentation_revision_reproduction.json",
    ]
    paths += list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "tools").glob("*.py"))
    paths += list((ROOT / "tools").glob("*.mjs"))
    paths += [
        ROOT / "artifacts" / name
        for name in ("presentation.pdf", "presentation.pptx", "presentation_data.json")
    ]
    paths += list((ROOT / "artifacts/formulas").glob("*.png"))
    paths += [ROOT / "artifacts/jury_guide.json"]
    paths += [p for p in (ROOT / "docs/assets").glob("*") if p.is_file()]
    paths += list((ROOT / "reports/reviewer_final_packaging").glob("*"))
    paths += list((ROOT / "tests").glob("*.py"))
    paths += list((ROOT / "configs").glob("*.json"))
    paths += list((ROOT / "docs").rglob("*.md"))
    paths += [ROOT / "README.md", ROOT / "reports/expansion_verification.json"]
    paths += [p for p in (ROOT / "dashboard").glob("*") if p.is_file()]
    paths += list(ROOT.glob("requirements-*.txt"))
    paths += list((ROOT / ".github/workflows").glob("*.yml"))
    paths += list((ROOT / "reports/holdout_data_search").glob("*"))
    paths += list((ROOT / "reports/pages_deployment").glob("*"))
    paths += [
        ROOT / "data/consumption.parquet",
        ROOT / "results/asof_cohort_protocol.json",
        ROOT / "results/municipal_lookup.csv",
        ROOT / "reports/operational_workflow/delayed_predictions.parquet",
    ]
    paths += [
        p
        for p in (ROOT / "data/external/regional_news_review").rglob("*")
        if p.is_file()
    ]
    for name in REPORTS:
        directory = ROOT / "reports" / name
        if not directory.is_dir():
            raise ValueError(f"Review block not yet complete: {name}")
        paths += [p for p in directory.iterdir() if p.is_file()]
    # Capture full external-source tree: registry builders consume earlier official
    # snapshots transitively. Keep them protected even when a block's short audit
    # lists only its own new corpus. No absent-vintage claim is created by hashing.
    paths += [p for p in (ROOT / "data/external").rglob("*") if p.is_file()]
    for path in list(paths):
        if path.suffix == ".json" and (
            path.parent in {ROOT / "reports" / name for name in REPORTS}
            or path.name == "capture_manifest.json"
            and "regional_news_review" in str(path)
        ):
            paths += list(declared_files(json.loads(path.read_text())))
    # Local vendor installations and trained checkpoints are reproducible caches,
    # not distributed evidence. Their revisions and weight hashes remain in provenance.
    candidates = sorted(set(paths))
    relative = [str(p.relative_to(ROOT)) for p in candidates]
    ignored = subprocess.run(
        ["git", "check-ignore", "-z", "--stdin"],
        cwd=ROOT,
        input="\0".join(relative) + "\0",
        text=True,
        capture_output=True,
        check=False,
    )
    if ignored.returncode not in {0, 1}:
        raise ValueError(f"Cannot inspect ignored local caches: {ignored.stderr}")
    ignored_names = set(ignored.stdout.strip("\0").split("\0"))
    paths = [p for p, name in zip(candidates, relative) if name not in ignored_names]
    manifest = {
        "status": "frozen_verified_snapshot",
        "independent_temporal_holdout": False,
        "verification_scope": "SHA256 of saved evidence, code, configs and declared sources; not full retraining",
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
    }
    verify(manifest)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["verify", "recompute", "record"], default="verify"
    )
    args = parser.parse_args()
    if args.mode != "verify" and (ROOT / "compact_distribution.json").exists():
        raise SystemExit("Full research data are in the immutable release archive. Run python tools/download_research_snapshot.py, then cd research_workspace before recompute/record.")
    if args.mode == "recompute":
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        for stage in STAGES:
            print(f"Recomputing {stage}", flush=True)
            arguments = [stage] if isinstance(stage, str) else list(stage)
            subprocess.run(
                [sys.executable, "-m", *arguments], cwd=ROOT, env=env, check=True
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sberindex.forecasting.foundation_covariate_review_report",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "tools/verify_prophet_backend.py")],
            cwd=ROOT,
            env=env,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "tools/plot_review_results.py")],
            cwd=ROOT,
            env=env,
            check=True,
        )
        # Recomputing changes runtime/provenance. Recording is an explicit new snapshot,
        # never a claim that the old numerical reference was reproduced unchanged.
        manifest = record()
    elif args.mode == "record":
        manifest = record()
    else:
        manifest = json.loads(MANIFEST.read_text())
    checked = verify(manifest)
    report = {
        "status": "passed",
        "files_checked": checked,
        "mode": args.mode,
        "verification_scope": manifest.get("verification_scope", "Saved evidence"),
        "meaning": "Saved file integrity only; scientific validity and retraining are separate checks",
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "review_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
