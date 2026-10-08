"""One entry point: refresh saved evidence, or refit all forecasting models."""

from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
sys.path.insert(0, str(SOURCE))


def module_index():
    """Resolve a stage name to its package without depending on the cwd."""
    files = sorted((SOURCE / "sberindex").glob("*/*.py"))
    modules = {
        path.name: f"sberindex.{path.parent.name}.{path.stem}"
        for path in files
        if path.name != "__init__.py"
    }
    if len(modules) != len([p for p in files if p.name != "__init__.py"]):
        raise ValueError("Duplicate pipeline stage names")
    return modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["refresh", "full"], default="refresh")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Explicitly rebuild the optional presentation",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Also recompute all fixed review extensions, including full-cohort Prophet and foundation inference",
    )
    args = parser.parse_args()
    from sberindex.input_integrity import check, register

    check(ROOT, refresh=args.mode == "refresh")
    modules = module_index()
    steps = []
    if args.mode == "full":
        steps += [
            "benchmark_rolling.py",
            "chronos2_rolling.py",
            "hgb_12m.py",
            "asof_cohort_forecast.py",
            "asof_foundation_forecast.py",
            "category_benchmark.py",
            "change_detection.py",
            "change_robustness.py",
            "real_alerts.py",
            "news_ablation.py",
        ]
    steps += [
        "news_alignment.py",
        "news_corpus_audit.py",
        "external_data.py",
        "news_registry.py",
        "weather_news_audit.py",
        "weather_body_audit.py",
        "seasonal_news_alignment.py",
        "interpretation.py",
        "category_full.py",
        "real_event_case.py",
        "summarize_results.py",
        "forecast_audit.py",
        "survivorship_audit.py",
        "asof_hgb_12m.py",
        "asof_cohort_summary.py",
        "asof_regional_forecast.py",
        "asof_month_robustness.py",
        "news_pair_eligibility.py",
        "asof_reporting_delay.py",
        "asof_regional_delay.py",
        "regional_seasonality.py",
        "regional_blend.py",
        "robust_anchor.py",
        "adaptive_forecast.py",
        "online_bias_correction.py",
        "deseasonal_hgb.py",
        "operational_audit.py",
        "online_intervals.py",
    ]
    if args.mode == "full":
        steps.append("geographic_extension.py")
    steps += [
        "calendar_audit.py",
        "foundation_seasonal_audit.py",
        "asof_distribution_audit.py",
        "foundation_delay_audit.py",
        "operational_early_audit.py",
        "regional_news_experiment.py",
        "change_scope_audit.py",
        "event_attribution_audit.py",
        "asof_detector_audit.py",
        "detector_delay_audit.py",
        "bocpd_audit.py",
        "real_event_registry.py",
        "hierarchical_changes.py",
        "forecast_residual_changes.py",
        "shift_confirmation.py",
        "build_figures.py",
        "research_hypothesis_figures.py",
        "verify_artifacts.py",
    ]
    steps += [
        "tools/detector_tradeoffs.py",
        "tools/residual_feature_experiment.py",
        "tools/plot_residual_features.py",
        "tools/operational_workflow.py",
        "tools/plot_operational_workflow.py",
    ]
    if args.review:
        steps.append("run_review.py")
    if args.presentation:
        steps.append("build_presentation.py")
    log_dir = ROOT / "run_logs"
    log_dir.mkdir(exist_ok=True)
    metadata = {
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "packages": {},
        "steps": [],
    }
    code_files = [
        ROOT / "run_pipeline.py",
        *sorted(SOURCE.rglob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
    ]
    metadata["code_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code_files
    }
    metadata["config_sha256"] = hashlib.sha256(
        (ROOT / "config.json").read_bytes()
    ).hexdigest()
    metadata["source_manifest_sha256"] = hashlib.sha256(
        (ROOT / "data_sources.json").read_bytes()
    ).hexdigest()
    for name in (
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "prophet",
        "chronos-forecasting",
        "torch",
        "matplotlib",
        "openpyxl",
        "reportlab",
    ):
        metadata["packages"][name] = importlib.metadata.version(name)
    for script in steps:
        start = time.perf_counter()
        print(f"Running {script}", flush=True)
        module = modules.get(script)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
        log_name = script.replace("/", "_") + ".log"
        with (log_dir / log_name).open("w") as log:
            command = (
                [sys.executable, "-m", module]
                if module
                else [sys.executable, str(ROOT / script)]
            )
            if script == "run_review.py":
                command += ["--mode", "recompute"]
            if (
                script
                in {
                    "foundation_seasonal_audit.py",
                    "foundation_delay_audit.py",
                    "operational_early_audit.py",
                }
                and args.mode == "refresh"
            ):
                command.append("--refresh")
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            if result.returncode == 0 and script in {
                "tools/detector_tradeoffs.py",
                "tools/residual_feature_experiment.py",
            }:
                result = subprocess.run(
                    command + ["--verify"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        metadata["steps"].append(
            {
                "script": script,
                "module": module,
                "seconds": round(time.perf_counter() - start, 3),
                "exit_code": result.returncode,
            }
        )
        (ROOT / "reports/run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        if result.returncode:
            raise RuntimeError(f"{script} failed; read {log_dir / log_name}")
    if args.mode == "full":
        register(
            ROOT,
            provenance={
                "mode": "full",
                "started_at": metadata["started_at"],
                "code_sha256": metadata["code_sha256"],
            },
        )
    print("All pipeline steps completed successfully", flush=True)


if __name__ == "__main__":
    main()
