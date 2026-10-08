"""Build and verify full and curated submission ZIPs from a clean Git snapshot."""

from pathlib import Path
import hashlib, json, subprocess, zipfile, shutil

r = Path(__file__).resolve().parents[1]
out = r.parent.parent / "outputs"
out.mkdir(exist_ok=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=r).strip()
sha = lambda b: hashlib.sha256(b).hexdigest()
assert not subprocess.check_output(
    ["git", "status", "--porcelain"], text=True, cwd=r
).strip(), "Commit final tracked evidence first"
names = sorted(
    subprocess.check_output(["git", "ls-files", "-z"], cwd=r)
    .decode()
    .strip("\0")
    .split("\0")
)
hashes = {n: sha((r / n).read_bytes()) for n in names}
meta = {
    "git_commit": head,
    "repository": "https://github.com/NikitaMeshDS/sberindex-task2",
    "repository_visibility": "public",
    "presentation": "complete: 21 slides PDF and editable PPTX",
    "submitted": False,
    "files_sha256": hashes,
}
path = out / "sberindex_task2_compact_project.zip"
with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for n in names:
        z.write(r / n, n)
    z.writestr(
        "PACKAGE_MANIFEST.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
with zipfile.ZipFile(path) as z:
    assert z.testzip() is None
    for n, h in hashes.items():
        assert sha(z.read(n)) == h, n
meta["archive_sha256"] = sha(path.read_bytes())
meta["archive_bytes"] = path.stat().st_size
meta["files_verified"] = len(hashes)
meta["status"] = "passed"
(out / "project_archive_manifest.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
)
selected = [
    "docs/JURY_START.md",
    "docs/research/SUBMISSION_REPORT.md",
    "docs/research/SOLUTION_IN_FIVE_SENTENCES.md",
    "docs/research/GLOSSARY.md",
    "docs/research/INFERENCE_AND_DELIVERY_CHECK.md",
    "docs/COMPETITION.md",
    "docs/GETTING_STARTED.md",
    "docs/CONFIGURATION.md",
    "docs/ARCHITECTURE.md",
    "docs/SUBMISSION.md",
    "requirements-lock.txt",
    "config.json",
    "data_sources.json",
    "artifacts/presentation.pdf",
    "artifacts/presentation.pptx",
    "artifacts/presentation_data.json",
    "artifacts/formulas/report_forecast.png",
]
selected += [str(p.relative_to(r)) for p in (r / "configs").glob("*.json")]
selected += [str(p.relative_to(r)) for p in (r / "docs/protocols").glob("*REVIEW*.md")]
keep = {
    "freeze_manifest.json",
    "evidence.json",
    "common_pairs.csv",
    "case_months.csv",
    "calibration.csv",
    "grid_check.csv",
    "may_peer_residual_ranks.csv",
    "REPORT.md",
    "summary.csv",
    "comparison.csv",
    "ablation_summary.csv",
    "monthly.csv",
    "ablation_monthly.csv",
    "coverage.csv",
    "audit.json",
    "protocol.json",
    "choice.json",
    "runtime.csv",
    "comparison.png",
    "comparison.svg",
    "cases.png",
    "cases.svg",
    "h3_dates.png",
    "h3_dates.svg",
    "verification.json",
    "graph_snapshot.json",
    "provenance.json",
    "selection.json",
    "late_summary.csv",
    "availability.png",
    "availability.svg",
    "conditional_summary.csv",
    "monitoring_burden.csv",
    "orsk_orenburg.csv",
    "reproduction.json",
    "sensitivity.png",
    "sensitivity.svg",
    "orsk_case.png",
    "orsk_case.svg",
    "case_yoy.csv",
    "news_lead_sensitivity.csv",
    "coverage.png",
    "coverage.svg",
    "warning_timing.csv",
    "cluster_intervals.csv",
    "temporal_tests.csv",
    "conditional_intervals.png",
    "conditional_intervals.svg",
    "full_pooled_summary.csv",
    "fullpanel_summary.csv",
    "FULLPANEL_REPORT.md",
    "fullpanel_coverage.csv",
    "final_audit.json",
    "fullpanel_audit.json",
    "fullpanel_comparison.png",
    "fullpanel_comparison.svg",
    "bolt_epoch1_training.json",
    "bolt_training.json",
    "structured_events.csv",
    "FORECAST_DIAGNOSTIC.md",
    "interface.jpg",
    "coverage_exclusions.csv",
    "batch_labels.csv",
    "full_pooled_comparison.png",
    "full_pooled_comparison.svg",
    "fixed_news_forecast_diagnostic.csv",
    "news_pairs.csv",
    "official_exposure_registry.csv",
    "municipal_case_months.csv",
    "llm_quote_exposure_registry.csv",
    "flood_summary.csv",
    "forecast_expense_alignment.csv",
}
for block in [
    "full_cohort_review",
    "regional_news_review",
    "event_metric_review",
    "foundation_covariate_review",
    "hierarchical_review",
    "direct_horizon_review",
    "spatial_review",
    "review_real_cases",
    "growth_bridge_review",
    "short_history_review",
    "news_alert_lead",
    "mchs_archive_review",
    "growth_sensitivity_review",
    "peer_residual_review",
    "presentation_comparison_cases",
    "presentation_claim_checks",
    "mchs_rss",
    "forecast_uncertainty_review",
    "category_growth_review",
    "foundation_expansion_review",
    "news_interval_review",
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
]:
    selected += [
        str(p.relative_to(r))
        for p in (r / "reports" / block).glob("*")
        if p.is_file() and p.name in keep
    ]
selected += [str(p.relative_to(r)) for p in (r / "dashboard").glob("*") if p.is_file()]
selected += ["reports/pages_deployment/audit.json", "reports/holdout_data_search/REPORT.md", "reports/holdout_data_search/availability_checks.json", "reports/linux_ci/latest_run.json", "reports/review_tests.json", "reports/expansion_verification.json"]
selected += ["reports/selected_model_holdout_20261007/predictions.csv"]
selected += ["configs/adaptive_growth_holdout.json", "docs/protocols/ADAPTIVE_GROWTH_HOLDOUT.md", "docs/protocols/NEWS_BODY_RECOVERY.md"]
selected += ["reports/adaptive_growth_holdout_20261007/" + n for n in ["predictions.csv", "growth_rates.csv", "diagnostic_2024_summary.csv", "change_from_fixed.csv"]]
selected += ["reports/news_body_recovery_20261007/quote_exposure_registry.csv"]
selected += ["artifacts/jury_guide.json"]
selected += [str(p.relative_to(r)) for p in (r / "docs/assets").glob("*") if p.is_file()]
selected = sorted(n for n in set(selected) if (r / n).is_file())
juryhashes = {n: sha((r / n).read_bytes()) for n in selected}
main_readme = (r / "README.md").read_text()
criteria_map = main_readme[main_readme.index("## Результаты и материалы"):main_readme.index("## Метод")]
readme = (
    "# Компактный комплект для жюри\n\n"
    + "[Презентация PDF](artifacts/presentation.pdf) · [Русский отчёт](docs/research/SUBMISSION_REPORT.md) · [Интерфейс](dashboard/index.html) · [Маршрут проверки](docs/JURY_START.md).\n\n"
    + "Этот архив служит для чтения. Код и быстрая проверка — в компактном проекте; полный исследовательский снимок — в GitHub Releases: https://github.com/NikitaMeshDS/sberindex-task2/releases/tag/research-snapshot-20261008.\n\n"
    + criteria_map
    + "Git SHA: " + head + "\n\n"
    + "Конфигурации находятся в configs/. Ссылки на детали, отсутствующие здесь, открывайте в полном архиве с той же структурой. Публичный сайт: https://NikitaMeshDS.github.io/sberindex-task2/; локальный интерфейс также работает без сервера.\n"
)
jury = out / "sberindex_task2_jury_reading.zip"
with zipfile.ZipFile(jury, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for n in selected:
        z.write(r / n, n)
    z.writestr("README.md", readme)
    z.writestr(
        "PACKAGE_MANIFEST.json",
        json.dumps(
            {
                "git_commit": head,
                "purpose": "curated reading; source code and complete evidence in full archive",
                "files_sha256": juryhashes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
with zipfile.ZipFile(jury) as z:
    assert z.testzip() is None
    for n, h in juryhashes.items():
        assert sha(z.read(n)) == h, n
(out / "jury_archive_manifest.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "git_commit": head,
            "archive_sha256": sha(jury.read_bytes()),
            "archive_bytes": jury.stat().st_size,
            "files_verified": len(juryhashes),
            "files_sha256": juryhashes,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
for src, dst in [
    ("docs/research/SUBMISSION_REPORT.md", "СберИндекс_задача2_краткий_отчёт.md"),
    ("docs/research/METHODOLOGY.md", "СберИндекс_задача2_методология.md"),
    ("reports/full_cohort_review/REPORT.md", "СберИндекс_все_муниципалитеты.md"),
    ("artifacts/presentation.pdf", "СберИндекс_задача2_презентация.pdf"),
    ("artifacts/presentation.pptx", "СберИндекс_задача2_презентация.pptx"),
]:
    shutil.copy2(r / src, out / dst)
print(
    json.dumps(
        {
            "git_commit": head,
            "full_files": len(hashes),
            "full_MB": round(path.stat().st_size / 1024**2, 2),
            "jury_files": len(juryhashes),
            "jury_MB": round(jury.stat().st_size / 1024**2, 2),
            "all_sha_crc_verified": True,
        },
        ensure_ascii=False,
    )
)
