"""Check presentation claims using saved results only; no fitting or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    "reports/growth_sensitivity_review/predictions.parquet",
    "reports/growth_sensitivity_review/summary.csv",
    "reports/growth_sensitivity_review/prophet_comparators.csv",
    "reports/peer_residual_review/residual_panel.parquet",
    "reports/peer_residual_review/case_yoy.csv",
    "reports/peer_residual_review/detector_panel.parquet",
    "reports/short_history_review/comparison.csv",
    "reports/short_history_review/series_metrics.csv",
    "reports/short_history_review/choice.json",
    "configs/short_history_review.json",
    "src/sberindex/detection/short_history_review.py",
    "reports/event_metric_review/choice.json",
    "src/sberindex/detection/peer_residual_review.py",
    "src/sberindex/forecasting/growth_sensitivity_review.py",
]
INPUTS += ["tools/presentation_claim_checks.py", "tests/test_presentation_claim_checks.py"]
OFFLINE = {"pelt", "kernelcpd"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def date_balanced_mae(frame, rate):
    """Use a frozen g=0 frame; k=0 or1 gives affine predictions."""
    if not np.isfinite(rate) or rate <= -1:
        raise ValueError("rate must be finite and > -1")
    residual = frame.actual - frame.predicted * (1 + rate) ** frame.year_bridges
    return float(residual.abs().groupby(frame.target).mean().mean())


def continuous_growth_check(base, comparators, lower=0.07, upper=0.17):
    """Convex MAE bounded above by maximum endpoint on the whole interval."""
    if lower >= upper:
        raise ValueError("lower must be below upper")
    if not set(base.year_bridges.unique()).issubset({0, 1}):
        raise ValueError("Convex affine proof requires k in {0,1}")
    if base.duplicated(["territory_id", "origin", "target", "horizon"]).any():
        raise ValueError("duplicate saved forecast pairs")
    rows = []
    for horizon, group in base.groupby("horizon"):
        candidates = comparators[comparators.horizon == horizon]
        best = candidates.loc[candidates.MAE.idxmin()]
        left, right = date_balanced_mae(group, lower), date_balanced_mae(group, upper)
        bound = max(left, right)
        rows.append(
            dict(
                horizon=int(horizon),
                lower_rate=lower,
                upper_rate=upper,
                lower_MAE=left,
                upper_MAE=right,
                continuous_upper_bound_MAE=bound,
                best_tested_prophet=str(best.model),
                best_tested_prophet_MAE=float(best.MAE),
                strict_margin_rubles=float(best.MAE - bound),
                all_rates_strictly_better=bool(bound < best.MAE),
                dates=int(group.target.nunique()),
                pairs=len(group),
            )
        )
    return rows


def rank_panel(panel):
    finite = panel[np.isfinite(panel.peer_residual_log)].copy()
    complete = finite.groupby("territory_id").target.nunique()
    ids = complete[complete == 12].index
    frames = []
    for name, subset in [
        ("may_available", finite[finite.target == "2024-05"]),
        (
            "complete_2024_peer_residual",
            finite[(finite.target == "2024-05") & finite.territory_id.isin(ids)],
        ),
    ]:
        frame = subset[["territory_id", "peer_residual_log"]].copy()
        frame["population"] = name
        frame["population_n"] = len(frame)
        frame["signed_desc_rank"] = frame.peer_residual_log.rank(
            ascending=False, method="min"
        ).astype(int)
        frame["absolute_desc_rank"] = (
            frame.peer_residual_log.abs()
            .rank(ascending=False, method="min")
            .astype(int)
        )
        frame["signed_top_fraction_pct"] = 100 * frame.signed_desc_rank / len(frame)
        frame["absolute_top_fraction_pct"] = 100 * frame.absolute_desc_rank / len(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def detector_checks(summary, series):
    """Reconstruct onset counts, null FAR and conditional delay from saved rows."""
    rows = []
    for split in ["selection", "evaluation"]:
        selected = summary[(summary.split == split) & (summary["shape"] == "all")]
        for item in selected.itertuples():
            group = series[(series.split == split) & (series.method == item.method)]
            tp, fp, fn = [
                int(group[c].sum()) for c in ["onset_tp", "onset_fp", "onset_fn"]
            ]
            onset_f1 = 2 * tp / (2 * tp + fp + fn)
            onset_recall = tp / (tp + fn)
            np.testing.assert_allclose(
                [onset_f1, onset_recall], [item.onset_f1, item.onset_recall], atol=1e-12
            )
            delays = [d for values in group.onset_delay for d in json.loads(values)]
            null = group[group["shape"] == "no_change"]
            # The saved profile has 12 evaluation months per series.
            null_months = len(null) * 12
            null_alarms = int(null.null_false_alarms.sum())
            null_far = null_alarms / null_months * 100
            np.testing.assert_allclose(
                null_far, item.null_false_alarms_per100_mo_month, atol=1e-12
            )
            offline = item.method in OFFLINE
            if offline:
                if delays or not np.isnan(item.mean_onset_delay_months):
                    raise ValueError("Offline onset delay must remain unavailable")
            else:
                if len(delays) != tp:
                    raise ValueError(
                        "Online onset-delay denominator differs from matched onsets"
                    )
                np.testing.assert_allclose(
                    np.mean(delays), item.mean_onset_delay_months, atol=1e-12
                )
            rows.append(
                dict(
                    split=split,
                    method=item.method,
                    mode="offline" if offline else "online",
                    boundary_f1=float(item.f1),
                    onset_f1=onset_f1,
                    onset_recall=onset_recall,
                    onset_tp=tp,
                    onset_fp=fp,
                    onset_fn=fn,
                    true_onsets=tp + fn,
                    null_false_alarm_count=null_alarms,
                    null_mo_months=null_months,
                    null_FAR_per100_mo_months=null_far,
                    mean_onset_delay_months=None if offline else float(np.mean(delays)),
                    onset_delay_denominator=len(delays),
                    missed_onsets_excluded_from_delay=fn,
                    mean_report_availability_delay_months=float(
                        item.mean_report_availability_delay_months
                    ),
                )
            )
    if len(rows) != 16 or len({r["method"] for r in rows}) != 8:
        raise ValueError("Expected all eight saved detectors in both splits")
    return rows


def run(root=ROOT, out=None):
    root = Path(root)
    out = Path(out) if out else root / "reports/presentation_claim_checks"
    before = {p: sha(root / p) for p in INPUTS}
    growth = pd.read_parquet(root / INPUTS[0])
    base = growth[growth.scenario == "g0"].copy()
    if len(base) != 60700:
        raise ValueError("Expected 60700 saved common pairs")
    comparators = pd.read_csv(root / INPUTS[2])
    continuous = continuous_growth_check(base, comparators)
    grid = []
    for rate in np.linspace(0.07, 0.17, 11):
        for horizon, group in base.groupby("horizon"):
            grid.append(
                dict(
                    rate=float(rate),
                    horizon=int(horizon),
                    MAE=date_balanced_mae(group, rate),
                )
            )
    # Verify saved transformations at every source scenario, without fitting.
    for (scenario, horizon), group in growth.groupby(["scenario", "horizon"]):
        rate = float(group.growth_rate.iloc[0])
        group0 = base[base.horizon == horizon]
        np.testing.assert_allclose(
            date_balanced_mae(group0, rate),
            group.groupby("target").ae.mean().mean(),
            atol=1e-8,
        )
    panel = pd.read_parquet(root / INPUTS[3])
    ranks = rank_panel(panel)
    case = panel[(panel.territory_id == 1673) & (panel.target == "2024-05")].iloc[0]
    # Confirm stored residual identities using the source log(actual/predicted).
    np.testing.assert_allclose(
        np.log(case.actual / case.predicted), case.own_residual_log, atol=1e-12
    )
    np.testing.assert_allclose(
        case.own_residual_log - case.peer_median_log, case.peer_residual_log, atol=1e-12
    )
    yoy = pd.read_csv(root / INPUTS[4])
    chosen = yoy[yoy.target.isin(["2024-05", "2024-06", "2024-07"])]
    detector_panel = pd.read_parquet(root / INPUTS[5])
    city_alerts = detector_panel[detector_panel.territory_id.isin([1665, 1673])]
    orsk = dict(
        territory_id=1673,
        target="2024-05",
        residual_log=float(case.peer_residual_log),
        residual_log_percent=float(100 * case.peer_residual_log),
        residual_ordinary_percent=float(100 * np.expm1(case.peer_residual_log)),
        reference="Own log(actual/prediction) residual minus contemporaneous leave-one-out regional peer median; regional code 56, 38 peers. Positive values mean excess over peers.",
        ranks=ranks[ranks.territory_id == 1673].to_dict("records"),
        claimed_rank39_of1998_verified=False,
        may_july_orsk_minus_orenburg_mean_pp=float(
            chosen.orsk_minus_orenburg_pp.mean()
        ),
        may_july_gaps_pp=chosen.orsk_minus_orenburg_pp.tolist(),
        alerts_both_cities_all_frozen_methods=int(city_alerts.active_alert.sum()),
        title_warning="Approximately +7 pp is the selected May–July Orsk–Orenburg YoY gap; +7.9 log-percent is the May regional peer residual. Do not conflate them.",
        caveats="Orenburg also flooded; selected window and retrospective residuals do not establish a causal flood or payments effect.",
    )
    detectors = detector_checks(
        pd.read_csv(root / INPUTS[6]), pd.read_csv(root / INPUTS[7])
    )
    choice = json.loads((root / INPUTS[8]).read_text())
    cfg = json.loads((root / INPUTS[9]).read_text())
    if cfg["evaluation_months"] != 12:
        raise ValueError(
            "Saved short-history denominator requires 12 evaluation months"
        )
    evidence = dict(
        schema_version=1,
        independent_validation=False,
        growth=dict(
            interval_percent=[7, 17],
            continuous_verified=all(r["all_rates_strictly_better"] for r in continuous),
            proof="With k in {0,1}, each prediction is affine in g. Absolute error is convex; positive date-balanced averaging preserves convexity. For every g in [0.07,0.17], MAE(g) <= max(MAE(0.07),MAE(0.17)). Each endpoint maximum is strictly below the minimum MAE among three saved Prophet variants for that horizon.",
            horizons=continuous,
            rate_selected=None,
            model_refitted=False,
            warning="Retrospective diagnostic on existing 2024 data; h12 has only one origin and one target date. No significance or transfer guarantee.",
        ),
        orsk=orsk,
        detectors=dict(
            profile="Saved short_history_review: 12 training and 12 evaluation months, simulated null/step/pulse.",
            selected_online_method=choice["selected_online_method"],
            selection_rule=choice["objective"],
            selection_fixed_before_evaluation=choice["fixed_before_evaluation"],
            rows=detectors,
            delay_definition="Only ±1-month matched first onsets; max(0,predicted-onset). Missed onsets excluded. Boundary F1 also counts pulse return; onset F1 neutralizes predictions matching return. Offline methods receive full sequence and have no online onset-delay estimate.",
            FAR_definition="False alarms on synthetic no-change series / no-change MO-months ×100. Real municipal alert burden is not FAR; real ground truth unavailable.",
            EWMA_role="Best eligible online onset F1 on saved short-history selection; not globally best boundary F1, not fastest online, and not independently validated on real shifts.",
            other_profile="Saved event_metric_review is a different 36+24-month boundary profile and selects no online method; do not combine its metrics with short-history selection.",
        ),
    )
    if before != {p: sha(root / p) for p in INPUTS}:
        raise ValueError("Source artifacts changed while reading")
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(grid).to_csv(out / "grid_check.csv", index=False)
    ranks.to_csv(out / "may_peer_residual_ranks.csv", index=False)
    lines = [
        "# Presentation claim checks",
        "",
        "Saved results only; no new model, threshold or detector experiment. These are retrospective checks, not independent validation.",
        "",
        "## Continuous growth claim",
        "",
        evidence["growth"]["proof"],
        "",
        "| h | MAE at 7% | MAE at 17% | best tested Prophet | MAE bound | strict margin |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for r in continuous:
        lines.append(
            f"| {r['horizon']} | {r['lower_MAE']:.2f} | {r['upper_MAE']:.2f} | {r['best_tested_prophet']} | {r['continuous_upper_bound_MAE']:.2f} | {r['strict_margin_rubles']:.2f} |"
        )
    lines += [
        "",
        evidence["growth"]["warning"],
        "",
        "## Orsk",
        "",
        f"May residual: {orsk['residual_log']:.9f} log units = {orsk['residual_log_percent']:.4f} log-percent; exponentiation gives {orsk['residual_ordinary_percent']:.4f}% excess in the multiplicative residual ratio. The upstream calculation uses log(actual/prediction), so exponentiation expresses the actual/prediction ratio relative to the median regional peer residual; it is not a YoY percentage-point gap.",
        "",
        "| Population | N | signed descending rank | absolute descending rank | signed top% | absolute top% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in orsk["ranks"]:
        lines.append(
            f"| {r['population']} | {r['population_n']} | {r['signed_desc_rank']} | {r['absolute_desc_rank']} | {r['signed_top_fraction_pct']:.3f} | {r['absolute_top_fraction_pct']:.3f} |"
        )
    lines += [
        "",
        "Rank ties use minimum competition rank. 1998 denotes municipalities with finite peer residuals in all 12 months of 2024. 2012 denotes available May observations. Rank 39/1998 is not reproduced. A top 2% description holds for signed positive ranks above, but not for absolute ranks.",
        "",
        orsk["title_warning"],
        "",
        f"The chosen May–July gaps are {orsk['may_july_gaps_pp']}; mean {orsk['may_july_orsk_minus_orenburg_mean_pp']:.4f} pp. Neither city triggers any of the four frozen own/peer detectors in 2024.",
        "",
        orsk["caveats"],
        "",
        "## All eight saved short-history detectors",
        "",
        "| Split | Method | Mode | boundary F1 | onset F1 | onset recall | null FAR/100 | onset delay | matched denominator | misses excluded |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in detectors:
        delay = (
            "unavailable"
            if r["mean_onset_delay_months"] is None
            else f"{r['mean_onset_delay_months']:.4f}"
        )
        lines.append(
            f"| {r['split']} | {r['method']} | {r['mode']} | {r['boundary_f1']:.4f} | {r['onset_f1']:.4f} | {r['onset_recall']:.4f} | {r['null_FAR_per100_mo_months']:.4f} | {delay} | {r['onset_delay_denominator']} | {r['missed_onsets_excluded_from_delay']} |"
        )
    lines += [
        "",
        evidence["detectors"]["EWMA_role"],
        "",
        evidence["detectors"]["delay_definition"],
        "",
        evidence["detectors"]["FAR_definition"],
        "",
        evidence["detectors"]["other_profile"],
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    outputs = [
        "evidence.json",
        "grid_check.csv",
        "may_peer_residual_ranks.csv",
        "REPORT.md",
    ]
    audit = dict(
        status="passed",
        source_reports_unmodified=True,
        independent_validation=False,
        model_or_threshold_changes=False,
        new_detector_experiment=False,
        input_sha256=before,
        source_code_sha256=sha(Path(__file__)),
        output_sha256={p: sha(out / p) for p in outputs},
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(out=args.output)
    print(
        json.dumps(
            {
                "continuous_growth_verified": result["growth"]["continuous_verified"],
                "orsk_rank39_verified": result["orsk"][
                    "claimed_rank39_of1998_verified"
                ],
                "detector_rows": len(result["detectors"]["rows"]),
            }
        )
    )
