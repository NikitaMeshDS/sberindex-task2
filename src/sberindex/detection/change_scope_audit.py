"""Paired sensitivity audit of existing, frozen local change detectors.

No threshold or model selection is performed. Unaltered real data are not
labelled negatives: 'control alarm' is not a measured real-world false positive.
Repetitions vary synthetic assignments, not independent time histories.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sberindex.detection.change_detection import CATEGORIES, panel_for, residuals, score

from sberindex.paths import ROOT
METHODS = ("spike", "rolling_3m", "ewma", "cusum")
PROTOCOL = {
    "coverage": [0.10, 0.25, 0.50, 0.75, 1.0],
    "shift_pct": [-20, 20],
    "seeds": list(range(20261001, 20261011)),
    "start": "2024-07", "calibration_end": "2024-06",
    "quantile": 0.99, "detrend": "level",
    "status": "Retrospective stress audit; no untouched holdout or threshold tuning",
}


def main():
    rows = []
    max_common_shift_error = 0.0
    max_future_perturbation_error = 0.0
    for category in CATEGORIES:
        _, values = panel_for(category)
        n = len(values)
        z = residuals(values, "level")
        # Changing Aug-Dec cannot change July's post-calibration residual/score.
        future = values.copy()
        future[:n // 2, 19:] *= 1.5
        future_z = residuals(future, "level")
        frozen = {}
        for method in METHODS:
            scores = score(z, method)
            threshold = np.quantile(scores[:, :6].max(axis=1), PROTOCOL["quantile"])
            baseline = (scores[:, 6:] > threshold).any(axis=1)
            frozen[method] = threshold, baseline
            err = np.max(np.abs(score(future_z, method)[:, 6] - scores[:, 6]))
            max_future_perturbation_error = max(max_future_perturbation_error, float(err))
        for seed in PROTOCOL["seeds"]:
            order = np.random.default_rng(seed).permutation(n)
            for coverage in PROTOCOL["coverage"]:
                treated = np.zeros(n, dtype=bool)
                treated[order[:round(n * coverage)]] = True
                control = ~treated
                for shift in PROTOCOL["shift_pct"]:
                    altered = values.copy()
                    altered[treated, 18:] *= 1 + shift / 100
                    changed_z = residuals(altered, "level")
                    if coverage == 1:
                        max_common_shift_error = max(max_common_shift_error,
                                                     float(np.abs(changed_z - z).max()))
                    for method, (threshold, baseline) in frozen.items():
                        alarms = score(changed_z, method)[:, 6:] > threshold
                        detected = alarms.any(axis=1)
                        previously_quiet = treated & ~baseline
                        newly_detected = treated & detected & ~baseline
                        first = alarms[treated & detected].argmax(axis=1)
                        rows.append({
                            "category": category, "seed": seed, "coverage": coverage,
                            "shift_pct": shift, "method": method, "threshold": threshold,
                            "treated_n": int(treated.sum()), "control_n": int(control.sum()),
                            "treated_alarm_rate": detected[treated].mean(),
                            "treated_baseline_alarm_rate": baseline[treated].mean(),
                            "new_alarm_rate_all_treated": newly_detected.sum() / treated.sum(),
                            "new_alarm_rate_previously_quiet": (
                                newly_detected.sum() / previously_quiet.sum()
                                if previously_quiet.any() else np.nan),
                            "suppressed_alarm_rate": (treated & baseline & ~detected).sum() / treated.sum(),
                            "control_alarm_rate": detected[control].mean() if control.any() else np.nan,
                            "control_baseline_alarm_rate": baseline[control].mean() if control.any() else np.nan,
                            "median_delay_detected": float(np.median(first)) if len(first) else np.nan,
                        })
    assert max_common_shift_error < 1e-12, max_common_shift_error
    assert max_future_perturbation_error < 1e-12, max_future_perturbation_error
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/change_scope_repetitions.csv", index=False)
    metrics = ["treated_alarm_rate", "treated_baseline_alarm_rate", "new_alarm_rate_all_treated",
               "new_alarm_rate_previously_quiet", "suppressed_alarm_rate", "control_alarm_rate",
               "control_baseline_alarm_rate", "median_delay_detected"]
    summary = result.groupby(["category", "coverage", "shift_pct", "method"])[metrics].agg(["mean", "min", "max"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.to_csv(ROOT / "results/change_scope_summary.csv")
    audit = {"protocol": PROTOCOL, "rows": len(result),
             "common_shift_residual_max_error": max_common_shift_error,
             "future_perturbation_july_score_max_error": max_future_perturbation_error,
             "limits": "Min/max span synthetic assignments, not confidence intervals. Natural changes are unlabelled."}
    (ROOT / "results/change_scope_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(summary.loc[("Все категории", slice(None), 20, "rolling_3m"),
                      ["treated_alarm_rate_mean", "new_alarm_rate_all_treated_mean", "control_alarm_rate_mean"]])


if __name__ == "__main__":
    main()
