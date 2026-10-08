"""Paired timing audit for synthetic municipal spending changes.

An alarm is attributed to the injection only when it occurs in an altered
series at a month where the same original series had no alarm. This does not
identify real events or estimate a real-world false-positive rate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sberindex.detection.change_detection import CATEGORIES, panel_for, residuals, score

from sberindex.paths import ROOT
METHODS = ("spike", "rolling_3m", "ewma", "cusum")
PROTOCOL = {
    "calibration": "January-June 2024; 99th percentile of per-MO maxima",
    "evaluation": "July-December 2024; score state continues from calibration",
    "treated_fraction": .25,
    "seeds": list(range(20261001, 20261011)),
    "shifts_pct": [-20, 20],
    "shapes": ["step", "pulse", "ramp"],
    "first_nonzero_month": {"step": 0, "pulse": 0, "ramp": 1},
    "pairing": "For each MO/month, changed alarm AND NOT original alarm at that same MO/month",
    "limits": "Retrospective synthetic sensitivity on the reused 2024 panel; control alarms can reflect real changes. Ten assignments are not independent time histories or confidence intervals.",
}


def factors(shape, shift):
    if shape == "step":
        return np.full(6, 1 + shift)
    if shape == "pulse":
        return np.array([1 + shift, 1., 1., 1., 1., 1.])
    if shape == "ramp":
        return 1 + np.linspace(0, shift, 6)
    raise ValueError(shape)


def paired_metrics(original_alarm, changed_alarm, treated, onset):
    """Return rates with denominator all assigned MO, including missed shocks."""
    assert original_alarm.shape == changed_alarm.shape
    assert original_alarm.shape[1] == 6 and 0 <= onset < 6
    assert treated.any() and (~treated).any()
    new = changed_alarm & ~original_alarm
    target = new[treated]
    first = np.where(target[:, onset:].any(axis=1),
                     target[:, onset:].argmax(axis=1), np.nan)
    detected = first[np.isfinite(first)]
    return {
        "new_at_onset": float(target[:, onset].mean()),
        "new_by_second_active_month": float(target[:, onset:onset+2].any(axis=1).mean()),
        "new_by_third_active_month": float(target[:, onset:onset+3].any(axis=1).mean()),
        "new_any_after_onset": float(target[:, onset:].any(axis=1).mean()),
        "new_later_without_onset": float((~target[:, onset] & target[:, onset+1:].any(axis=1)).mean()),
        "median_delay_if_new": float(np.median(detected)) if len(detected) else np.nan,
        "original_any_treated": float(original_alarm[treated].any(axis=1).mean()),
        "changed_any_treated": float(changed_alarm[treated].any(axis=1).mean()),
        "original_any_control": float(original_alarm[~treated].any(axis=1).mean()),
        "changed_any_control": float(changed_alarm[~treated].any(axis=1).mean()),
        "new_any_control": float(new[~treated].any(axis=1).mean()),
    }


def main():
    rows = []
    max_prechange_difference = 0.
    for category in CATEGORIES:
        _, values = panel_for(category)
        n = len(values)
        baseline_signal = residuals(values, "level")
        frozen = {}
        for method in METHODS:
            baseline_score = score(baseline_signal, method)
            threshold = float(np.quantile(baseline_score[:, :6].max(axis=1), .99))
            frozen[method] = (threshold, baseline_score[:, 6:] > threshold)
        for seed in PROTOCOL["seeds"]:
            treated = np.zeros(n, dtype=bool)
            order = np.random.default_rng(seed).permutation(n)
            treated[order[:round(n*PROTOCOL["treated_fraction"])]] = True
            for shift_pct in PROTOCOL["shifts_pct"]:
                for shape in PROTOCOL["shapes"]:
                    altered = values.copy()
                    altered[treated, 18:] *= factors(shape, shift_pct/100)
                    changed_signal = residuals(altered, "level")
                    max_prechange_difference = max(max_prechange_difference,
                        float(np.max(np.abs(changed_signal[:, :6]-baseline_signal[:, :6]))))
                    onset = PROTOCOL["first_nonzero_month"][shape]
                    for method, (threshold, original_alarm) in frozen.items():
                        changed_alarm = score(changed_signal, method)[:, 6:] > threshold
                        rows.append({"category": category, "seed": seed,
                                     "shift_pct": shift_pct, "shape": shape,
                                     "method": method, "onset_index": onset,
                                     "treated_n": int(treated.sum()),
                                     "control_n": int((~treated).sum()),
                                     "threshold": threshold,
                                     **paired_metrics(original_alarm, changed_alarm, treated, onset)})
    assert max_prechange_difference < 1e-12, max_prechange_difference
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/event_attribution_runs.csv", index=False)
    columns = ["new_at_onset", "new_by_second_active_month", "new_by_third_active_month",
               "new_any_after_onset", "new_later_without_onset", "median_delay_if_new",
               "original_any_treated", "changed_any_treated", "original_any_control",
               "changed_any_control", "new_any_control"]
    summary = frame.groupby(["category", "shift_pct", "shape", "method"])[columns].mean().reset_index()
    summary.to_csv(ROOT / "results/event_attribution_summary.csv", index=False)
    audit = {"protocol": PROTOCOL, "rows": len(frame),
             "max_prechange_signal_difference": max_prechange_difference,
             "interpretation": "New alarms are paired monthwise against unaltered data; later-only alarms for a pulse are not onset detections."}
    (ROOT / "results/event_attribution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(summary[(summary.category == "Все категории") & (summary.shift_pct == 20)].to_string(index=False))


if __name__ == "__main__":
    main()
