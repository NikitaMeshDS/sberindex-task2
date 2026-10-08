"""Stress-test frozen change detectors under three synthetic shock shapes.

The experimental panel and threshold are identical to change_detection.py.
The shape of each injected shock is specified before inspecting post-June data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sberindex.detection.change_detection import CATEGORIES, panel_for, residuals, score


from sberindex.paths import ROOT
METHODS = ("spike", "rolling_3m", "ewma", "cusum")
SHAPES = {
    "permanent_20pct": np.full(6, 1.20),
    "ramp_0_to_20pct": np.linspace(1.0, 1.20, 6),
    "one_month_20pct": np.array([1.20, 1.0, 1.0, 1.0, 1.0, 1.0]),
}


def main():
    rows = []
    for category in CATEGORIES:
        _, values = panel_for(category)
        n = len(values)
        treated = np.random.default_rng(20260930).choice(n, n // 4, replace=False)
        controls = np.ones(n, bool); controls[treated] = False
        original = residuals(values, "level")
        for method in METHODS:
            threshold = np.quantile(score(original, method)[:, :6].max(axis=1), .99)
            for shape, multipliers in SHAPES.items():
                altered = values.copy()
                altered[treated, 18:24] *= multipliers
                alarms = score(residuals(altered, "level"), method)[:, 6:] > threshold
                first = alarms[treated].argmax(axis=1)
                any_treated = alarms[treated].any(axis=1)
                first = first[any_treated]
                tpr = any_treated.mean()
                fpr = alarms[controls].any(axis=1).mean()
                precision = (len(treated) * tpr / (len(treated) * tpr + controls.sum() * fpr)
                             if tpr + fpr else np.nan)
                rows.append({"category": category, "shock_shape": shape,
                             "method": method, "threshold": threshold,
                             "detection_rate": tpr, "false_alarm_rate": fpr,
                             "precision_at_25pct_prevalence": precision,
                             "detected_by_first_month": alarms[treated, 0].mean(),
                             "detected_by_third_month": alarms[treated, :3].any(axis=1).mean(),
                             "median_detection_delay_months": np.median(first) if len(first) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/change_robustness.csv", index=False)
    print(result.loc[result.category == "Все категории"].to_string(index=False))


if __name__ == "__main__":
    main()
