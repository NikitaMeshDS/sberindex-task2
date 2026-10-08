"""Controlled synthetic level-shift experiment on the 2024 municipal panel.

We use category-specific year-over-year log growth, subtract the current
cross-sectional median, and calibrate thresholds using January-June only.
No real structural-change labels are present in the contest archive.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


from sberindex.paths import ROOT
CONFIG = json.loads((ROOT / "config.json").read_text())
DATA = ROOT / "data" / "consumption.parquet"
CATEGORIES = ["Все категории", "Здоровье", "Маркетплейсы",
              "Общественное питание", "Продовольствие", "Транспорт"]


def panel_for(category):
    data = pd.read_parquet(DATA)
    panel = (data.loc[data.category == category]
             .pivot(index="territory_id", columns="date", values="value")
             .dropna().sort_index())
    return panel.index.to_numpy(), panel.to_numpy(float)


def residuals(values, detrend):
    growth = np.log(values[:, 12:24] / values[:, :12])
    national = np.median(growth, axis=0, keepdims=True)
    centered = growth - national
    if detrend == "level":
        baseline = np.median(centered[:, :6], axis=1, keepdims=True)
        return centered - baseline
    if detrend == "linear":
        t = np.arange(12)
        slope = ((centered[:, :6] - centered[:, :6].mean(axis=1, keepdims=True))
                 @ (t[:6] - t[:6].mean()) / ((t[:6] - t[:6].mean()) ** 2).sum())
        intercept = centered[:, :6].mean(axis=1) - slope * t[:6].mean()
        return centered - intercept[:, None] - slope[:, None] * t
    raise ValueError(detrend)


def score(z, method):
    if method == "spike":
        return np.abs(z)
    if method == "ewma":
        state = np.zeros(z.shape[0]); out = np.zeros_like(z)
        for t in range(z.shape[1]):
            state = 0.45 * z[:, t] + 0.55 * state
            out[:, t] = np.abs(state)
        return out
    if method == "cusum":
        pos = np.zeros(z.shape[0]); neg = np.zeros(z.shape[0]); out = np.zeros_like(z)
        for t in range(z.shape[1]):
            pos = np.maximum(0, pos + z[:, t] - 0.025)
            neg = np.maximum(0, neg - z[:, t] - 0.025)
            out[:, t] = np.maximum(pos, neg)
        return out
    if method == "rolling_3m":
        out = np.zeros_like(z)
        for t in range(z.shape[1]):
            out[:, t] = np.abs(z[:, max(0, t - 2):t + 1].mean(axis=1))
        return out
    raise ValueError(method)


def main():
    rng = np.random.default_rng(CONFIG["random_seed"])
    start = (pd.Period(CONFIG["synthetic_shift_start_month"], freq="M")
             - pd.Period("2023-01", freq="M")).n
    pre_months = start - 12
    assert 2 <= pre_months <= 10
    rows = []
    for category in CATEGORIES:
        ids, values = panel_for(category)
        n = len(ids)
        treated = rng.choice(n, n // 4, replace=False)
        control = np.ones(n, bool); control[treated] = False
        for detrend in ("level", "linear"):
            original = residuals(values, detrend)
            for shift in (s / 100 for s in CONFIG["shift_sizes_pct"]):
                altered = values.copy()
                altered[treated, start:24] *= 1 + shift
                changed = residuals(altered, detrend)
                for method in ("spike", "ewma", "cusum", "rolling_3m"):
                    baseline_score = score(original, method)
                    for quantile in CONFIG["alarm_calibration_quantiles"]:
                        threshold = np.quantile(baseline_score[:, :pre_months].max(axis=1), quantile)
                        detected = score(changed, method) > threshold
                        alarms = detected[:, pre_months:12]
                        tpr = alarms[treated].any(axis=1).mean()
                        fpr = alarms[control].any(axis=1).mean()
                        first = np.where(alarms[treated].any(axis=1),
                                         alarms[treated].argmax(axis=1), np.nan)
                        rows.append({"category": category, "detrend": detrend,
                                     "shift_pct": int(shift * 100), "method": method,
                                     "calibration_quantile": quantile,
                                     "threshold": threshold, "treated": len(treated),
                                     "control": control.sum(), "detection_rate": tpr,
                                     "false_alarm_rate": fpr,
                                     "median_delay_months": np.nanmedian(first)})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results" / "change_detection_metrics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
