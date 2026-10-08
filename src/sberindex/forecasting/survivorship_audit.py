"""Audit the future-completeness restriction in the forecast sample.

The 2023-complete cohort is known before 2024. Forecast comparisons still
use the original 256 IDs and saved Prophet predictions, so this is a
profile sensitivity study, not a corrected independent benchmark.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sberindex.paths import ROOT
OUT = ROOT / "results"


def main():
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (raw.loc[raw.category == "Все категории"]
             .pivot(index="territory_id", columns="date", values="value")
             .sort_index())
    train = panel.loc[:, panel.columns.str.startswith("2023")]
    test = panel.loc[:, panel.columns.str.startswith("2024")]
    known = train.notna().all(axis=1)
    balanced = known & test.notna().all(axis=1)
    assert len(train.columns) == len(test.columns) == 12
    assert (int(known.sum()), int(balanced.sum())) == (2075, 2016)

    known_profile = train.loc[known].sum().to_numpy(float)
    balanced_profile = train.loc[balanced].sum().to_numpy(float)
    coverage = pd.DataFrame({
        "month": panel.columns,
        "observed_2023_cohort": panel.loc[known].notna().sum().to_numpy(int),
        "observed_balanced": panel.loc[balanced].notna().sum().to_numpy(int),
        "observed_all": panel.notna().sum().to_numpy(int),
    })
    coverage.to_csv(OUT / "survivorship_coverage.csv", index=False)

    saved = pd.read_parquet(OUT / "rolling_predictions.parquet")
    ids = np.sort(saved.loc[saved.model == "prophet", "territory_id"].unique())
    assert len(ids) == 256 and balanced.loc[ids].all()
    saved = saved[saved.model.isin(["prophet", "seasonal_pooled"])
                  & saved.territory_id.isin(ids)]
    wide = saved.pivot(index=["territory_id", "origin", "horizon"],
                       columns="model", values=["actual", "predicted"])
    actual = wide[("actual", "prophet")].to_numpy(float)
    np.testing.assert_allclose(actual, wide[("actual", "seasonal_pooled")])
    index = wide.index
    origins = index.get_level_values("origin")
    horizons = index.get_level_values("horizon").to_numpy(int)
    targets = (pd.PeriodIndex(origins, freq="M") + horizons).astype(str)
    origin_month = pd.PeriodIndex(origins, freq="M").month.to_numpy() - 1
    target_month = pd.PeriodIndex(targets, freq="M").month.to_numpy() - 1
    anchor = np.array([panel.loc[i, origin] for i, origin in
                       zip(index.get_level_values("territory_id"), origins)])
    original = anchor * balanced_profile[target_month] / balanced_profile[origin_month]
    asof = anchor * known_profile[target_month] / known_profile[origin_month]
    np.testing.assert_allclose(original,
                               wide[("predicted", "seasonal_pooled")], atol=1e-7)
    prophet = wide[("predicted", "prophet")].to_numpy(float)
    early = (horizons == 1) & (origins >= "2024-01") & (targets <= "2024-06")
    weights = [0, .25, .5, .75, 1]
    validation = {str(weight): float(np.abs(actual[early] -
                  (weight * asof[early] + (1-weight) * prophet[early])).mean())
                  for weight in weights}
    selected = min(weights, key=lambda weight: validation[str(weight)])

    rows = []
    for horizon in [1, 3, 6]:
        use = (horizons == horizon) & (origins >= "2024-06")
        candidates = {
            "seasonal_original": original[use],
            "seasonal_asof": asof[use],
            "blend_original": .75*original[use] + .25*prophet[use],
            "blend_asof_fixed75": .75*asof[use] + .25*prophet[use],
            "blend_asof_reselected": selected*asof[use] + (1-selected)*prophet[use],
        }
        for name, prediction in candidates.items():
            rows.append({"horizon": horizon, "model": name,
                         "observations": int(use.sum()),
                         "MAE": float(np.abs(actual[use]-prediction).mean())})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "survivorship_forecast_sensitivity.csv", index=False)

    excluded = panel.loc[known & ~balanced]
    partial = []
    for horizon in [1, 3, 6]:
        errors = []
        for origin_pos in range(17, 23):
            target_pos = origin_pos + horizon
            if target_pos >= 24:
                continue
            anchor_value = excluded.iloc[:, origin_pos]
            target_value = excluded.iloc[:, target_pos]
            available = anchor_value.notna() & target_value.notna()
            prediction = (anchor_value.loc[available] *
                          known_profile[target_pos % 12] / known_profile[origin_pos % 12])
            errors.extend(np.abs(target_value.loc[available]-prediction).to_list())
        partial.append({"horizon": horizon, "observed_pairs": len(errors),
                        "seasonal_asof_MAE": float(np.mean(errors)) if errors else np.nan})
    pd.DataFrame(partial).to_csv(OUT / "survivorship_partial_ids.csv", index=False)

    ratios_known = known_profile[:, None] / known_profile[None, :]
    ratios_balanced = balanced_profile[:, None] / balanced_profile[None, :]
    report = {
        "complete_2023_ids": int(known.sum()),
        "complete_2023_and_2024_ids": int(balanced.sum()),
        "excluded_by_future_completeness": int((known & ~balanced).sum()),
        "absent_all_2024_among_2023_complete": int((known & ~test.notna().any(axis=1)).sum()),
        "sample_256_all_balanced": bool(balanced.loc[ids].all()),
        "validation_MAE_asof_profile": validation,
        "selected_asof_weight": selected,
        "max_relative_seasonal_ratio_change_pct": float(
            100 * np.max(np.abs(ratios_known/ratios_balanced-1))),
        "limits": ("Saved 256 Prophet forecasts and their future-complete sample remain fixed. "
                   "Partial-ID MAE uses observed origin-target pairs only; missing outcomes "
                   "cannot be scored. Neither analysis is a new independent time test."),
    }
    (OUT / "survivorship_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
