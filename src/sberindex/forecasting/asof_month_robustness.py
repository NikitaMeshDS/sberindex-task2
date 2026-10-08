"""Audit forecast conclusions across target months without changing model selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from sberindex.paths import ROOT

OUT = ROOT / "results"
MODELS = ("seasonal_pooled", "blend_selected", "prophet", "chronos_2")


def main():
    scored = pd.read_parquet(OUT / "asof_cohort_scored.parquet")
    keys = ["territory_id", "origin", "target", "horizon"]
    assert not scored.duplicated(keys + ["model"]).any()
    monthly_rows = []
    sensitivity_rows = []
    for horizon in (1, 3, 6, 12):
        part = scored[scored.horizon == horizon]
        wide = part.pivot(index=keys, columns="model", values="absolute_error")
        assert wide.notna().all().all(), horizon
        monthly = wide.groupby(level="target").mean()
        for target, row in monthly.iterrows():
            for model in wide.columns:
                monthly_rows.append({"horizon": horizon, "target": target,
                                     "model": model, "municipalities": int((wide.index.get_level_values("target") == target).sum()),
                                     "MAE": float(row[model])})
        # The 6/12-month horizons have one target date, so no date robustness estimate exists.
        if horizon not in (1, 3):
            continue
        for challenger in MODELS[1:]:
            by_date = monthly[challenger] - monthly["seasonal_pooled"]
            pooled = wide[challenger].mean() - wide["seasonal_pooled"].mean()
            leave_one = [float(by_date.drop(target).mean()) for target in by_date.index]
            # Identical municipality counts on every date make these exact paired pooled deltas.
            assert np.isclose(pooled, by_date.mean())
            sensitivity_rows.append({
                "horizon": horizon, "benchmark": "seasonal_pooled",
                "challenger": challenger, "dates": len(by_date),
                "municipalities_per_date": int(len(wide) // len(by_date)),
                "MAE_difference_rub": float(pooled),
                "months_challenger_better": int((by_date < 0).sum()),
                "months_challenger_worse": int((by_date > 0).sum()),
                "leave_one_date_out_min_rub": min(leave_one),
                "leave_one_date_out_max_rub": max(leave_one),
                "most_favorable_target": str(by_date.idxmin()),
                "least_favorable_target": str(by_date.idxmax()),
            })
    pd.DataFrame(monthly_rows).sort_values(["horizon", "target", "model"]).to_csv(
        OUT / "asof_monthly_mae.csv", index=False)
    pd.DataFrame(sensitivity_rows).sort_values(["horizon", "challenger"]).to_csv(
        OUT / "asof_month_robustness.csv", index=False)


if __name__ == "__main__":
    main()
