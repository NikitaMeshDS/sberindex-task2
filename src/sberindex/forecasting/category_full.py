"""Vectorized six-category benchmark on every complete municipal series."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


from sberindex.paths import ROOT
HORIZONS = (1, 3, 6)


def main():
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    rows = []
    for category, subset in raw.groupby("category"):
        panel = (subset.pivot(index="territory_id", columns="date", values="value")
                 .dropna().sort_index())
        values = panel.to_numpy(float)
        assert values.shape == (2016, 24)
        for horizon in HORIZONS:
            actuals = []; predictions = {name: [] for name in ("last", "year_ago", "seasonal_pooled", "seasonal_local")}
            for target in range(18, 24):
                origin = target - horizon
                ratio = values[:, target % 12].sum() / values[:, origin % 12].sum()
                actuals.append(values[:, target])
                predictions["last"].append(values[:, origin])
                predictions["year_ago"].append(values[:, target - 12])
                predictions["seasonal_pooled"].append(values[:, origin] * ratio)
                predictions["seasonal_local"].append(
                    values[:, origin] * values[:, target % 12] / values[:, origin % 12])
            actual = np.concatenate(actuals)
            for name, estimates in predictions.items():
                forecast = np.concatenate(estimates)
                ae = np.abs(actual - forecast)
                rows.append({"category": category, "horizon": horizon,
                             "model": name, "months": 6, "municipalities": len(panel),
                             "MAE": ae.mean(), "R2": r2_score(actual, forecast),
                             "WAPE_pct": 100 * ae.sum() / actual.sum()})
    result = pd.DataFrame(rows).sort_values(["category", "horizon", "MAE"])
    result.to_csv(ROOT / "results/category_full_comparison.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
