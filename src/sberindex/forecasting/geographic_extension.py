"""Frozen-model diagnostic on regions absent from the original 256-MO sample.

Up to four municipalities per missing region, selected with fixed seed using
IDs only. This is a geographic diagnostic on already reused dates, not a new
independent time holdout. Original sample and weights are not changed.
"""

import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sberindex.forecasting.prophet_backend import legacy_predict
from sberindex.forecasting.benchmark_rolling import load_panel

from sberindex.paths import ROOT


def main():
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    panel = load_panel()
    values = panel.to_numpy(float)
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    lookup = lookup[lookup.year == 2024].set_index("territory_id")
    meta = lookup.loc[panel.index]
    primary = pd.read_parquet(ROOT / "results/primary_predictions.parquet")
    sampled = meta.loc[primary.territory_id.unique(), "region_code"].unique()
    missing = sorted(set(meta.region_code) - set(sampled))
    chosen = []
    rng = np.random.default_rng(20261002)
    for region in missing:
        ids = meta.index[meta.region_code == region].to_numpy()
        chosen.extend(rng.choice(ids, min(4, len(ids)), replace=False))
    chosen = sorted(chosen)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    records = []
    for city in chosen:
        history = panel.loc[city].to_numpy(float)
        for origin in range(17, 23):
            horizons = [h for h in [1, 3, 6] if origin + h < 24]
            max_h = max(horizons)
            prophet = legacy_predict(
                history[: origin + 1],
                months[: origin + 1],
                months[origin + 1 : origin + 1 + max_h],
            )
            for horizon in horizons:
                target = origin + horizon
                season = (
                    history[origin]
                    * values[:, target % 12].sum()
                    / values[:, origin % 12].sum()
                )
                pred = max(0, float(prophet[horizon - 1]))
                for name, prediction in [
                    ("prophet", pred),
                    ("seasonal_pooled", season),
                    ("blend_75", 0.75 * season + 0.25 * pred),
                ]:
                    records.append(
                        {
                            "territory_id": city,
                            "region_code": int(meta.loc[city, "region_code"]),
                            "region_name": meta.loc[city, "region_name"],
                            "origin": str(months[origin].to_period("M")),
                            "target": str(months[target].to_period("M")),
                            "horizon": horizon,
                            "model": name,
                            "actual": history[target],
                            "predicted": prediction,
                        }
                    )
        print(f"Completed {city}", flush=True)
    result = pd.DataFrame(records)
    result.to_parquet(
        ROOT / "results/geographic_extension_predictions.parquet", index=False
    )
    result["ae"] = (result.actual - result.predicted).abs()
    result.groupby(["horizon", "model"]).agg(
        MAE=("ae", "mean"),
        municipalities=("territory_id", "nunique"),
        regions=("region_code", "nunique"),
        dates=("target", "nunique"),
    ).to_csv(ROOT / "results/geographic_extension_summary.csv")
    result.groupby(["region_name", "horizon", "model"]).ae.mean().to_csv(
        ROOT / "results/geographic_extension_regions.csv"
    )
    (ROOT / "results/geographic_extension_protocol.json").write_text(
        json.dumps(
            {
                "seed": 20261002,
                "regions": [int(r) for r in missing],
                "municipalities": [int(i) for i in chosen],
                "models": "same Prophet settings and fixed blend 0.75",
                "limits": "Retrospective geographic diagnostic, equal up-to-four allocation not population representative; no new independent dates.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(result.groupby(["horizon", "model"]).ae.mean().to_string())


if __name__ == "__main__":
    main()
