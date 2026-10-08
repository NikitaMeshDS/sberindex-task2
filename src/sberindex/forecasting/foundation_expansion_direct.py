"""Direct HGB full-panel extension using the frozen direct-horizon algorithm."""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from sberindex.forecasting.direct_horizon_review import feature_block, training_indices
from sberindex.forecasting.foundation_expansion_review import OUT, metrics
from sberindex.paths import ROOT


def run():
    cfg = json.loads((ROOT / "configs/direct_horizon_review.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    months = pd.period_range("2023-01", "2024-12", freq="M").astype(str)
    panel = (
        raw[raw.category.eq("Все категории")]
        .pivot(index="territory_id", columns="date", values="value")
        .reindex(columns=months)
        .sort_index()
    )
    valid = (np.isfinite(panel.iloc[:, :12]) & (panel.iloc[:, :12] > 0)).all(axis=1)
    panel = panel.loc[valid]
    values = panel.to_numpy(float)
    cats = np.stack(
        [
            raw[raw.category.eq(c)]
            .pivot(index="territory_id", columns="date", values="value")
            .reindex(index=panel.index, columns=months)
            .to_numpy(float)
            for c in cfg["covariate_categories"]
        ],
        axis=1,
    )
    geo = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    regions = (
        geo[geo.year.eq(2023)]
        .set_index("territory_id")
        .region_code.reindex(panel.index)
        .to_numpy(float)
    )
    full = pd.read_parquet(ROOT / "reports/full_cohort_review/predictions.parquet")
    base = full[full.model.eq("seasonal_pooled")].copy()
    records = []
    runtime = []
    with threadpool_limits(limits=1):
        for (origin, h), group in base.groupby(["origin", "horizon"], sort=True):
            t = pd.Period(origin, "M").ordinal - pd.Period("2023-01", "M").ordinal
            h = int(h)
            idx = training_indices(values, t, h, 2)
            ids = panel.index.get_indexer(group.territory_id)
            for variant in cfg["variants"]:
                tick = time.monotonic()
                context = variant == "direct_context"
                fallback = len(idx) < cfg["minimum_training_rows"]
                if fallback:
                    pred = group.predicted.to_numpy()
                    name = variant + "_full__seasonal_fallback_not_trainable"
                else:
                    features = []
                    labels = []
                    for s in np.unique(idx[:, 1]):
                        members = idx[idx[:, 1] == s, 0]
                        features.append(
                            feature_block(values, cats, regions, int(s), h, context)[
                                members
                            ]
                        )
                        labels.append(
                            np.log(
                                values[members, int(s) + h] / values[members, int(s)]
                            )
                        )
                    model = HistGradientBoostingRegressor(**cfg["model"]).fit(
                        np.concatenate(features), np.concatenate(labels)
                    )
                    pred = values[ids, t] * np.exp(
                        np.clip(
                            model.predict(
                                feature_block(values, cats, regions, t, h, context)[ids]
                            ),
                            -1,
                            1,
                        )
                    )
                    name = variant + "_full"
                z = group.copy()
                z["model"] = name
                z["predicted"] = pred
                z["category"] = "Все категории"
                z["status"] = (
                    "seasonal_fallback_not_trainable" if fallback else "trained_direct"
                )
                records.append(z)
                runtime.append(
                    {
                        "origin": origin,
                        "horizon": h,
                        "model": name,
                        "training_rows": len(idx),
                        "inference_pairs": len(group),
                        "seconds": time.monotonic() - tick,
                    }
                )
                print(runtime[-1], flush=True)
    frame = pd.concat(records, ignore_index=True)
    frame.to_parquet(OUT / "direct_full_predictions.parquet", index=False)
    pd.DataFrame(runtime).to_csv(OUT / "direct_full_runtime.csv", index=False)
    pd.DataFrame(
        [
            dict(model=m, horizon=int(h), **metrics(g))
            for (m, h), g in frame.groupby(["model", "horizon"])
        ]
    ).to_csv(OUT / "direct_full_summary.csv", index=False)


if __name__ == "__main__":
    run()
