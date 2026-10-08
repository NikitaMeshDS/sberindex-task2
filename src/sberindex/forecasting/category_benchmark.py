"""Point-in-time category forecasts on the same late-2024 target months.

The 128 IDs are a deterministic subset of the original 256-ID benchmark sample.
Prophet is fitted independently per category and forecast origin.  The seasonal
ratio uses the 2023 cross-sectional panel only, including no 2024 targets.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sberindex.forecasting.prophet_backend import legacy_predict
from sklearn.metrics import r2_score


from sberindex.paths import ROOT

CATEGORIES = [
    "Все категории",
    "Здоровье",
    "Маркетплейсы",
    "Общественное питание",
    "Продовольствие",
    "Транспорт",
]
MONTHS = pd.date_range("2023-01-01", periods=24, freq="MS")
HORIZONS = (1, 3, 6)


def main():
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.disable(logging.INFO)
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    original = pd.read_parquet(ROOT / "results/rolling_predictions.parquet")
    original_ids = np.sort(
        original.loc[original.model == "prophet", "territory_id"].unique()
    )
    sample_ids = np.sort(
        np.random.default_rng(991).choice(original_ids, 128, replace=False)
    )
    assert len(sample_ids) == 128
    records = []
    for category in CATEGORIES:
        panel = (
            raw.loc[raw.category == category]
            .pivot(index="territory_id", columns="date", values="value")
            .reindex(columns=MONTHS.strftime("%Y-%m"))
            .dropna()
            .sort_index()
        )
        assert panel.index.isin(sample_ids).sum() == len(sample_ids)
        values = panel.to_numpy(float)
        idx = panel.index.get_indexer(sample_ids)
        sample_values = values[idx]
        for origin in range(11, 23):
            horizons = [h for h in HORIZONS if origin + h < 24 and origin + h >= 18]
            if not horizons:
                continue
            max_h = max(horizons)
            preds = np.empty((len(sample_ids), max_h))
            for j, history in enumerate(sample_values):
                preds[j] = legacy_predict(
                    history[: origin + 1],
                    MONTHS[: origin + 1],
                    MONTHS[origin + 1 : origin + 1 + max_h],
                )
            for horizon in horizons:
                target = origin + horizon
                seasonal_ratio = (
                    values[:, target % 12].sum() / values[:, origin % 12].sum()
                )
                actual = sample_values[:, target]
                candidates = {
                    "prophet": preds[:, horizon - 1],
                    "last": sample_values[:, origin],
                    "year_ago": sample_values[:, target - 12],
                    "seasonal_pooled": sample_values[:, origin] * seasonal_ratio,
                    "seasonal_local": sample_values[:, origin]
                    * sample_values[:, target % 12]
                    / sample_values[:, origin % 12],
                }
                for model, forecast in candidates.items():
                    records.extend(
                        (
                            category,
                            int(city),
                            MONTHS[origin].strftime("%Y-%m"),
                            MONTHS[target].strftime("%Y-%m"),
                            horizon,
                            model,
                            float(y),
                            float(max(0, p)),
                        )
                        for city, y, p in zip(sample_ids, actual, forecast)
                    )
        print(f"{category}: done", flush=True)
    predictions = pd.DataFrame(
        records,
        columns=[
            "category",
            "territory_id",
            "origin",
            "target",
            "horizon",
            "model",
            "actual",
            "predicted",
        ],
    )
    predictions.to_parquet(ROOT / "results/category_predictions.parquet", index=False)
    summary = []
    for (category, horizon, model), group in predictions.groupby(
        ["category", "horizon", "model"]
    ):
        ae = np.abs(group.actual - group.predicted)
        summary.append(
            {
                "category": category,
                "horizon": horizon,
                "model": model,
                "months": group.target.nunique(),
                "municipalities": group.territory_id.nunique(),
                "MAE": ae.mean(),
                "R2": r2_score(group.actual, group.predicted),
                "WAPE_pct": 100 * ae.sum() / group.actual.sum(),
            }
        )
    pd.DataFrame(summary).sort_values(["category", "horizon", "MAE"]).to_csv(
        ROOT / "results/category_forecast_comparison.csv", index=False
    )


if __name__ == "__main__":
    main()
