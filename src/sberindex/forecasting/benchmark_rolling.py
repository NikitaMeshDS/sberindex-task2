"""Reproducible rolling-origin pilot for the SberIndex municipal panel.

All feature values precede the forecast origin. 2024 is evaluation only.
The 256-municipality sample is fixed before model comparison.
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


from sberindex.paths import ROOT

DATA = ROOT / "data" / "consumption.parquet"
CONFIG = json.loads((ROOT / "config.json").read_text())
SAMPLE_SIZE = CONFIG["prophet_chronos_sample_size"]
HORIZONS = tuple(CONFIG["forecast_horizons_months"])


def load_panel():
    data = pd.read_parquet(DATA)
    panel = (
        data.loc[data.category == "Все категории"]
        .pivot(index="territory_id", columns="date", values="value")
        .dropna()
        .sort_index()
    )
    assert panel.shape == (2016, 24)
    assert np.all(panel.to_numpy() > 0)
    return panel


def score(frame):
    rows = []
    for (model, horizon), group in frame.groupby(["model", "horizon"]):
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "origins": group.origin.nunique(),
                "municipalities": group.territory_id.nunique(),
                "MAE": mean_absolute_error(group.actual, group.predicted),
                "R2": r2_score(group.actual, group.predicted),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "MAE"])


def main():
    panel = load_panel()
    values = panel.to_numpy(float)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(CONFIG["random_seed"])
    selected_ids = np.sort(
        rng.choice(panel.index.to_numpy(), SAMPLE_SIZE, replace=False)
    )
    sample_mask = panel.index.isin(selected_ids)
    sample_values = values[sample_mask]
    sample_ids = panel.index.to_numpy()[sample_mask]
    records = []

    def add(model, origin, horizon, ids, actual, prediction):
        assert np.isfinite(prediction).all(), (model, origin, horizon)
        records.extend(
            (
                int(i),
                months[origin].strftime("%Y-%m"),
                horizon,
                model,
                float(a),
                float(max(0, p)),
            )
            for i, a, p in zip(ids, actual, prediction)
        )

    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    from sberindex.forecasting.prophet_backend import legacy_predict
    import torch
    from chronos import ChronosBoltPipeline

    torch.set_num_threads(4)
    chronos = ChronosBoltPipeline.from_pretrained(
        CONFIG["chronos_model"], revision=CONFIG["chronos_revision"], device_map="cpu"
    )
    print("loaded data, Prophet and Chronos", flush=True)

    for origin in range(11, 23):
        horizons = [h for h in HORIZONS if origin + h < 24]
        origin_month = origin % 12
        max_h = max(horizons)
        seasonal_ratio = np.array(
            [
                values[:, h_target % 12].sum() / values[:, origin_month].sum()
                for h_target in range(origin + 1, origin + max_h + 1)
            ]
        )

        # Prophet fits each sampled city once per origin; Chronos batches contexts.
        prophet_forecast = np.empty((SAMPLE_SIZE, max_h))
        for j, history in enumerate(sample_values):
            prophet_forecast[j] = legacy_predict(
                history[: origin + 1],
                months[: origin + 1],
                months[origin + 1 : origin + 1 + max_h],
            )
        chronos_forecast = []
        for start in range(0, SAMPLE_SIZE, 64):
            context = torch.from_numpy(
                sample_values[start : start + 64, : origin + 1].astype(np.float32)
            )
            pred = chronos.predict(context, prediction_length=max_h)[:, 4, :]
            chronos_forecast.append(pred.detach().numpy())
        chronos_forecast = np.concatenate(chronos_forecast)

        for horizon in horizons:
            target = origin + horizon
            actual = values[:, target]
            anchor = values[:, origin]
            yearago = values[:, target - 12]
            ratio = seasonal_ratio[horizon - 1]
            add("last", origin, horizon, panel.index, actual, anchor)
            add("year_ago", origin, horizon, panel.index, actual, yearago)
            add("seasonal_pooled", origin, horizon, panel.index, actual, anchor * ratio)
            add(
                "seasonal_local",
                origin,
                horizon,
                panel.index,
                actual,
                anchor * values[:, target % 12] / values[:, origin_month],
            )
            if origin >= 12:
                add(
                    "same_month_yoy",
                    origin,
                    horizon,
                    panel.index,
                    actual,
                    yearago * anchor / values[:, origin - 12],
                )
            add(
                "prophet",
                origin,
                horizon,
                sample_ids,
                actual[sample_mask],
                prophet_forecast[:, horizon - 1],
            )
            add(
                "chronos_bolt_tiny",
                origin,
                horizon,
                sample_ids,
                actual[sample_mask],
                chronos_forecast[:, horizon - 1],
            )
        print(f"origin {months[origin]:%Y-%m} done", flush=True)

    frame = pd.DataFrame(
        records,
        columns=["territory_id", "origin", "horizon", "model", "actual", "predicted"],
    )
    frame.to_parquet(ROOT / "results" / "rolling_predictions.parquet", index=False)
    score(frame).to_csv(ROOT / "results" / "rolling_metrics_mixed.csv", index=False)
    sampled = frame[frame.territory_id.isin(selected_ids)]
    score(sampled).to_csv(ROOT / "results" / "rolling_metrics_sample.csv", index=False)
    print(score(sampled).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
