"""As-of CBR-news ablation for a one-month municipal forecasting model.

Training transitions end in 2023; late-2024 targets are excluded from fitting.
These evaluation dates have been repeatedly analysed, so they are not an unseen holdout.
Only releases available by the previous month-end may enter a prediction.
This tests incremental predictive value, not a causal effect of rate news.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score


from sberindex.paths import ROOT
CONFIG = json.loads((ROOT / "config.json").read_text())


def known_news(origin, events):
    end = origin + pd.offsets.MonthEnd(0)
    known = events.loc[events.available_from <= end]
    recent = known.loc[known.available_from > end - pd.Timedelta(days=90)]
    return np.array([known.rate_after_pct.iloc[-1] if len(known) else 7.5,
                     recent.delta_bps.sum() / 100, len(recent)], float)


def main():
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (raw.loc[raw.category == "Все категории"]
             .pivot(index="territory_id", columns="date", values="value")
             .dropna().sort_index())
    values = panel.to_numpy(float)
    logs = np.log(values)
    events = pd.read_csv(ROOT / "data/external/news_events.csv", parse_dates=["published_date", "available_from"])
    months = pd.date_range("2023-01-01", periods=24, freq="MS")

    def features(origin):
        base = np.column_stack([
            logs[:, origin],
            logs[:, origin] - logs[:, origin - 1],
            logs[:, origin - 1] - logs[:, origin - 2],
            np.full(len(values), np.sin(2 * np.pi * ((origin + 1) % 12) / 12)),
            np.full(len(values), np.cos(2 * np.pi * ((origin + 1) % 12) / 12)),
        ])
        news = np.tile(known_news(months[origin], events), (len(values), 1))
        return base, np.column_stack([base, news])

    train = [features(i) for i in range(2, 11)]
    target = np.concatenate([logs[:, i + 1] - logs[:, i] for i in range(2, 11)])
    model_kwargs = dict(max_iter=CONFIG["hgb_max_iter"],
                        max_leaf_nodes=CONFIG["hgb_max_leaf_nodes"],
                        learning_rate=CONFIG["hgb_learning_rate"],
                        min_samples_leaf=CONFIG["hgb_min_samples_leaf"],
                        l2_regularization=CONFIG["hgb_l2_regularization"],
                        random_state=CONFIG["random_seed"])
    models = {}
    for name, column in (("without_news", 0), ("with_asof_news", 1)):
        model = HistGradientBoostingRegressor(**model_kwargs)
        model.fit(np.concatenate([part[column] for part in train]), target)
        models[name] = model

    rows = []
    for origin in range(12, 23):  # targets February-December 2024
        base, enriched = features(origin)
        actual = values[:, origin + 1]
        for name, model in models.items():
            X = base if name == "without_news" else enriched
            predicted = np.exp(logs[:, origin] + np.clip(model.predict(X), -0.5, 0.5))
            rows.extend((int(city), months[origin].strftime("%Y-%m"),
                         months[origin + 1].strftime("%Y-%m"), name, float(y), float(p))
                        for city, y, p in zip(panel.index, actual, predicted))
    predictions = pd.DataFrame(rows, columns=["territory_id", "origin", "target",
                                               "model", "actual", "predicted"])
    predictions.to_parquet(ROOT / "results/news_ablation_predictions.parquet", index=False)
    summary = []
    late = predictions[predictions.target >= "2024-07"]
    for name, group in late.groupby("model"):
        ae = np.abs(group.actual - group.predicted)
        summary.append({"model": name, "dates": group.target.nunique(),
                        "municipalities": group.territory_id.nunique(), "MAE": ae.mean(),
                        "R2": r2_score(group.actual, group.predicted),
                        "WAPE_pct": 100 * ae.sum() / group.actual.sum()})
    summary = pd.DataFrame(summary).sort_values("MAE")
    summary.to_csv(ROOT / "results/news_ablation.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
