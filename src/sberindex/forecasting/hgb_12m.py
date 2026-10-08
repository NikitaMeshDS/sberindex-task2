"""Global recursive 12-month baseline trained only on 2023 monthly history."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


from sberindex.paths import ROOT
root = ROOT
config = json.loads((root / "config.json").read_text())
data = pd.read_parquet(root / "data" / "consumption.parquet")
panel = (data.loc[data.category == "Все категории"]
         .pivot(index="territory_id", columns="date", values="value")
         .dropna().sort_index())
values = panel.to_numpy(float)
logv = np.log(values[:, :12])
X, y = [], []
for target in range(3, 12):
    month = target + 1
    features = np.column_stack([
        logv[:, target - 1],
        logv[:, target - 1] - logv[:, target - 2],
        logv[:, target - 2] - logv[:, target - 3],
        np.full(len(values), np.sin(2 * np.pi * month / 12)),
        np.full(len(values), np.cos(2 * np.pi * month / 12)),
    ])
    X.append(features)
    y.append(logv[:, target] - logv[:, target - 1])
model = HistGradientBoostingRegressor(
    max_iter=config["hgb_max_iter"],
    max_leaf_nodes=config["hgb_max_leaf_nodes"],
    learning_rate=config["hgb_learning_rate"],
    min_samples_leaf=config["hgb_min_samples_leaf"],
    l2_regularization=config["hgb_l2_regularization"],
    random_state=config["random_seed"])
model.fit(np.concatenate(X), np.concatenate(y))

forecast_log = logv.copy()
for target in range(12, 24):
    month = target % 12 + 1
    features = np.column_stack([
        forecast_log[:, -1],
        forecast_log[:, -1] - forecast_log[:, -2],
        forecast_log[:, -2] - forecast_log[:, -3],
        np.full(len(values), np.sin(2 * np.pi * month / 12)),
        np.full(len(values), np.cos(2 * np.pi * month / 12)),
    ])
    growth = np.clip(model.predict(features), -0.5, 0.5)
    forecast_log = np.column_stack([forecast_log, forecast_log[:, -1] + growth])

forecast = np.exp(forecast_log[:, 12:24])
result = pd.DataFrame({"territory_id": panel.index,
                       "actual": values[:, 23], "predicted": forecast[:, 11]})
result.to_csv(root / "results" / "hgb_12m_predictions.csv", index=False)
print("full 2016", mean_absolute_error(result.actual, result.predicted),
      r2_score(result.actual, result.predicted))
comparison = pd.read_parquet(root / "results" / "rolling_predictions.parquet")
ids = set(comparison.loc[(comparison.model == "prophet") &
                         (comparison.horizon == 12), "territory_id"])
sample = result[result.territory_id.isin(ids)]
print("sample 256", mean_absolute_error(sample.actual, sample.predicted),
      r2_score(sample.actual, sample.predicted))
