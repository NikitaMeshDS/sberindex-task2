"""Recompute the exploratory 12-month HGB using only 2023-known IDs."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sberindex.paths import ROOT
OUT = ROOT / "results"


def main():
    config = json.loads((ROOT / "config.json").read_text())
    protocol = json.loads((OUT / "asof_cohort_protocol.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (raw.loc[raw.category == "Все категории"]
             .pivot(index="territory_id", columns="date", values="value")
             .sort_index())
    eligible = panel.iloc[:, :12].notna().all(axis=1)
    ids = panel.index[eligible].to_numpy()
    assert len(ids) == 2075
    values = panel.loc[ids].iloc[:, :12].to_numpy(float)
    log_values = np.log(values)
    features, targets = [], []
    for target in range(3, 12):
        month = target+1
        features.append(np.column_stack([
            log_values[:, target-1],
            log_values[:, target-1]-log_values[:, target-2],
            log_values[:, target-2]-log_values[:, target-3],
            np.full(len(ids), np.sin(2*np.pi*month/12)),
            np.full(len(ids), np.cos(2*np.pi*month/12)),
        ]))
        targets.append(log_values[:, target]-log_values[:, target-1])
    model = HistGradientBoostingRegressor(
        max_iter=config["hgb_max_iter"],
        max_leaf_nodes=config["hgb_max_leaf_nodes"],
        learning_rate=config["hgb_learning_rate"],
        min_samples_leaf=config["hgb_min_samples_leaf"],
        l2_regularization=config["hgb_l2_regularization"],
        random_state=config["random_seed"])
    model.fit(np.concatenate(features), np.concatenate(targets))
    trajectory = log_values.copy()
    for target in range(12, 24):
        month = target%12+1
        step = np.column_stack([
            trajectory[:, -1],
            trajectory[:, -1]-trajectory[:, -2],
            trajectory[:, -2]-trajectory[:, -3],
            np.full(len(ids), np.sin(2*np.pi*month/12)),
            np.full(len(ids), np.cos(2*np.pi*month/12)),
        ])
        growth = np.clip(model.predict(step), -.5, .5)
        trajectory = np.column_stack([trajectory, trajectory[:, -1]+growth])
    forecast = pd.Series(np.exp(trajectory[:, -1]), index=ids)
    selected_ids = protocol["sample_ids"]
    actual = panel.loc[selected_ids, "2024-12"]
    observed = actual.notna()
    result = pd.DataFrame({
        "territory_id": np.asarray(selected_ids)[observed.to_numpy()],
        "origin": "2023-12", "target": "2024-12", "horizon": 12,
        "actual": actual.loc[observed].to_numpy(float),
        "predicted": forecast.loc[actual.index[observed]].to_numpy(float),
    })
    assert len(result) == 252
    result.to_csv(OUT / "asof_hgb_12m.csv", index=False)
    print(f"As-of HGB train IDs {len(ids)}, observed test IDs {len(result)}, MAE {np.abs(result.actual-result.predicted).mean():.2f}")


if __name__ == "__main__":
    main()
