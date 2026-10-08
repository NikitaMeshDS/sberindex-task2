"""Rolling-origin Chronos-2 benchmark on the same fixed 256 municipalities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline


from sberindex.paths import ROOT
CONFIG = json.loads((ROOT / "config.json").read_text())
torch.set_num_threads(4)
data = pd.read_parquet(ROOT / "data" / "consumption.parquet")
panel = (data.loc[data.category == "Все категории"]
         .pivot(index="territory_id", columns="date", values="value")
         .dropna().sort_index())
ids = np.sort(np.random.default_rng(CONFIG["random_seed"]).choice(panel.index.to_numpy(), CONFIG["prophet_chronos_sample_size"], replace=False))
panel = panel.loc[ids]
months = pd.date_range("2023-01-01", periods=24, freq="MS")
pipeline = Chronos2Pipeline.from_pretrained(CONFIG["chronos2_model"], revision=CONFIG["chronos2_revision"], device_map="cpu")
rows = []
for origin in range(11, 23):
    horizons = [h for h in (1, 3, 6, 12) if origin + h < 24]
    history = panel.iloc[:, :origin + 1]
    long = history.stack().rename("target").reset_index()
    long.columns = ["item_id", "date", "target"]
    long["timestamp"] = pd.to_datetime(long.pop("date") + "-01")
    pred = pipeline.predict_df(long, prediction_length=max(horizons),
                               quantile_levels=[0.5], batch_size=64, freq="MS")
    pred["timestamp"] = pd.to_datetime(pred.timestamp)
    for horizon in horizons:
        target_date = months[origin + horizon]
        q = pred.loc[pred.timestamp == target_date].set_index("item_id")
        actual = panel.iloc[:, origin + horizon].to_numpy(float)
        prediction = q.loc[ids, "0.5"].to_numpy(float)
        rows.extend({"territory_id": int(i),
                     "origin": months[origin].strftime("%Y-%m"),
                     "horizon": horizon, "model": "chronos_2",
                     "actual": a, "predicted": max(0, p)}
                    for i, a, p in zip(ids, actual, prediction))
    print(f"origin {months[origin]:%Y-%m} done", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "chronos2_rolling.csv", index=False)
