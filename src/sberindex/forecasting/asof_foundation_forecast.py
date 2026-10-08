"""Chronos benchmarks on the 2023-known sample used by asof_cohort_forecast.

Model revisions, origins, admissible histories and scored targets match the
other as-of cohort forecasts. This is an auxiliary retrospective audit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline, ChronosBoltPipeline

from sberindex.paths import ROOT
OUT = ROOT / "results"


def main():
    config = json.loads((ROOT / "config.json").read_text())
    cohort = json.loads((OUT / "asof_cohort_protocol.json").read_text())
    ids = np.array(cohort["sample_ids"], dtype=int)
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (raw.loc[raw.category == "Все категории"]
             .pivot(index="territory_id", columns="date", values="value")
             .loc[ids])
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    assert list(panel.columns) == [str(d.to_period("M")) for d in months]
    torch.set_num_threads(4)
    bolt = ChronosBoltPipeline.from_pretrained(
        config["chronos_model"], revision=config["chronos_revision"], device_map="cpu")
    chronos2 = Chronos2Pipeline.from_pretrained(
        config["chronos2_model"], revision=config["chronos2_revision"], device_map="cpu")
    rows = []
    for origin in [11, *range(12, 23)]:
        horizons = ([12] if origin == 11 else [1] if origin < 17 else
                    [h for h in [1, 3, 6] if origin+h < 24])
        history = panel.iloc[:, :origin+1]
        valid = history.notna().all(axis=1)
        active = history.loc[valid]
        active_ids = active.index.to_numpy(int)
        if len(active) == 0:
            continue
        max_h = max(horizons)
        bolt_parts = []
        for start in range(0, len(active), 64):
            context = torch.from_numpy(active.iloc[start:start+64].to_numpy(np.float32))
            block = bolt.predict(context, prediction_length=max_h)[:, 4, :]
            bolt_parts.append(block.detach().cpu().numpy())
        bolt_forecast = np.concatenate(bolt_parts)

        long = active.stack().rename("target").reset_index()
        long.columns = ["item_id", "date", "target"]
        long["timestamp"] = pd.to_datetime(long.pop("date") + "-01")
        c2_frame = chronos2.predict_df(long, prediction_length=max_h,
                                       quantile_levels=[0.5], batch_size=64, freq="MS")
        c2_frame["timestamp"] = pd.to_datetime(c2_frame.timestamp)
        for horizon in horizons:
            target = origin+horizon
            actual = panel.loc[active_ids].iloc[:, target].to_numpy(float)
            observed = np.isfinite(actual)
            if not observed.any():
                continue
            q = c2_frame.loc[c2_frame.timestamp == months[target]].set_index("item_id")
            c2_forecast = q.loc[active_ids, "0.5"].to_numpy(float)
            for name, prediction in [("chronos_bolt_tiny", bolt_forecast[:, horizon-1]),
                                     ("chronos_2", c2_forecast)]:
                if not np.isfinite(prediction[observed]).all():
                    raise ValueError(f"Non-finite {name} forecast at {months[origin]:%Y-%m}")
                rows.extend({"territory_id": int(i),
                             "origin": str(months[origin].to_period("M")),
                             "target": str(months[target].to_period("M")),
                             "horizon": horizon, "model": name,
                             "actual": float(a), "predicted": max(0., float(p))}
                            for i,a,p in zip(active_ids[observed],actual[observed],prediction[observed]))
        print(f"As-of foundation origin {months[origin]:%Y-%m} complete", flush=True)
    result = pd.DataFrame(rows)
    assert not result.duplicated(["territory_id", "origin", "horizon", "model"]).any()
    result.to_parquet(OUT / "asof_foundation_predictions.parquet", index=False)
    protocol = {
        "sample_source": "results/asof_cohort_protocol.json",
        "chronos_bolt_revision": config["chronos_revision"],
        "chronos_2_revision": config["chronos2_revision"],
        "history_rule": "All observed months from 2023-01 through origin; skip ID-origin with missing history",
        "target_rule": "Score only observed target; never impute a missing actual",
        "models": ["chronos_bolt_tiny", "chronos_2"],
        "limits": "Same already studied 2024 archive and model families; no independent validation.",
    }
    (OUT / "asof_foundation_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(f"Saved {len(result)} foundation forecast rows")


if __name__ == "__main__":
    main()
