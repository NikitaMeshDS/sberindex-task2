"""Prophet and seasonal benchmarks on IDs selectable at the end of 2023.

This auxiliary experiment changes the municipal sample and seasonal profile.
It uses the already explored 2024 archive, so it is a design audit rather than
an independent evaluation period.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sberindex.forecasting.prophet_backend import legacy_predict

from sberindex.paths import ROOT

OUT = ROOT / "results"


def main():
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    config = json.loads((ROOT / "config.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (
        raw.loc[raw.category == "Все категории"]
        .pivot(index="territory_id", columns="date", values="value")
        .sort_index()
    )
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    assert list(panel.columns) == [str(date.to_period("M")) for date in months]
    known = panel.iloc[:, :12].notna().all(axis=1)
    eligible_ids = panel.index[known].to_numpy()
    assert len(eligible_ids) == 2075
    ids = np.sort(
        np.random.default_rng(config["random_seed"]).choice(
            eligible_ids, config["prophet_chronos_sample_size"], replace=False
        )
    )
    profile = panel.loc[eligible_ids].iloc[:, :12].sum().to_numpy(float)
    records = []
    skipped_history = 0
    for origin in [11, *range(12, 23)]:
        horizons = (
            [12]
            if origin == 11
            else [1]
            if origin < 17
            else [h for h in [1, 3, 6] if origin + h < 24]
        )
        max_h = max(horizons)
        for territory_id in ids:
            values = panel.loc[territory_id].to_numpy(float)
            history = values[: origin + 1]
            if not np.isfinite(history).all():
                skipped_history += 1
                continue
            observed = [h for h in horizons if np.isfinite(values[origin + h])]
            if not observed:
                continue
            predicted = legacy_predict(
                history, months[: origin + 1], months[origin + 1 : origin + max_h + 1]
            )
            for horizon in observed:
                target = origin + horizon
                actual = float(values[target])
                season = float(
                    values[origin] * profile[target % 12] / profile[origin % 12]
                )
                key = {
                    "territory_id": int(territory_id),
                    "origin": str(months[origin].to_period("M")),
                    "target": str(months[target].to_period("M")),
                    "horizon": horizon,
                    "actual": actual,
                }
                records.append(
                    {
                        **key,
                        "model": "prophet",
                        "predicted": max(0.0, float(predicted[horizon - 1])),
                    }
                )
                records.append({**key, "model": "seasonal_pooled", "predicted": season})
        print(f"As-of cohort origin {months[origin]:%Y-%m} complete", flush=True)
    frame = pd.DataFrame(records)
    assert not frame.duplicated(["territory_id", "origin", "horizon", "model"]).any()
    frame.to_parquet(OUT / "asof_cohort_predictions.parquet", index=False)
    protocol = {
        "sample_selection": "Random seed applied to 2075 IDs with all 2023 months, without inspecting 2024 completeness",
        "seed": config["random_seed"],
        "eligible_ids": len(eligible_ids),
        "sample_ids": ids.astype(int).tolist(),
        "seasonal_profile": "Sum of 2023 monthly spending over all 2075 eligible IDs",
        "prophet": "Same settings as benchmark_rolling.py, fit on observed history through each origin",
        "skipped_id_origins_with_missing_history": skipped_history,
        "missing_targets": "No score row when actual target is absent; no imputation",
        "horizons": "Dec 2023: 12; Jan-May 2024: 1 for early weight selection; Jun-Nov 2024: observable 1,3,6",
        "limits": "Retrospective 2024 design audit after exploration of archive; not an independent time test. Model family and seed were already studied on old cohort.",
    }
    (OUT / "asof_cohort_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2)
    )
    print(f"Saved {len(frame)} model rows for {len(ids)} selected municipalities")


if __name__ == "__main__":
    main()
