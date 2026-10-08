"""Retrospective reporting-lag sensitivity on the fixed 2023-known cohort.

Issue at the end of month t, target t+1, latest visible fact t-delay.
Only the delay=1 / horizon=2 Prophet fits are new. Delays 0 and 2 reuse
the as-of cohort forecasts, giving matched September-December targets.
Publication vintages are unavailable, so these are hypothetical lags.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sberindex.forecasting.prophet_backend import legacy_predict
from sklearn.metrics import r2_score

from sberindex.paths import ROOT

OUT = ROOT / "results"
TARGETS = [str(p) for p in pd.period_range("2024-09", "2024-12", freq="M")]


def main():
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    protocol = json.loads((OUT / "asof_cohort_protocol.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (
        raw.loc[raw.category == "Все категории"]
        .pivot(index="territory_id", columns="date", values="value")
        .sort_index()
    )
    ids = np.asarray(protocol["sample_ids"], dtype=int)
    eligible = panel.index[panel.iloc[:, :12].notna().all(axis=1)]
    profile = panel.loc[eligible].iloc[:, :12].sum().to_numpy(float)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    assert panel.columns.tolist() == [str(x.to_period("M")) for x in months]
    earlier = pd.read_parquet(OUT / "asof_cohort_predictions.parquet")
    validation = pd.read_csv(OUT / "asof_cohort_validation.csv")
    weight = float(validation.loc[validation.MAE.idxmin(), "seasonal_weight"])
    assert weight == 0.75

    records = []
    for target in TARGETS:
        target_idx = (
            pd.Period(target, freq="M").ordinal - pd.Period("2023-01", freq="M").ordinal
        )
        origin_idx = target_idx - 2  # one month of reporting lag
        origin = str(months[origin_idx].to_period("M"))
        for territory_id in ids:
            values = panel.loc[territory_id].to_numpy(float)
            history = values[: origin_idx + 1]
            if not np.isfinite(history).all() or not np.isfinite(values[target_idx]):
                continue
            predicted = legacy_predict(
                history,
                months[: origin_idx + 1],
                months[origin_idx + 1 : target_idx + 1],
            )
            prophet = max(0.0, float(predicted[-1]))
            seasonal = float(
                values[origin_idx] * profile[target_idx % 12] / profile[origin_idx % 12]
            )
            key = {
                "territory_id": int(territory_id),
                "origin": origin,
                "target": target,
                "horizon": 2,
                "actual": float(values[target_idx]),
            }
            records.extend(
                [
                    {**key, "model": "prophet", "predicted": prophet},
                    {**key, "model": "seasonal_pooled", "predicted": seasonal},
                ]
            )
        print(f"One-month reporting lag, target {target} complete", flush=True)

    new = pd.DataFrame(records)
    reused = earlier[(earlier.target.isin(TARGETS)) & (earlier.horizon.isin([1, 3]))]
    base = pd.concat([reused, new], ignore_index=True)
    key = ["territory_id", "origin", "target", "horizon"]
    wide = base.pivot(index=key, columns="model", values=["actual", "predicted"])
    assert wide.notna().all().all()
    np.testing.assert_allclose(
        wide[("actual", "prophet")], wide[("actual", "seasonal_pooled")]
    )
    wide["blend_75"] = (
        weight * wide[("predicted", "seasonal_pooled")]
        + (1 - weight) * wide[("predicted", "prophet")]
    )
    prepared = wide.reset_index()
    prepared.columns = [
        "_".join(str(c) for c in pair if c) if isinstance(pair, tuple) else pair
        for pair in prepared.columns
    ]
    prepared["actual"] = prepared["actual_prophet"]
    # Each target must be scored on the intersection across all three lags.
    common = (
        prepared.groupby(["territory_id", "target"])
        .horizon.nunique()
        .loc[lambda s: s == 3]
        .index
    )
    prepared = prepared.set_index(["territory_id", "target"]).loc[common].reset_index()
    assert prepared.groupby(["territory_id", "target"]).horizon.nunique().eq(3).all()
    assert prepared.groupby(["territory_id", "target"]).actual.nunique().eq(1).all()
    rows = []
    for model, column in [
        ("prophet", "predicted_prophet"),
        ("seasonal_pooled", "predicted_seasonal_pooled"),
        ("blend_75", "blend_75"),
    ]:
        q = prepared[
            ["territory_id", "origin", "target", "horizon", "actual", column]
        ].copy()
        q = q.rename(columns={column: "predicted"})
        q["model"] = model
        q["reporting_delay_months"] = q.horizon - 1
        q["decision_month"] = (pd.PeriodIndex(q.target, freq="M") - 1).astype(str)
        q["absolute_error"] = (q.actual - q.predicted).abs()
        rows.append(q)
    scored = pd.concat(rows, ignore_index=True)
    assert not scored.duplicated(
        ["territory_id", "target", "model", "reporting_delay_months"]
    ).any()
    assert set(scored.decision_month) == {"2024-08", "2024-09", "2024-10", "2024-11"}
    scored.to_parquet(OUT / "asof_reporting_delay_predictions.parquet", index=False)

    summary = []
    for (model, delay), q in scored.groupby(["model", "reporting_delay_months"]):
        by_month = q.groupby("target").absolute_error.mean()
        summary.append(
            {
                "model": model,
                "reporting_delay_months": int(delay),
                "MAE_date_balanced": float(by_month.mean()),
                "MAE_pooled": float(q.absolute_error.mean()),
                "R2_pooled": float(r2_score(q.actual, q.predicted)),
                "dates": int(q.target.nunique()),
                "municipalities": int(q.territory_id.nunique()),
                "observations": len(q),
            }
        )
    pd.DataFrame(summary).sort_values(["model", "reporting_delay_months"]).to_csv(
        OUT / "asof_reporting_delay_summary.csv", index=False
    )
    scored.groupby(["model", "reporting_delay_months", "target"]).agg(
        MAE=("absolute_error", "mean"), municipalities=("territory_id", "nunique")
    ).to_csv(OUT / "asof_reporting_delay_monthly.csv")
    audit = {
        "decision_rule": "At end of t, target t+1; last observation t-delay; effective horizon 1+delay",
        "delays_months": [0, 1, 2],
        "target_months": TARGETS,
        "sample": "256 fixed IDs selected using only 2023 completeness; score observed targets present under all lags",
        "new_training": "Only delay=1 Prophet is refit; delay=0 and 2 reuse asof_cohort_predictions.parquet",
        "seasonal_profile": "2023 sum across all 2075 eligible municipalities; identical to asof_cohort_forecast.py",
        "blend_weight": weight,
        "blend_selection": "February-June 2024 targets (five dates), already selected in asof_cohort_summary.py",
        "limits": "Hypothetical reporting delays: historical release dates and vintage revisions are unavailable. Retrospective reused 2024 data, no independent test. Other models are not compared here. Four shared target dates limit inference.",
    }
    (OUT / "asof_reporting_delay_protocol.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
