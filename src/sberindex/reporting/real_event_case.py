"""Descriptive, externally dated Orsk event check; not a causal evaluation.

The event occurs inside the detector's January-June calibration period, so
this script does not score it as a true positive or a false negative.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sberindex.paths import ROOT
def main():
    event = pd.read_csv(ROOT / "data/external/mchs_orsk_event.csv").iloc[0]
    assert event.event_id == "orsk_flood_2024" and int(event.territory_id) == 1673
    assert event.event_start == "2024-04-05" and event.first_report_date == "2024-04-06"
    assert event.annotation_role == "illustrative_context_not_spending_shift_ground_truth"
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    forecasts = pd.read_parquet(ROOT / "results/rolling_predictions.parquet")
    rows = []
    for category, subset in raw.groupby("category"):
        panel = subset.pivot(index="territory_id", columns="date", values="value")
        assert int(event.territory_id) in panel.index
        for month in pd.period_range("2024-03", "2024-06", freq="M"):
            now, before = str(month), str(month - 12)
            valid = panel[[now, before]].dropna()
            yoy = 100 * (valid[now] / valid[before] - 1)
            city = int(event.territory_id)
            assert city in valid.index and len(valid) >= 2016
            row = {
                "event_id": event.event_id,
                "territory_id": city,
                "category": category,
                "month": now,
                "event_month": now == "2024-04",
                "spending_rub": float(valid.loc[city, now]),
                "spending_prior_year_rub": float(valid.loc[city, before]),
                "yoy_pct": float(yoy.loc[city]),
                "panel_median_yoy_pct": float(yoy.median()),
                "panel_percentile_yoy": float(yoy.rank(pct=True).loc[city] * 100),
                "observed_pairs": len(valid),
                "seasonal_forecast_rub": np.nan,
                "seasonal_forecast_error_pct": np.nan,
            }
            if category == "Все категории" and now == "2024-04":
                forecast = forecasts.loc[
                    (forecasts.territory_id == city)
                    & (forecasts.origin == "2024-03")
                    & (forecasts.horizon == 1)
                    & (forecasts.model == "seasonal_pooled")]
                assert len(forecast) == 1
                predicted = float(forecast.predicted.iloc[0])
                assert float(forecast.actual.iloc[0]) == row["spending_rub"]
                row["seasonal_forecast_rub"] = predicted
                row["seasonal_forecast_error_pct"] = 100 * (row["spending_rub"] / predicted - 1)
            rows.append(row)
    result = pd.DataFrame(rows)
    assert len(result) == 24 and not result.duplicated(["category", "month"]).any()
    result.to_csv(ROOT / "results/real_event_orsk.csv", index=False)
    print(result[result.category.isin(["Все категории", "Продовольствие"])].to_string(index=False))


if __name__ == "__main__":
    main()
