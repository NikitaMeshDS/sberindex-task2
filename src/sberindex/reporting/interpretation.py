"""Descriptive nominal/real growth on the fixed complete municipal panel.

All CPI adjustments are retrospective. Medians of municipality-month observations
are not estimates of population-weighted national expenditure growth.
"""
from pathlib import Path
import pandas as pd

from sberindex.paths import ROOT
def main():
    data = pd.read_parquet(ROOT / "results/consumption_with_geography_cpi.parquet")
    complete = data[data.category == "Все категории"].groupby("territory_id").date.nunique()
    ids = complete[complete == 24].index
    data = data[data.territory_id.isin(ids) & data.category.isin(["Все категории", "Продовольствие"])].copy()
    lag = data[["territory_id", "date", "category", "value", "real_value_dec2022_prices"]].copy()
    lag["date"] = (pd.PeriodIndex(lag.date, freq="M") + 12).astype(str)
    lag = lag.rename(columns={"value": "lag12", "real_value_dec2022_prices": "real_lag12"})
    growth = data.merge(lag, on=["territory_id", "date", "category"], validate="one_to_one")
    growth["nominal_yoy_pct"] = 100 * (growth.value / growth.lag12 - 1)
    growth["real_yoy_pct"] = 100 * (growth.real_value_dec2022_prices / growth.real_lag12 - 1)
    fields = ["territory_id", "date", "category", "region_code", "region_name",
              "municipal_district_name_short", "value", "real_value_dec2022_prices",
              "nominal_yoy_pct", "real_yoy_pct"]
    growth[fields].to_parquet(ROOT / "results/real_growth_interpretation.parquet", index=False)
    summary = growth.groupby("category")[["nominal_yoy_pct", "real_yoy_pct"]].median()
    summary["municipalities"] = len(ids)
    summary["observations_per_category"] = len(ids) * 12
    summary.to_csv(ROOT / "results/real_growth_summary.csv")
    region = growth.groupby(["category", "region_code", "region_name"]).agg(
        nominal_yoy_pct=("nominal_yoy_pct", "median"), real_yoy_pct=("real_yoy_pct", "median"),
        municipalities=("territory_id", "nunique"))
    region.to_csv(ROOT / "results/region_real_growth.csv")
    cases = growth[(growth.category == "Все категории") & growth.territory_id.isin([2593, 2595])].copy()
    cases["period"] = cases.date.map(lambda x: "Jan-Jun" if x < "2024-07" else "Jul-Dec")
    cases.groupby(["territory_id", "municipal_district_name_short", "period"])[
        ["nominal_yoy_pct", "real_yoy_pct"]].median().to_csv(ROOT / "results/named_case_growth.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
