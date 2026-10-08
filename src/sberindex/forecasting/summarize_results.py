"""Compare forecasts only after blend selection was available (June 2024).

The earlier target-only cutoff allowed a June-selected weight at January origins.
All short-horizon primary comparisons now require origin >= June 2024.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from sberindex.paths import ROOT
root = ROOT
config = json.loads((root / "config.json").read_text())
raw = pd.read_parquet(root / "results" / "rolling_predictions.parquet")
sample_ids = np.sort(raw.loc[raw.model == "prophet", "territory_id"].unique())
sample = raw[raw.territory_id.isin(sample_ids)].copy()
hgb = pd.read_csv(root / "results" / "hgb_12m_predictions.csv")
hgb = hgb[hgb.territory_id.isin(sample_ids)].assign(
    origin="2023-12", horizon=12, model="global_hgb")
sample = pd.concat([sample, hgb[sample.columns]], ignore_index=True)
chronos2_path = root / "results" / "chronos2_rolling.csv"
if chronos2_path.exists():
    chronos2 = pd.read_csv(chronos2_path)
    sample = pd.concat([sample, chronos2[sample.columns]], ignore_index=True)
sample["target"] = (pd.PeriodIndex(sample.origin.to_numpy(), freq="M")
                    + sample.horizon.to_numpy()).astype(str)

# A weight chosen using only the February-June 2024 one-month validation dates.
short = sample[(sample.horizon == 1) & (sample.target <= "2024-06")
               & (sample.origin >= "2024-01")]
wide = short.pivot(index=["territory_id", "origin"], columns="model",
                   values=["actual", "predicted"])
actual = wide[("actual", "prophet")].to_numpy()
season = wide[("predicted", "seasonal_pooled")].to_numpy()
prophet = wide[("predicted", "prophet")].to_numpy()
weights = config["blend_candidate_seasonal_weights"]
validation = pd.DataFrame({"seasonal_weight": weights,
                           "MAE": [mean_absolute_error(actual, w * season + (1-w) * prophet)
                                   for w in weights]})
validation.to_csv(root / "results" / "blend_validation.csv", index=False)
weight = float(validation.loc[validation.MAE.idxmin(), "seasonal_weight"])
assert weight == 0.75

blend_rows = []
for horizon in (1, 3, 6):
    relevant = sample[(sample.horizon == horizon) & (sample.origin >= "2024-01")
                      & sample.model.isin(["prophet", "seasonal_pooled"])]
    wide = relevant.pivot(index=["territory_id", "origin", "target"],
                          columns="model", values=["actual", "predicted"])
    blend_rows.append(pd.DataFrame({
        "territory_id": wide.index.get_level_values("territory_id"),
        "origin": wide.index.get_level_values("origin"),
        "horizon": horizon, "model": "blend_75", 
        "actual": wide[("actual", "prophet")].to_numpy(),
        "predicted": weight * wide[("predicted", "seasonal_pooled")].to_numpy()
        + (1 - weight) * wide[("predicted", "prophet")].to_numpy(),
        "target": wide.index.get_level_values("target"),
    }))
sample = pd.concat([sample, *blend_rows], ignore_index=True)
primary = sample[(sample.horizon == 12) | (sample.origin >= "2024-06")].copy()
primary.to_parquet(root / "results/primary_predictions.parquet", index=False)

rows = []
for horizon in (1, 3, 6, 12):
    subset = sample[sample.horizon == horizon]
    if horizon < 12:
        subset = subset[subset.origin >= "2024-06"]
    for model, group in subset.groupby("model"):
        error = np.abs(group.actual - group.predicted)
        rows.append({"horizon": horizon, "model": model,
                     "dates": group.target.nunique(),
                     "municipalities": group.territory_id.nunique(),
                     "observations": len(group),
                     "MAE": error.mean(),
                     "R2": r2_score(group.actual, group.predicted),
                     "WAPE_pct": 100 * error.sum() / group.actual.sum()})
summary = pd.DataFrame(rows).sort_values(["horizon", "MAE"])
summary.to_csv(root / "results" / "forecast_comparison.csv", index=False)
boot_rows = []
rng = np.random.default_rng(426)
for horizon, selected_model in [(1, "blend_75"), (3, "blend_75"),
                                (6, "blend_75")]:
    part = sample[(sample.horizon == horizon) & (sample.origin >= "2024-06")
                  & sample.model.isin(["prophet", selected_model])].copy()
    part["absolute_error"] = np.abs(part.actual - part.predicted)
    monthly = part.groupby(["target", "model"]).absolute_error.mean().unstack()
    gain = (monthly.prophet - monthly[selected_model]).to_numpy()
    if len(gain) > 1:
        replicates = gain[rng.integers(0, len(gain), size=(10000, len(gain)))].mean(axis=1)
        low, high = np.quantile(replicates, [0.025, 0.975])
    else:
        low, high = np.nan, np.nan
    boot_rows.append({"horizon": horizon, "model": selected_model,
                      "months": len(gain), "mean_MAE_gain_rub": gain.mean(),
                      "month_bootstrap_95_low": low,
                      "month_bootstrap_95_high": high})
pd.DataFrame(boot_rows).to_csv(root / "results" / "bootstrap_month.csv", index=False)
print(summary.to_string(index=False))
