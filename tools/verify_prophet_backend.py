import hashlib, json, logging
from pathlib import Path
import numpy as np, pandas as pd
from prophet import Prophet
from sberindex.forecasting.prophet_backend import legacy_predict
from sberindex.paths import ROOT

logging.getLogger("cmdstanpy").disabled = True
raw = pd.read_parquet(ROOT / "data/consumption.parquet")
panel = raw[raw.category == "Все категории"].pivot(
    index="territory_id", columns="date", values="value"
)
months = pd.date_range("2023-01-01", periods=24, freq="MS")
rows = []
for city in [12, 21, 27]:
    if city not in panel.index or not np.isfinite(panel.loc[city]).all():
        continue
    for origin, h in [(11, 12), (17, 6), (22, 1)]:
        history = panel.loc[city].to_numpy(float)[: origin + 1]
        old = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        old.fit(pd.DataFrame({"ds": months[: origin + 1], "y": history}))
        reference = old.predict(
            pd.DataFrame({"ds": months[origin + 1 : origin + 1 + h]})
        ).yhat.to_numpy()
        result = legacy_predict(
            history, months[: origin + 1], months[origin + 1 : origin + 1 + h]
        )
        np.testing.assert_array_equal(result, reference)
        rows.append(
            dict(
                territory_id=city,
                origin=str(months[origin].to_period("M")),
                horizon=h,
                maximum_absolute_difference=float(np.max(np.abs(result - reference))),
            )
        )
paths = [
    "src/sberindex/forecasting/prophet_backend.py",
    "src/sberindex/forecasting/benchmark_rolling.py",
    "src/sberindex/forecasting/category_benchmark.py",
    "src/sberindex/forecasting/asof_cohort_forecast.py",
    "src/sberindex/forecasting/asof_reporting_delay.py",
    "src/sberindex/forecasting/geographic_extension.py",
    "src/sberindex/forecasting/operational_early_audit.py",
    "tools/operational_workflow.py",
]
out = ROOT / "reports/prophet_backend_review"
out.mkdir(exist_ok=True)
(out / "regression.json").write_text(
    json.dumps(
        dict(
            status="passed",
            historical_settings_unchanged=True,
            comparison="fresh original constructor versus shared backend; exact arrays before clipping",
            cases=rows,
            source_sha256={
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
            },
        ),
        indent=2,
    )
    + "\n"
)
print(rows)
