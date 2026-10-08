# Matched presentation comparison and forecast cases

Common rows use identical municipality/origin/target/horizon pairs and exactly equal saved actuals. No model was refitted.

Full growth cohort and common cached cohort are separate panels; do not rank full and common values against each other.

Direct HGB is the predeclared lags variant. At h12 it was not trainable; the seasonal fallback is separately stored and the HGB MAE cell is null.

Chronos-2 covariates means the saved univariate seasonal-profile variant, not the grouped category variant.

Pooled seasonal/profile training pools differ across research runs. Here baseline/Prophet/hierarchy predictions are the saved full-cohort growth-run forecasts, restricted to common evaluation pairs.

MAE averages within each target month then weights target months equally. Within-MO R2 = 1 - pooled prediction SSE / sum of within-municipality centered actual squares; it is not the mean municipality R2.

h12 has one target month per municipality: all within-MO R2 are undefined, represented by null. Constant-target municipality counts are explicit.

Cases are predetermined by metadata and 2023 mean expenditure, not forecast error: fixed Ufa regional capital, lowest 2023 expenditure municipal district/okrug, and fixed Orsk.

The expenditure proxy is per-consumer expense in rubles, not population or city size. Ufa is labelled regional capital; low expense is not evidence of small population.

Example intervals are NEW illustrative global scaled residual bands for the selected growth curve. They use only preceding three completed target months of that SAME saved model, 2023 scale, and an 80% rank-corrected residual quantile. Zero reporting lag is assumed.

Intervals are posthoc illustrations on reused 2024; temporal/dependent residuals and model selection preclude independent coverage guarantees. They are not copied from blend_75, and they are not a validated production uncertainty model.

Jan–Mar have insufficient three-month saved residual history: bands are absent. h1 points are rolling one-month forecasts, not a single Jan-origin 12-month forecast.

## Common cached pairs: MAE (rubles)

| Model | h1 | h3 | h6 | h12 |
| --- | --- | --- | --- | --- |
| Seasonal naive | 4226.9 | 4484.0 | 4482.6 | 4853.7 |
| Prophet: yearly disabled | 1838.7 | 1774.1 | 2213.4 | 2930.6 |
| Prophet + pooled seasonal profile | 1616.0 | 1665.0 | 1937.3 | 3002.2 |
| Prophet: yearly Fourier order 3 | 2402.4 | 2507.4 | 2935.8 | 7619.1 |
| Seasonal profile | 979.1 | 1633.0 | 1913.6 | 4853.7 |
| Hierarchy 0.5 | 927.0 | 1513.1 | 1906.7 | 4853.7 |
| Direct HGB (lags) | 2165.6 | 1251.2 | 3188.6 | not trained (fallback only) |
| Chronos-2 univariate | 2176.7 | 2734.0 | 4064.0 | 9665.0 |
| Chronos-2 + seasonal covariate | 1937.9 | 2575.9 | 3599.8 | 9125.4 |
| Selected hierarchy 0.5 + growth bridge | 921.3 | 1245.5 | 1338.1 | 1357.0 |

## Cases

- regional_capital: Уфа (ID 21), mean 2023 expense 31198.5 RUB.
- low_expense_municipality: Яльчикский (ID 502), mean 2023 expense 13604.6 RUB.
- orsk_stress_case: Орск (ID 1673), mean 2023 expense 25757.7 RUB.
