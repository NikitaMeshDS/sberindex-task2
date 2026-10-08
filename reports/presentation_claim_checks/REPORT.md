# Presentation claim checks

Saved results only; no new model, threshold or detector experiment. These are retrospective checks, not independent validation.

## Continuous growth claim

With k in {0,1}, each prediction is affine in g. Absolute error is convex; positive date-balanced averaging preserves convexity. For every g in [0.07,0.17], MAE(g) <= max(MAE(0.07),MAE(0.17)). Each endpoint maximum is strictly below the minimum MAE among three saved Prophet variants for that horizon.

| h | MAE at 7% | MAE at 17% | best tested Prophet | MAE bound | strict margin |
|---|---:|---:|---|---:|---:|
| 1 | 851.20 | 928.56 | prophet_pooled_profile | 928.56 | 666.03 |
| 3 | 1346.00 | 1269.43 | prophet_pooled_profile | 1346.00 | 313.50 |
| 6 | 1652.95 | 1372.35 | prophet_pooled_profile | 1652.95 | 263.70 |
| 12 | 2558.16 | 1290.21 | prophet_disabled | 2558.16 | 347.64 |

Retrospective diagnostic on existing 2024 data; h12 has only one origin and one target date. No significance or transfer guarantee.

## Orsk

May residual: 0.078612658 log units = 7.8613 log-percent; exponentiation gives 8.1785% excess in the multiplicative residual ratio. The upstream calculation uses log(actual/prediction), so exponentiation expresses the actual/prediction ratio relative to the median regional peer residual; it is not a YoY percentage-point gap.

| Population | N | signed descending rank | absolute descending rank | signed top% | absolute top% |
|---|---:|---:|---:|---:|---:|
| may_available | 2012 | 16 | 46 | 0.795 | 2.286 |
| complete_2024_peer_residual | 1998 | 15 | 41 | 0.751 | 2.052 |

Rank ties use minimum competition rank. 1998 denotes municipalities with finite peer residuals in all 12 months of 2024. 2012 denotes available May observations. Rank 39/1998 is not reproduced. A top 2% description holds for signed positive ranks above, but not for absolute ranks.

Approximately +7 pp is the selected May–July Orsk–Orenburg YoY gap; +7.9 log-percent is the May regional peer residual. Do not conflate them.

The chosen May–July gaps are [9.193815741510036, 6.231773777493309, 6.109093137930401]; mean 7.1782 pp. Neither city triggers any of the four frozen own/peer detectors in 2024.

Orenburg also flooded; selected window and retrospective residuals do not establish a causal flood or payments effect.

## All eight saved short-history detectors

| Split | Method | Mode | boundary F1 | onset F1 | onset recall | null FAR/100 | onset delay | matched denominator | misses excluded |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| selection | bocpd | online | 0.1097 | 0.1597 | 0.0875 | 0.1389 | 0.0000 | 21 | 219 |
| selection | cusum | online | 0.5088 | 0.6019 | 0.5417 | 0.1389 | 0.7615 | 130 | 110 |
| selection | cusum_spike | online | 0.5168 | 0.6157 | 0.5542 | 0.1389 | 0.3609 | 133 | 107 |
| selection | ewma | online | 0.5151 | 0.6393 | 0.5833 | 0.0694 | 0.5286 | 140 | 100 |
| selection | kernelcpd | offline | 0.1432 | 0.2066 | 0.1167 | 0.0000 | unavailable | 0 | 212 |
| selection | pelt | offline | 0.7069 | 0.7225 | 0.5750 | 0.0694 | unavailable | 0 | 102 |
| selection | rolling_3m | online | 0.3180 | 0.3641 | 0.2958 | 0.0694 | 0.9859 | 71 | 169 |
| selection | spike | online | 0.4308 | 0.5158 | 0.4750 | 0.0694 | 0.2018 | 114 | 126 |
| evaluation | bocpd | online | 0.0625 | 0.0921 | 0.0486 | 0.0926 | 0.0286 | 35 | 685 |
| evaluation | cusum | online | 0.5100 | 0.6016 | 0.5389 | 0.0694 | 0.7680 | 388 | 332 |
| evaluation | cusum_spike | online | 0.5044 | 0.5983 | 0.5389 | 0.0926 | 0.3763 | 388 | 332 |
| evaluation | ewma | online | 0.5323 | 0.6471 | 0.5833 | 0.0463 | 0.5429 | 420 | 300 |
| evaluation | kernelcpd | offline | 0.1328 | 0.1880 | 0.1042 | 0.0231 | unavailable | 0 | 645 |
| evaluation | pelt | offline | 0.6975 | 0.7219 | 0.5806 | 0.1389 | unavailable | 0 | 302 |
| evaluation | rolling_3m | online | 0.3329 | 0.3443 | 0.2750 | 0.0694 | 0.9899 | 198 | 522 |
| evaluation | spike | online | 0.4425 | 0.5254 | 0.4958 | 0.0694 | 0.2577 | 357 | 363 |

Best eligible online onset F1 on saved short-history selection; not globally best boundary F1, not fastest online, and not independently validated on real shifts.

Only ±1-month matched first onsets; max(0,predicted-onset). Missed onsets excluded. Boundary F1 also counts pulse return; onset F1 neutralizes predictions matching return. Offline methods receive full sequence and have no online onset-delay estimate.

False alarms on synthetic no-change series / no-change MO-months ×100. Real municipal alert burden is not FAR; real ground truth unavailable.

Saved event_metric_review is a different 36+24-month boundary profile and selects no online method; do not combine its metrics with short-history selection.
