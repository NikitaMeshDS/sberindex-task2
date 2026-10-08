# Foundation seasonal audit results

The frozen experiment in docs/protocols/FOUNDATION_SEASONAL_EXPERIMENT.md was run on 2026-10-05. Model parameters and revision hashes were fixed; no training or weight selection was performed. Saved raw forecasts were reused exactly. All seasonal information comes from the 2,075 IDs complete in 2023, regardless of later missingness.

## MAE on identical observed pairs

MAE is in the original spending units; primary column is equal target-date MAE. Here each date has the same row count, so pooled and date-balanced MAE agree.

| Horizon | Target dates | Pairs | Bolt raw | Bolt normalized | Chronos-2 raw | Chronos-2 normalized | Pooled seasonal | Regional selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1506 | 2313.79 | 848.92 | 1951.75 | 833.32 | 801.03 | 867.89 |
| 3 | 4 | 1004 | 3025.98 | 1917.94 | 2363.13 | 1442.13 | 1108.47 | 1198.08 |
| 6 | 1 | 251 | 9579.41 | 3328.40 | 7114.14 | 2255.58 | 1648.40 | 1079.17 |
| 12 | 1 | 252 | 11059.88 | 5122.16 | 9664.99 | 5198.05 | 4853.75 | not extrapolated |

Normalization substantially reduces raw-model MAE at every horizon, but neither normalized foundation model beats the pooled seasonal baseline on horizon-level MAE. The regional mixture has lower six-month MAE than either normalized model. At 12 months, saved HGB MAE is 2528.69 and saved Prophet MAE is 2930.61; no primary model change is supported. The full summary includes pooled R² and WAPE; a high R² does not override a larger MAE.

## Per-date contrasts

- chronos_bolt_tiny_seasonal versus chronos_bolt_tiny: lower MAE in 9 of 12 horizon/date cells; higher in 3. Every difference is disclosed in foundation_seasonal_paired_monthly.csv.
- chronos_bolt_tiny_seasonal versus seasonal_pooled: lower MAE in 2 of 12 horizon/date cells; higher in 10. Every difference is disclosed in foundation_seasonal_paired_monthly.csv.
- chronos_2_seasonal versus chronos_2: lower MAE in 7 of 12 horizon/date cells; higher in 5. Every difference is disclosed in foundation_seasonal_paired_monthly.csv.
- chronos_2_seasonal versus seasonal_pooled: lower MAE in 3 of 12 horizon/date cells; higher in 9. Every difference is disclosed in foundation_seasonal_paired_monthly.csv.

Cells overlap calendar dates and forecast contexts; these counts are descriptive, not independent replications.

## Runtime and verification

- chronos_2: model load 0.079s; seven origin calls 4.161s total. CPU, Torch threads 2, batch 64, cached offline weights.
- chronos_bolt_tiny: model load 0.109s; seven origin calls 0.339s total. CPU, Torch threads 2, batch 64, cached offline weights.

Measured times cover model loading and normalized inference including restoration, not complete process startup/data reads/scoring. Historical raw runtimes are unavailable, so there is no raw-versus-normalized speed claim. Cached weight loading is unusually fast and depends on machine and filesystem cache.

Both callable invariant and artifact checks passed. The latter recomputes profile/metrics, checks provenance hashes, all model pairs and actuals, exact raw predictions, finite nonnegative forecasts, and unchanged June history after future perturbation. Full unittest discovery initially failed because the separately developed detector module was not yet present. After that module became available, the complete current unittest suite was rerun: all four DetectorAvailabilityTests passed.

## Limits and integration

This is the repeatedly studied 2024 archive. Six- and twelve-month horizons each have one target date; 251/252 observed municipalities do not create independent temporal evidence. Model pretraining overlap or contamination cannot be excluded. Only complete histories are forecast, only observed targets are scored, and reporting delay is assumed zero. Geography for the saved regional reference was retrieved retrospectively; its original caveat remains. No model is selected using the new late errors.

Run `PYTHONPATH=src HF_HUB_OFFLINE=1 ../venv/bin/python -m sberindex.forecasting.foundation_seasonal_audit` for cached normalized inference, or use `--verify` for toy invariants. For pipeline verification call `verify_foundation_seasonal_artifacts()` from the module; this requires no Torch/Chronos model import or inference. It returns nine named checks, including real-data future invariance. Use `--refresh` or call `refresh()` to regenerate the three metric CSV tables from hash-protected cached predictions/protocol/runtime without loading models. Integrate the module and new foundation_seasonal_* files explicitly; shared pipeline/manifest/status files were not edited by this worker.
