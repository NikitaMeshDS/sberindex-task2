# Foundation models with a known 2023 seasonal profile

Specification frozen 2026-10-05 before normalized inference or error calculation.
This is an authorized retrospective hypothesis audit of the repeatedly studied
2024 archive, not an independent validation or replacement of the primary model.

## Fixed design

Use all-category monthly spending, January 2023–December 2024. Determine eligibility
using complete 2023 alone: all 2,075 eligible IDs contribute to the pooled monthly
sum profile. Divide that 12-element profile by its arithmetic mean to obtain
positive dimensionless factors. Never use 2024 completeness to select profile IDs.
Reuse exactly the 256 sample IDs from asof_cohort_protocol.json. Every prediction
history must be complete from January 2023 through its origin; targets are scored
only when actually observed, with no interpolation or imputation.

Input to each foundation model is observed history divided by the corresponding
2023 month factor. The median prediction is multiplied by the target-month factor
and clipped below at zero, matching the saved raw forecasts' nonnegative rule.
Use amazon/chronos-bolt-tiny revision a0e552de83495b5c28c14c71c374f3e33280b340
and amazon/chronos-2 revision 29ec3766d36d6f73f0696f85560a422f50e8498c,
unchanged from config.json, offline local cache, CPU, two Torch threads, batches 64.
No fine tuning, cross learning, new hyperparameter search, or new blend weights.
Chronos-Bolt uses its 0.5 quantile (index 4); Chronos-2 requests quantile 0.5.
The installed primary package documentation confirms those API meanings.

Evaluate December 2023 origin at 12 months and June–November 2024 origins at
observable horizons 1/3/6. Raw mode is loaded from saved
asof_foundation_predictions.parquet, never rerun. Normalize only those origins.
The early 2024 dates are not required because no new mixture will be selected.
Require exact pair and actual-value equality with saved asof_cohort forecasts.
Compare raw/normalized models, seasonal_pooled, saved Prophet, existing blend
selected from early dates, and the saved regional_asof_selected mixture (1/3/6
only). Do not extrapolate the regional mixture's weight to 12 months.

Report equal-target-date MAE as the main measure; also pooled MAE, pooled R²,
WAPE, target dates, municipality count and row count. Report every target month
and paired normalized-minus-raw and normalized-minus-seasonal MAE differences.
No independence claims or municipal bootstrap significance: horizons 6/12 each
have one target date. Save per-model/origin inference wall times and model load
times. Historical raw timings were not recorded, so make no raw versus normalized
runtime claim. Reference forecasts have zero new inference cost, not zero historical
cost. Record revision identifiers and input SHA256 for provenance.

## Verification planned before helper implementation

A toy positive profile and constant deseasonalized level must analytically restore
the known seasonal level. Perturb every 2024 value, including making later history
missing, and prove profile eligibility/factors and an earlier origin's normalized
history stay identical. Verify incomplete histories are excluded using only the
origin prefix. Provide a callable verification function for project integration.
Verify exact scored pairs, actuals, nonnegative finite predictions and deterministic
recalculation of summary tables after inference. Dataset retrieval is retrospective;
reporting lag remains assumed zero, and historical model training contamination
cannot be excluded from these fixed pretrained models.
