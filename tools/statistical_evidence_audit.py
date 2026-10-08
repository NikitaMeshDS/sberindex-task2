"""Descriptive evidence audit of saved forecasts; no fitting or significance test.

Run separately: python tools/statistical_evidence_audit.py
Output is an appendix, not a new stage of the frozen 51-stage pipeline.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'reports/statistical_evidence'
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = ['results/primary_predictions.parquet', 'results/forecast_comparison.csv',
          'results/bootstrap_month.csv', 'results/online_interval_summary.csv',
          'reports/operational_workflow/detector_choice.json']
p = pd.read_parquet(ROOT / INPUTS[0])
assert not p.duplicated(['territory_id', 'origin', 'horizon', 'model']).any()
p['ae'] = (p.actual - p.predicted).abs()
summary = pd.read_csv(ROOT / INPUTS[1])
rows, monthly_rows, deletion_rows = [], [], []
for h, candidate in [(1, 'blend_75'), (3, 'blend_75'), (6, 'blend_75'), (12, 'global_hgb')]:
    for reference in ['prophet', 'seasonal_pooled']:
        part = p[(p.horizon == h) & p.model.isin([candidate, reference])]
        keys = ['territory_id', 'origin', 'target']
        actual = part.pivot(index=keys, columns='model', values='actual')
        errors = part.pivot(index=keys, columns='model', values='ae')
        assert not errors.isna().any().any()
        assert np.array_equal(actual[candidate].to_numpy(), actual[reference].to_numpy())
        by_date = errors.groupby(level='target').mean().sort_index()
        gain = by_date[reference] - by_date[candidate]
        for model in [candidate, reference]:
            stored = summary[(summary.horizon == h) & (summary.model == model)].iloc[0]
            assert np.isclose(errors[model].mean(), stored.MAE, rtol=0, atol=1e-8)
            assert len(errors) == stored.observations
        label = dict(horizon=h, candidate=candidate, reference=reference)
        for target, g in gain.items():
            monthly_rows.append(dict(**label, target=target,
                candidate_MAE=by_date.loc[target, candidate], reference_MAE=by_date.loc[target, reference],
                gain_rub=g, paired_municipalities=len(errors.xs(target, level='target'))))
        for delete_n in [1, 2]:
            if len(gain) <= delete_n:
                continue
            for removed in combinations(gain.index, delete_n):
                remaining = gain.drop(list(removed))
                deletion_rows.append(dict(**label, removed_count=delete_n,
                    removed_targets=';'.join(removed), remaining_dates=len(remaining), gain_rub=remaining.mean()))
        without_december = by_date.drop('2024-12', errors='ignore')
        rows.append(dict(**label, dates=len(gain), observations=len(errors),
            gain_rub=gain.mean(), gain_pct=100*gain.mean()/by_date[reference].mean(),
            median_date_gain_rub=gain.median(), positive_dates=int((gain > 0).sum()),
            minimum_date_gain_rub=gain.min(), maximum_date_gain_rub=gain.max(),
            december_share_of_net_gain=(gain.get('2024-12', np.nan)/gain.sum() if reference == 'prophet' else np.nan),
            without_december_gain_rub=(without_december[reference]-without_december[candidate]).mean(),
            without_december_gain_pct=(100*(without_december[reference]-without_december[candidate]).mean()/without_december[reference].mean())))
monthly = pd.DataFrame(monthly_rows)
monthly.to_csv(OUT/'monthly.csv', index=False)
pd.DataFrame(rows).to_csv(OUT/'summary.csv', index=False)
pd.DataFrame(deletion_rows).to_csv(OUT/'date_deletions.csv', index=False)
# Recompute the legacy bootstrap verbatim from monthly paired errors, preserving
# RNG order. This checks arithmetic, not the iid-month assumption or selection.
rng = np.random.default_rng(426)
boot = pd.read_csv(ROOT/INPUTS[2])
for h in [1, 3, 6]:
    gain = monthly[(monthly.horizon == h) & (monthly.reference == 'prophet')].gain_rub.to_numpy()
    stored = boot[boot.horizon == h].iloc[0]
    assert np.isclose(gain.mean(), stored.mean_MAE_gain_rub, rtol=0, atol=1e-8)
    if len(gain) > 1:
        simulations = gain[rng.integers(0,len(gain),size=(10000,len(gain)))].mean(axis=1)
        assert np.allclose(np.quantile(simulations,[.025,.975]),
            [stored.month_bootstrap_95_low, stored.month_bootstrap_95_high],rtol=0,atol=1e-8)
    else:
        assert pd.isna(stored.month_bootstrap_95_low) and pd.isna(stored.month_bootstrap_95_high)
inputs = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in INPUTS}
outputs = {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
           for name in ['monthly.csv','summary.csv','date_deletions.csv']}
(OUT/'audit.json').write_text(json.dumps(dict(
    status='arithmetic_verified', input_sha256=inputs, output_sha256=outputs,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    legacy_bootstrap_recomputed=True, models_refitted=False, significance_test_performed=False,
    independent_validation_performed=False,
    scope='Post-selection primary window; already studied 2024; h12 is a separate HGB comparison.',
    limits='Date deletions and positive-date counts are descriptive, not confidence intervals. Municipalities share calendar shocks; overlapping horizons induce temporal dependence. Existing iid-month bootstrap ignores this and model-selection uncertainty. One target at h6/h12 cannot establish time generalization.'
),ensure_ascii=False,indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
