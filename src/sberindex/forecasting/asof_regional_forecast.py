"""Regional seasonal mixtures on the 2023-known cohort; reused 2024 audit.

Reuses the previous 25 candidates and saved Prophet forecasts. Selection uses
five early target dates with equal date weight; late data never choose a model.
The source geography is the 2023 dictionary version, retrieved retrospectively.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sberindex.paths import ROOT
from sberindex.forecasting.regional_seasonality import CANDIDATES, ratios

OUT = ROOT / 'results'
KEYS = ['territory_id', 'origin', 'target', 'horizon']


def seasonal_predictions(panel, regions, forecasts, candidate):
    """All profile values are restricted to 2023; anchor is at the origin."""
    values = panel.to_numpy(float)
    positions = pd.Series(np.arange(len(panel)), index=panel.index)
    result = pd.Series(index=forecasts.index, dtype=float)
    for (origin, target), group in forecasts.groupby(['origin', 'target']):
        o = list(panel.columns).index(origin)
        t = list(panel.columns).index(target)
        idx = positions.loc[group.territory_id].to_numpy()
        result.loc[group.index] = values[idx, o] * ratios(values, regions, o, t, candidate)[idx]
    assert np.isfinite(result).all()
    return result


def metrics(group):
    errors = abs(group.actual-group.predicted)
    return dict(MAE_date_balanced=errors.groupby(group.target).mean().mean(),
        MAE_pooled=errors.mean(), R2_pooled=r2_score(group.actual, group.predicted),
        WAPE_pct=100*errors.sum()/group.actual.sum(), dates=group.target.nunique(),
        municipalities=group.territory_id.nunique(), observations=len(group))


def main():
    source = pd.read_parquet(ROOT/'data/consumption.parquet')
    panel = source[source.category == 'Все категории'].pivot(
        index='territory_id', columns='date', values='value').sort_index()
    panel = panel.loc[panel.iloc[:, :12].notna().all(axis=1)]
    lookup = pd.read_csv(OUT/'municipal_lookup.csv')
    lookup = lookup[lookup.year == 2023].set_index('territory_id')
    regions = lookup.loc[panel.index, 'region_code'].to_numpy()
    raw = pd.read_parquet(OUT/'asof_cohort_predictions.parquet')
    base = raw[(raw.model == 'prophet') & raw.horizon.isin([1, 3, 6])].copy()
    base = base.sort_values(KEYS).reset_index(drop=True)
    weights = json.loads((ROOT/'config.json').read_text())['blend_candidate_seasonal_weights']
    early = (base.horizon == 1) & base.target.between('2024-02', '2024-06')
    late = base.origin.ge('2024-06')
    assert base[early].target.nunique() == 5
    validation = []
    forecasts = {}
    all_scores = []
    for candidate in CANDIDATES:
        seasonal = seasonal_predictions(panel, regions, base, candidate)
        for weight in weights:
            prediction = weight*seasonal + (1-weight)*base.predicted
            forecasts[(candidate, weight)] = prediction
            trial = base.copy()
            trial['predicted'] = prediction
            validation.append(dict(candidate=candidate, seasonal_weight=weight,
                                   **metrics(trial[early])))
            for horizon, group in trial[late].groupby('horizon'):
                all_scores.append(dict(candidate=candidate, seasonal_weight=weight,
                                       horizon=int(horizon), **metrics(group)))
    validation = pd.DataFrame(validation)
    best = validation.loc[validation.MAE_date_balanced.idxmin()]
    selected = base[late].copy()
    selected['predicted'] = forecasts[(best.candidate, best.seasonal_weight)].loc[selected.index]
    selected['model'] = 'regional_asof_selected'
    selected = selected[KEYS+['model', 'actual', 'predicted']]
    selected.to_parquet(OUT/'asof_regional_predictions.parquet', index=False)
    validation.to_csv(OUT/'asof_regional_validation.csv', index=False)
    pd.DataFrame(all_scores).to_csv(OUT/'asof_regional_all_candidates.csv', index=False)
    original = pd.read_parquet(OUT/'asof_cohort_scored.parquet')
    original = original[original.horizon.isin([1, 3, 6])]
    pairs = selected[KEYS+['actual']]
    original = original.merge(pairs, on=KEYS, validate='many_to_one', suffixes=('', '_check'))
    np.testing.assert_allclose(original.actual, original.actual_check)
    compare = pd.concat([original[KEYS+['model', 'actual', 'predicted']], selected], ignore_index=True)
    counts = compare.groupby(['horizon', 'model']).size()
    assert counts.groupby(level='horizon').nunique().eq(1).all()
    summary = [dict(horizon=int(h), model=m, **metrics(g))
               for (h, m), g in compare.groupby(['horizon', 'model'])]
    pd.DataFrame(summary).to_csv(OUT/'asof_regional_comparison.csv', index=False)
    compare['ae'] = abs(compare.actual-compare.predicted)
    compare.groupby(['horizon', 'model', 'target']).agg(
        MAE=('ae', 'mean'), municipalities=('territory_id', 'nunique')).to_csv(
            OUT/'asof_regional_monthly.csv')
    protocol = dict(eligible_ids=len(panel), profile_year=2023, geography_year=2023,
        candidates=CANDIDATES, weights=weights, candidate_count=len(validation),
        regional_minimum_n=10, selected_candidate=best.candidate,
        selected_seasonal_weight=float(best.seasonal_weight),
        validation_MAE_date_balanced=float(best.MAE_date_balanced),
        selection='Five one-month target dates February-June 2024; equal target-date weights; first minimum in candidate/weight order',
        sample='Saved 256 IDs selected using 2023 completeness; identical saved Prophet pairs, observed history and targets',
        evaluation='Origins June-November 2024, horizons 1/3/6; no selection on later errors; all 25 candidates disclosed',
        missingness='Saved Prophet skips incomplete full histories or missing targets; identical scored pairs in every model',
        limits='Repeatedly explored archive, not independent holdout. Historical dictionary publication vintages unknown. Reporting lag assumed zero. No 12-month extrapolation of a one-month-selected regional mixture. Primary model unchanged.')
    (OUT/'asof_regional_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(validation.sort_values('MAE_date_balanced').head(5).to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
