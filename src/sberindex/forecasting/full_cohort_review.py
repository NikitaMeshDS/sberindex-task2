"""All-2023-eligible municipality comparison; separately frozen review extension."""
import concurrent.futures
import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sberindex.forecasting.prophet_seasonality import fit_predict
from sberindex.paths import ROOT


def eligible_ids(panel):
    """Use only the twelve 2023 observations, never survival in 2024."""
    history = panel.iloc[:, :12]
    return history.index[np.isfinite(history).all(axis=1) & (history > 0).all(axis=1)]


def evaluation_pairs(values, horizons):
    """Return common evaluable origin/horizon keys plus exclusive drop counts."""
    pairs = []
    reasons = {'missing_history': 0, 'missing_target': 0, 'retained': 0}
    for origin in range(11, 23):
        for horizon in horizons:
            target = origin + horizon
            if target >= 24:
                continue
            if not (np.isfinite(values[:origin + 1]).all() and (values[:origin + 1] > 0).all()):
                reasons['missing_history'] += 1
            elif not np.isfinite(values[target]) or values[target] <= 0:
                reasons['missing_target'] += 1
            else:
                pairs.append((origin, horizon))
                reasons['retained'] += 1
    return pairs, reasons


def refit(job):
    city, origin, history, profile, horizon, config = job
    logging.getLogger('cmdstanpy').disabled = True
    return (city, origin), {
        variant: fit_predict(history, profile, horizon, variant, config)
        for variant in config['models']
    }


def summarize(result):
    stats = []
    for (horizon, model), group in result.groupby(['horizon', 'model']):
        within = group.actual - group.groupby('territory_id').actual.transform('mean')
        denominator = float((within ** 2).sum())
        squared_error = float(((group.actual - group.predicted) ** 2).sum())
        stats.append(dict(
            horizon=horizon, model=model, dates=group.target.nunique(),
            municipalities=group.territory_id.nunique(), observations=len(group),
            MAE=group.groupby('target').ae.mean().mean(),
            R2_pooled=r2_score(group.actual, group.predicted),
            R2_within_MO=1 - squared_error / denominator if denominator > 0 else np.nan,
            YoY_MAE_pp=group.groupby('target').yoy_ae.mean().mean(),
            YoY_R2=r2_score(group.yoy_actual, group.yoy_predicted),
            MoM_R2=r2_score(group.mom_actual, group.mom_predicted) if horizon == 1 else np.nan,
        ))
    summary = pd.DataFrame(stats)
    for reference in ['seasonal_naive', 'seasonal_pooled']:
        reference_mae = summary[summary.model == reference].set_index('horizon').MAE
        summary['skill_vs_' + reference] = 1 - summary.MAE / summary.horizon.map(reference_mae)
    return summary


def main():
    started = time.monotonic()
    config = json.loads((ROOT / 'configs/full_cohort_review.json').read_text())
    raw = pd.read_parquet(ROOT / 'data/consumption.parquet')
    panel = raw[raw.category == 'Все категории'].pivot(index='territory_id', columns='date', values='value')
    panel = panel.reindex(columns=pd.period_range('2023-01', '2024-12', freq='M').astype(str)).sort_index()
    ids = eligible_ids(panel)
    sums = panel.loc[ids].iloc[:, :12].sum(axis=0).to_numpy(float)
    profile = sums / sums.mean()
    coverage, pairs, jobs = [], {}, []
    for city in ids:
        values = panel.loc[city].to_numpy(float)
        retained, reasons = evaluation_pairs(values, config['horizons'])
        coverage.append(dict(territory_id=int(city), **reasons))
        for origin in sorted({origin for origin, _ in retained}):
            horizons = [horizon for o, horizon in retained if o == origin]
            pairs[(int(city), origin)] = horizons
            jobs.append((int(city), origin, values[:origin + 1], profile, max(horizons), config))
    out = ROOT / 'reports/full_cohort_review'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures = [pool.submit(refit, job) for job in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            (city, origin), predictions = future.result()
            values = panel.loc[city].to_numpy(float)
            for horizon in pairs[(city, origin)]:
                target = origin + horizon
                info = dict(territory_id=city, origin=str(panel.columns[origin]),
                            target=str(panel.columns[target]), horizon=horizon,
                            actual=values[target], year_ago=values[target - 12], origin_actual=values[origin])
                forecasts = {
                    'seasonal_pooled': values[origin] * profile[target % 12] / profile[origin % 12],
                    'seasonal_naive': values[target - 12],
                    **{model: float(pred[horizon - 1]) for model, pred in predictions.items()},
                }
                for model, predicted in forecasts.items():
                    rows.append(dict(**info, model=model, predicted=predicted))
            if completed % 1000 == 0 or completed == len(jobs):
                print(f'Full cohort: {completed}/{len(jobs)} ID/origins; {time.monotonic()-started:.1f}s', flush=True)
    result = pd.DataFrame(rows).sort_values(['territory_id', 'origin', 'horizon', 'model']).reset_index(drop=True)
    result['ae'] = (result.actual - result.predicted).abs()
    result['yoy_actual'] = 100 * (result.actual / result.year_ago - 1)
    result['yoy_predicted'] = 100 * (result.predicted / result.year_ago - 1)
    result['yoy_ae'] = (result.yoy_actual - result.yoy_predicted).abs()
    result['mom_actual'] = np.where(result.horizon == 1, 100 * (result.actual / result.origin_actual - 1), np.nan)
    result['mom_predicted'] = np.where(result.horizon == 1, 100 * (result.predicted / result.origin_actual - 1), np.nan)
    result.to_parquet(out / 'predictions.parquet', index=False)
    summary = summarize(result)
    summary.to_csv(out / 'summary.csv', index=False)
    result.groupby(['horizon', 'model', 'target']).agg(MAE=('ae', 'mean'), observations=('ae', 'size')).reset_index().to_csv(out / 'monthly.csv', index=False)
    names = ['data/consumption.parquet', 'configs/full_cohort_review.json',
             'docs/protocols/FULL_COHORT_REVIEW.md', 'src/sberindex/forecasting/full_cohort_review.py',
             'src/sberindex/forecasting/prophet_seasonality.py']
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit = dict(status='passed', eligible_2023=len(ids), source_ids=len(panel),
                 cohort_excluded_2023=len(panel)-len(ids), new_prophet_fits=len(jobs)*len(config['models']),
                 profile=profile.tolist(), reporting_lag_assumption=0, independent_time_validation=False,
                 parameters_tuned=False, seconds=time.monotonic()-started,
                 coverage_totals=pd.DataFrame(coverage).drop(columns='territory_id').sum().to_dict(),
                 input_sha256={name: sha(ROOT/name) for name in names},
                 output_sha256={name: sha(out/name) for name in ['coverage.csv', 'predictions.parquet', 'summary.csv', 'monthly.csv']})
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(summary[['horizon', 'model', 'dates', 'municipalities', 'MAE', 'YoY_R2']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
