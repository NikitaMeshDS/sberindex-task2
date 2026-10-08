"""Distribution diagnostics on frozen as-of forecasts; no refit or selection."""
import hashlib
import json

import numpy as np
import pandas as pd

from sberindex.paths import ROOT

KEYS = ['territory_id', 'origin', 'target', 'horizon']
TOL = 1e-9
PREFIX = 'asof_distribution_'
INPUTS = ['data/consumption.parquet', 'results/municipal_lookup.csv',
          'results/foundation_seasonal_predictions.parquet',
          'results/calendar_predictions.parquet',
          'docs/protocols/ASOF_DISTRIBUTION_AUDIT.md']


def training_metadata(panel):
    dates = pd.period_range('2023-01', periods=12, freq='M').astype(str)
    train = panel.loc[:, dates]
    valid = np.isfinite(train.to_numpy(float)).all(axis=1) & train.gt(0).all(axis=1)
    means = train.loc[valid].mean(axis=1)
    cuts = np.quantile(means, [.25, .5, .75])
    result = means.rename('expense_2023').to_frame()
    result['expense_quartile'] = ['Q'+str(i+1) for i in np.searchsorted(cuts, means, side='left')]
    return result, cuts


def metrics(frame):
    ae = (frame.predicted-frame.actual).abs()
    municipal = ae.groupby(frame.territory_id).mean()
    top_n = int(np.ceil(.05*len(municipal)))
    total = municipal.sum()
    return {'MAE_date_balanced': float(ae.groupby(frame.target).mean().mean()),
            'MAE_pooled': float(ae.mean()),
            'bias_date_balanced_rub': float((frame.predicted-frame.actual).groupby(frame.target).mean().mean()),
            'NMAE_2023_pct': float(100*(ae/frame.expense_2023).groupby(frame.target).mean().mean()),
            'WAPE_pct': float(100*ae.sum()/frame.actual.sum()),
            'median_AE_rub': float(ae.median()), 'p90_AE_rub': float(ae.quantile(.9)),
            'median_municipal_MAE_rub': float(municipal.median()),
            'p90_municipal_MAE_rub': float(municipal.quantile(.9)),
            'top5pct_municipal_error_share': float(municipal.nlargest(top_n).sum()/total) if total else 0.,
            'top5pct_municipalities': top_n, 'municipalities': len(municipal),
            'dates': frame.target.nunique(), 'observations': len(frame)}


def match_errors(candidate, reference):
    assert not candidate.duplicated(KEYS).any()
    assert not reference.duplicated(KEYS).any()
    left = candidate.sort_values(KEYS).reset_index(drop=True)
    right = reference.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[KEYS], right[KEYS])
    np.testing.assert_array_equal(left.actual, right.actual)
    result = left.copy()
    result['reference_ae'] = (right.predicted-right.actual).abs()
    result['candidate_ae'] = (left.predicted-left.actual).abs()
    result['difference'] = result.candidate_ae-result.reference_ae
    return result


def paired_metrics(pairs):
    municipal = pairs.groupby('territory_id').difference.mean()
    dates = pairs.groupby('target').difference.mean()
    return {'MAE_difference_rub': float(dates.mean()),
            'observation_win_fraction': float((pairs.difference < -TOL).mean()),
            'observation_tie_fraction': float((pairs.difference.abs() <= TOL).mean()),
            'municipal_win_fraction': float((municipal < -TOL).mean()),
            'municipalities_better': int((municipal < -TOL).sum()),
            'municipalities_worse': int((municipal > TOL).sum()),
            'municipalities_tied': int((municipal.abs() <= TOL).sum()),
            'median_municipal_difference_rub': float(municipal.median()),
            'p90_municipal_difference_rub': float(municipal.quantile(.9)),
            'dates_better': int((dates < -TOL).sum()), 'dates_worse': int((dates > TOL).sum()),
            'dates_tied': int((dates.abs() <= TOL).sum()),
            'dates': len(dates), 'municipalities': len(municipal), 'observations': len(pairs)}



def validate_actuals(predictions, source):
    targets = predictions.merge(source[['territory_id','date','value']],
        left_on=['territory_id','target'],right_on=['territory_id','date'],
        how='left',validate='many_to_one',indicator='_source_match')
    assert len(targets) == len(predictions)
    assert targets._source_match.eq('both').all(), 'Forecast actual absent from source'
    assert np.isfinite(targets.value.to_numpy(float)).all(), 'Source actual is unobserved'
    np.testing.assert_array_equal(targets.actual,targets.value)


def load_inputs(root=ROOT):
    raw = pd.read_parquet(root/'data/consumption.parquet')
    panel = raw[raw.category.eq('Все категории')].pivot(index='territory_id',columns='date',values='value').sort_index()
    metadata, cuts = training_metadata(panel)
    lookup = pd.read_csv(root/'results/municipal_lookup.csv')
    lookup = lookup[lookup.year.eq(2023)].set_index('territory_id')
    metadata = metadata.join(lookup[['region_code','region_name','municipal_district_name_short']],validate='one_to_one')
    foundation = pd.read_parquet(root/'results/foundation_seasonal_predictions.parquet')
    calendar = pd.read_parquet(root/'results/calendar_predictions.parquet')
    calendar = calendar[calendar.model.isin(['hgb_frozen','hgb_frozen_calendar']) &
                        (calendar.horizon.eq(12) | calendar.origin.ge('2024-06'))]
    old_hgb = foundation[foundation.model.eq('global_hgb_asof')]
    new_hgb = calendar[calendar.model.eq('hgb_frozen') & calendar.horizon.eq(12)]
    match_errors(old_hgb,new_hgb)
    np.testing.assert_allclose(old_hgb.sort_values(KEYS).predicted,new_hgb.sort_values(KEYS).predicted,rtol=1e-10)
    foundation = foundation[~foundation.model.eq('global_hgb_asof')]
    columns = KEYS+['model','actual','predicted']
    frame = pd.concat([foundation[columns],calendar[columns]],ignore_index=True)
    assert not frame.duplicated(KEYS+['model']).any()
    assert (frame.horizon.eq(12) | frame.origin.ge('2024-06')).all()
    assert np.isfinite(frame[['actual','predicted']].to_numpy(float)).all()
    for horizon, group in frame.groupby('horizon'):
        baseline = group[group.model.eq('seasonal_pooled')]
        for _, candidate in group.groupby('model'):
            match_errors(candidate,baseline)
        validate_actuals(baseline,raw[raw.category.eq('Все категории')])
    frame = frame.join(metadata,on='territory_id',validate='many_to_one')
    assert frame[['expense_2023','expense_quartile','region_code','region_name']].notna().all().all()
    return frame, metadata, cuts, panel


def tables(frame):
    overview, groups, paired, sensitivity, municipal = [], [], [], [], []
    for (horizon, model), part in frame.groupby(['horizon','model'],sort=True):
        labels = {'horizon':int(horizon),'model':model}
        overview.append({**labels,**metrics(part)})
        for dimension in ['expense_quartile','region_name']:
            for label, group in part.groupby(dimension,sort=True):
                groups.append({**labels,'dimension':dimension,'group':label,**metrics(group)})
        for city, group in part.groupby('territory_id',sort=True):
            row = group.iloc[0]
            municipal.append({**labels,'territory_id':city,
                'municipality_name':row.municipal_district_name_short, 'region_name':row.region_name,
                'expense_quartile':row.expense_quartile,'expense_2023':row.expense_2023,
                'MAE_rub':float((group.predicted-group.actual).abs().mean()),
                'bias_rub':float((group.predicted-group.actual).mean()),'dates':len(group)})
        references = {'seasonal_pooled','prophet'}
        if model.endswith('_seasonal'):
            references.add(model.removesuffix('_seasonal'))
        if model == 'hgb_frozen_calendar':
            references.add('hgb_frozen')
        for reference in sorted(references-{model}):
            base = frame[frame.horizon.eq(horizon) & frame.model.eq(reference)]
            pairs = match_errors(part,base)
            labels_pair = {**labels,'reference':reference}
            paired.append({**labels_pair,**paired_metrics(pairs)})
            for dimension in ['target','region_name']:
                for excluded in sorted(pairs[dimension].unique()):
                    keep = pairs[pairs[dimension].ne(excluded)]
                    if len(keep):
                        sensitivity.append({**labels_pair,'exclusion':dimension,'excluded':excluded,
                            **paired_metrics(keep)})
            expenses = pairs[['territory_id','expense_2023']].drop_duplicates().sort_values(
                ['expense_2023','territory_id'],ascending=[False,True])
            remove = expenses.head(int(np.ceil(.05*len(expenses)))).territory_id
            keep = pairs[~pairs.territory_id.isin(remove)]
            sensitivity.append({**labels_pair,'exclusion':'top5pct_expense_2023',
                'excluded':','.join(map(str,remove)),**paired_metrics(keep)})
    return {name:pd.DataFrame(rows) for name, rows in [('overview',overview),('groups',groups),
             ('paired',paired),('sensitivity',sensitivity),('municipal',municipal)]}


def audit_metadata(root, metadata, cuts):
    return {'input_sha256':{path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in INPUTS},
            'training_ids':metadata.index.astype(int).tolist(),'quartile_boundaries_2023':cuts.tolist(),
            'strata_period':'2023 only; all 2075 complete positive histories',
            'geography_role':'2023 lookup for descriptive labels only; no forecast features',
            'tie_tolerance_rub':TOL,'selection':'no model fit, tuning or promotion',
            'limits':'Reused 2024, one target at 6/12m; regional counts are small. No population claim, confidence interval or new independent test.'}


def verify():
    frame, metadata, cuts, panel = load_inputs()
    altered = panel.copy();altered.loc[:,altered.columns >= '2024-01'] = np.nan
    after, after_cuts = training_metadata(altered)
    pd.testing.assert_frame_equal(metadata[['expense_2023','expense_quartile']],after,check_index_type=False)
    np.testing.assert_array_equal(cuts,after_cuts)
    assert len(metadata) == 2075
    for name, expected in tables(frame).items():
        saved = pd.read_csv(ROOT/'results'/f'{PREFIX}{name}.csv')
        pd.testing.assert_frame_equal(saved,expected,check_dtype=False,check_exact=False,rtol=1e-11,atol=1e-9)
    saved = json.loads((ROOT/'results'/f'{PREFIX}audit.json').read_text())
    assert saved == audit_metadata(ROOT,metadata,cuts)
    overview = pd.read_csv(ROOT/'results'/f'{PREFIX}overview.csv')
    foundation = pd.read_csv(ROOT/'results/foundation_seasonal_summary.csv')
    for row in overview.itertuples():
        if row.model in ['hgb_frozen','hgb_frozen_calendar']:
            continue
        base = foundation[foundation.horizon.eq(row.horizon) & foundation.model.eq(row.model)].iloc[0]
        np.testing.assert_allclose(row.MAE_date_balanced,base.MAE_date_balanced,rtol=1e-12)
    return True


def main():
    frame, metadata, cuts, _ = load_inputs()
    for name, table in tables(frame).items():
        table.to_csv(ROOT/'results'/f'{PREFIX}{name}.csv',index=False)
    (ROOT/'results'/f'{PREFIX}audit.json').write_text(json.dumps(audit_metadata(ROOT,metadata,cuts),ensure_ascii=False,indent=2))
    print('Frozen distribution audit generated; no model selection.')


if __name__ == '__main__':
    main()
