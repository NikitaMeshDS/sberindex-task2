"""Transfer the early-selected regional mixture to hypothetical reporting lags.

No new parameter selection or Prophet training. Four already studied target
months are compared on the identical saved 251-ID reporting-delay intersection.
"""
import json
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
from sberindex.forecasting.asof_regional_forecast import seasonal_predictions, metrics

OUT = ROOT/'results'
KEYS = ['territory_id','origin','target','horizon','reporting_delay_months','decision_month']


def main():
    source = pd.read_parquet(ROOT/'data/consumption.parquet')
    panel = source[source.category == 'Все категории'].pivot(
        index='territory_id',columns='date',values='value').sort_index()
    panel = panel.loc[panel.iloc[:,:12].notna().all(axis=1)]
    geo = pd.read_csv(OUT/'municipal_lookup.csv')
    regions = geo[geo.year == 2023].set_index('territory_id').loc[panel.index,'region_code'].to_numpy()
    selected = json.loads((OUT/'asof_regional_protocol.json').read_text())
    candidate = selected['selected_candidate']
    weight = selected['selected_seasonal_weight']
    saved = pd.read_parquet(OUT/'asof_reporting_delay_predictions.parquet')
    base = saved[saved.model == 'prophet'].copy().reset_index(drop=True)
    regional = base[KEYS+['actual']].copy()
    regional['predicted'] = weight*seasonal_predictions(panel,regions,base,candidate)+(1-weight)*base.predicted
    regional['model'] = 'regional_asof_transferred'
    compare = pd.concat([saved[KEYS+['actual','predicted','model']],regional],ignore_index=True)
    assert not compare.duplicated(KEYS+['model']).any()
    assert compare.groupby(['territory_id','target']).actual.nunique().eq(1).all()
    compare.to_parquet(OUT/'asof_regional_delay_predictions.parquet',index=False)
    summary = [dict(model=model,reporting_delay_months=int(delay),**metrics(group))
               for (model,delay),group in compare.groupby(['model','reporting_delay_months'])]
    pd.DataFrame(summary).to_csv(OUT/'asof_regional_delay_summary.csv',index=False)
    compare['ae'] = abs(compare.actual-compare.predicted)
    monthly = compare.groupby(['model','reporting_delay_months','target']).agg(
        MAE=('ae','mean'),municipalities=('territory_id','nunique')).reset_index()
    monthly.to_csv(OUT/'asof_regional_delay_monthly.csv',index=False)
    sensitivity = []
    for delay,group in monthly.groupby('reporting_delay_months'):
        errors = group.pivot(index='target',columns='model',values='MAE')
        for benchmark in ['blend_75','seasonal_pooled','prophet']:
            delta = errors.regional_asof_transferred-errors[benchmark]
            leave_one_out = [delta.drop(month).mean() for month in delta.index]
            sensitivity.append(dict(reporting_delay_months=int(delay),benchmark=benchmark,
                dates=len(delta),MAE_difference_rub=delta.mean(),months_regional_better=int((delta<0).sum()),
                leave_one_date_out_min_rub=min(leave_one_out),leave_one_date_out_max_rub=max(leave_one_out)))
    pd.DataFrame(sensitivity).to_csv(OUT/'asof_regional_delay_sensitivity.csv',index=False)
    protocol = dict(candidate=candidate,seasonal_weight=weight,eligible_ids=len(panel),
        selection='Transferred unchanged from asof_regional_protocol.json; selected on Feb-Jun one-step forecasts with zero lag',
        operational_rule='At end of t predict t+1; last observed month t-delay; effective horizon 1+delay',
        target_months=sorted(compare.target.unique()),delays_months=[0,1,2],
        geography_year=2023,profile_year=2023,
        pairs='Saved common 251-ID reporting-delay intersection, four observed target months, all models and delays',
        sensitivity='Leave-one-target-date-out difference, not confidence interval; no strategy selection',
        limits='Hypothetical release lags; real publication vintages unknown. Reused 2024 archive, not independent. Early zero-lag selection may transfer poorly to lagged forecasts. No new model training, no 6/12-month lagged evaluation, primary model unchanged.')
    (OUT/'asof_regional_delay_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
