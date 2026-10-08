"""Frozen-model sensitivity to reporting delay and short calibration intervals.

Delay experiment: same Sep-Dec 2024 targets, decisions Aug-Nov, latest observed
month is decision minus 0 or 2 months. Effective horizons are 1 or 3.
Intervals: July-Aug residual calibration, Sep-Dec evaluation, no data delay.
Separate experiments: intervals do not claim coverage under delayed reporting.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
def finite_quantile(values, coverage):
    a=np.sort(np.asarray(values,float))
    rank=int(np.ceil((len(a)+1)*coverage))
    if rank>len(a): return np.inf
    return float(a[rank-1])


def main():
    p=pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    p=pd.concat([p,pd.read_parquet(ROOT/'results/adaptive_forecast_predictions.parquet')],ignore_index=True)
    models=['blend_75','prophet','seasonal_pooled','adaptive_expanding','adaptive_rolling_3m']
    p=p[p.model.isin(models)]
    delay_rows=[]
    for delay in [0,2]:
        group=p[(p.horizon==1+delay)&p.target.between('2024-09','2024-12')].copy()
        group['reporting_delay_months']=delay
        group['decision_month']=(pd.PeriodIndex(group.origin,freq='M')+delay).astype(str)
        assert (group.decision_month>='2024-08').all()
        delay_rows.append(group)
    delayed=pd.concat(delay_rows,ignore_index=True)
    key=['territory_id','target','model']
    zero=delayed[delayed.reporting_delay_months==0].set_index(key)
    two=delayed[delayed.reporting_delay_months==2].set_index(key)
    assert set(zero.index)==set(two.index)
    pd.testing.assert_series_equal(zero.actual.sort_index(),two.actual.sort_index())
    delayed['ae']=(delayed.actual-delayed.predicted).abs()
    delayed.to_parquet(ROOT/'results/reporting_delay_predictions.parquet',index=False)
    summary=delayed.groupby(['model','reporting_delay_months']).agg(MAE=('ae','mean'),dates=('target','nunique'),municipalities=('territory_id','nunique'))
    summary.to_csv(ROOT/'results/reporting_delay_summary.csv')
    delayed.groupby(['target','model','reporting_delay_months']).ae.mean().to_csv(ROOT/'results/reporting_delay_monthly.csv')

    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    train=raw[(raw.category=='Все категории') & (raw.date.astype(str).str[:4]=='2023')]
    scale=train.groupby('territory_id').value.mean()
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv')
    lookup=lookup[lookup.year==2024].set_index('territory_id')
    calibration=p[(p.horizon==1)&p.target.isin(['2024-07','2024-08'])].copy()
    evaluation=p[(p.horizon==1)&p.target.between('2024-09','2024-12')].copy()
    assert calibration.target.max()<=evaluation.origin.min()
    calibration['scale']=calibration.territory_id.map(scale)
    evaluation['scale']=evaluation.territory_id.map(scale)
    interval_rows=[];thresholds=[]
    for model in models:
        cal=calibration[calibration.model==model]
        test=evaluation[evaluation.model==model]
        assert len(cal)==512 and len(test)==1024
        for method in ['absolute','scaled_by_2023_mean']:
            errors=(cal.actual-cal.predicted).abs()
            if method=='scaled_by_2023_mean': errors=errors/cal.scale
            for level in [.8,.95]:
                q=finite_quantile(errors,level)
                thresholds.append({'model':model,'interval_method':method,'nominal_coverage':level,'q':q,'calibration_rows':len(cal),'calibration_dates':2})
                width=q*test.scale if method=='scaled_by_2023_mean' else q
                result=test.copy()
                result['lower']=np.maximum(0,result.predicted-width)
                result['upper']=np.maximum(0,result.predicted+width)
                result['covered']=(result.actual>=result.lower)&(result.actual<=result.upper)
                result['width']=result.upper-result.lower
                alpha=1-level
                result['interval_score']=result.width+2/alpha*(result.lower-result.actual).clip(lower=0)+2/alpha*(result.actual-result.upper).clip(lower=0)
                result['nominal_coverage']=level;result['interval_method']=method
                result['region_name']=result.territory_id.map(lookup.region_name)
                interval_rows.append(result)
    intervals=pd.concat(interval_rows,ignore_index=True)
    intervals.to_parquet(ROOT/'results/forecast_intervals.parquet',index=False)
    pd.DataFrame(thresholds).to_csv(ROOT/'results/interval_calibration.csv',index=False)
    keys=['model','interval_method','nominal_coverage']
    stats=intervals.groupby(keys).agg(empirical_coverage=('covered','mean'),mean_width_rub=('width','mean'),mean_interval_score=('interval_score','mean'),observations=('covered','size'))
    stats.to_csv(ROOT/'results/interval_summary.csv')
    intervals.groupby(keys+['target']).agg(coverage=('covered','mean'),width_rub=('width','mean')).to_csv(ROOT/'results/interval_monthly.csv')
    intervals.groupby(keys+['region_name']).agg(coverage=('covered','mean'),observations=('covered','size')).to_csv(ROOT/'results/interval_regions.csv')
    audit={'delay_scenarios':[0,2],'decision_months':'August-November 2024','targets':'September-December 2024',
        'blend_selection_known':'June targets would be available in August under the assumed two-month lag',
        'interval_calibration':'July-August 2024; 512 municipality-months but only two shared dates',
        'interval_evaluation':'September-December, assuming no publication lag',
        'interval_quantile':'sorted absolute score at ceil((n+1)*coverage), lower clipped at zero',
        'limits':'Scenarios, not measured publication delays. Interval calibration observations are dependent; no distribution-free coverage guarantee, no independent holdout, no choice among interval variants on these scores.'}
    (ROOT/'results/operational_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(summary.to_string());print(stats.to_string())


if __name__=='__main__':main()
