"""Past-only multiplicative MAE correction, calibrated separately by horizon.

Fixed candidate grid, selected on February-June h1 targets. Reused archive.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
CANDIDATES=['none','global_expanding','global_3m','municipal_expanding','municipal_3m']
KEYS=['territory_id','origin','target','horizon']


def weighted_median(values,weights):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    assert len(values) and np.all(weights>0) and np.isfinite(values).all()
    order=np.argsort(values,kind='stable')
    return float(values[order][np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)])


def factors(history,origin,horizon,ids,candidate):
    ids=np.asarray(ids)
    if candidate=='none':return np.ones(len(ids)),dict(months=0,rows=0,last_target='',global_factor=1.)
    known=history[(history.horizon==horizon)&(history.target<=origin)].copy()
    if candidate.endswith('_3m'):
        known=known[known.target>=str(pd.Period(origin,freq='M')-2)]
    meta=dict(months=int(known.target.nunique()),rows=len(known),last_target=known.target.max() if len(known) else '',global_factor=1.)
    # No transfers from h1 to h3/h6: the bias may depend on the forecast horizon.
    if meta['months']<2:return np.ones(len(ids)),meta
    global_factor=np.clip(weighted_median(known.actual/known.predicted,known.predicted),.85,1.15)
    meta['global_factor']=float(global_factor)
    result=np.full(len(ids),global_factor)
    if candidate.startswith('municipal'):
        grouped={city:g for city,g in known.groupby('territory_id')}
        for i,city in enumerate(ids):
            group=grouped.get(city)
            if group is None or group.target.nunique()<2:continue
            local=weighted_median(group.actual/group.predicted,group.predicted)
            n=group.target.nunique();weight=n/(n+3)
            result[i]=np.clip(weight*local+(1-weight)*global_factor,.85,1.15)
    return result,meta


def base_predictions():
    raw=pd.read_parquet(ROOT/'results/rolling_predictions.parquet')
    raw['target']=(pd.PeriodIndex(raw.origin,freq='M')+raw.horizon.to_numpy()).astype(str)
    ids=raw.loc[raw.model=='prophet','territory_id'].unique()
    part=raw[raw.model.isin(['prophet','seasonal_pooled'])&raw.territory_id.isin(ids)&raw.horizon.isin([1,3,6])&raw.origin.ge('2024-01')]
    p=part[part.model=='prophet'].merge(part[part.model=='seasonal_pooled'],on=KEYS,validate='one_to_one',suffixes=('_p','_s'))
    np.testing.assert_allclose(p.actual_p,p.actual_s)
    result=p[KEYS].copy();result['actual']=p.actual_p;result['predicted']=.25*p.predicted_p+.75*p.predicted_s
    assert result.predicted.gt(0).all()
    return result.sort_values(KEYS).reset_index(drop=True)


def main():
    base=base_predictions()
    rows=[];traces=[]
    for (origin,horizon),group in base.groupby(['origin','horizon']):
        for candidate in CANDIDATES:
            scale,meta=factors(base,origin,horizon,group.territory_id.to_numpy(),candidate)
            out=group.copy();out['factor']=scale;out['predicted']=out.predicted*scale;out['candidate']=candidate
            rows.append(out);traces.append(dict(origin=origin,horizon=int(horizon),candidate=candidate,**meta))
    predictions=pd.concat(rows,ignore_index=True)
    predictions['ae']=(predictions.actual-predictions.predicted).abs()
    early=predictions[(predictions.horizon==1)&predictions.target.le('2024-06')]
    validation=early.groupby('candidate',sort=False).agg(MAE=('ae','mean'),dates=('target','nunique')).reset_index()
    selected=validation.loc[validation.MAE.idxmin(),'candidate']
    late=predictions[predictions.origin.ge('2024-06')].copy()
    late.to_parquet(ROOT/'results/bias_correction_candidates.parquet',index=False)
    validation.to_csv(ROOT/'results/bias_correction_validation.csv',index=False)
    pd.DataFrame(traces).to_csv(ROOT/'results/bias_correction_calibration.csv',index=False)
    late.groupby(['horizon','candidate']).agg(MAE=('ae','mean'),dates=('target','nunique'),municipalities=('territory_id','nunique')).to_csv(ROOT/'results/bias_correction_comparison.csv')
    chosen=late[late.candidate==selected].copy();chosen['model']='bias_corrected_selected'
    chosen[KEYS+['actual','predicted','model']].to_parquet(ROOT/'results/bias_correction_predictions.parquet',index=False)
    monthly=late.groupby(['horizon','target','candidate'],as_index=False).ae.mean().rename(columns={'ae':'MAE'})
    monthly.to_csv(ROOT/'results/bias_correction_monthly.csv',index=False)
    gains=[]
    for h in [1,3,6]:
        m=monthly[monthly.horizon==h].pivot(index='target',columns='candidate',values='MAE')
        gain=m['none']-m[selected]
        for month in gain.index:
            remaining=gain.drop(month)
            gains.append(dict(horizon=h,excluded_target=month,remaining_dates=len(remaining),mean_gain_rub=remaining.mean()))
    pd.DataFrame(gains).to_csv(ROOT/'results/bias_correction_sensitivity.csv',index=False)
    protocol=dict(candidates=CANDIDATES,selected=selected,validation='February-June2024 h1; fixed original75/25 base; same early period previously used to select that base, so calibration is not independent.',
        evaluation='Origins June2024 onward; previously analysed archive. No primary replacement based on late scores.',
        factor='weighted median(actual/predicted), weights=predicted: minimizes past ruble absolute loss for a common positive multiplier. Clip[0.85,1.15].',
        local_shrinkage='n_dates/(n_dates+3) toward global; minimum2 completed target dates. No cross-horizon transfer.',
        availability='target<=origin month; zero reporting delay assumed. No future actuals may enter correction.',
        limitation='Few distinct dates; clips/shrinkage/window are fixed design choices, not independently validated. h6 may lack completed historical errors.')
    (ROOT/'results/bias_correction_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(validation.to_string(index=False));print('SELECTED',selected)
    print(late[late.candidate.isin(['none',selected])].groupby(['horizon','candidate']).ae.mean().to_string())


if __name__=='__main__':main()
