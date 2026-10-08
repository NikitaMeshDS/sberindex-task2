"""Sequential model selection from past one-step errors, not future horizon loss.

Two predeclared policies: expanding and rolling-three-month calibration.
Candidates are the same 25 regional-seasonal/Prophet mixtures. No policy is
selected on late results. Data publication lag is assumed zero.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sberindex.detection.change_detection import panel_for
from sberindex.forecasting.regional_seasonality import CANDIDATES,ratios
from sberindex.paths import ROOT
def choose(scores,origin,policy):
    known=scores[scores.target.le(origin)]
    if policy=='rolling_3m':known=known[known.target.ge(str(pd.Period(origin,freq='M')-2))]
    assert len(known) and known.target.max()<=origin
    ranked=known.groupby(['candidate','weight'],sort=True).ae.mean().reset_index()
    best=ranked.loc[ranked.ae.idxmin()]
    return str(best.candidate),float(best.weight),float(best.ae),known.target.min(),known.target.max()


def main():
    ids,values=panel_for('Все категории')
    raw=pd.read_parquet(ROOT/'results/rolling_predictions.parquet')
    prophet=raw[(raw.model=='prophet')&(raw.horizon==1)&raw.origin.between('2024-01','2024-10')]
    sample_ids=np.sort(prophet.territory_id.unique());mask=np.isin(ids,sample_ids)
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv')
    regions=lookup[lookup.year==2024].set_index('territory_id').loc[ids,'region_code'].to_numpy()
    weights=json.loads((ROOT/'config.json').read_text())['blend_candidate_seasonal_weights']
    scores=[]
    for origin in range(12,22):
        month=str(pd.Period('2023-01',freq='M')+origin)
        p=prophet[prophet.origin==month].set_index('territory_id').loc[sample_ids]
        np.testing.assert_allclose(p.actual,values[mask,origin+1])
        for candidate in CANDIDATES:
            regional=(values[:,origin]*ratios(values,regions,origin,origin+1,candidate))[mask]
            for weight in weights:
                errors=np.abs(p.actual.to_numpy()-(weight*regional+(1-weight)*p.predicted.to_numpy()))
                # Equal sample in each month: mean monthly AE equals pooled AE.
                scores.append({'origin':month,'target':str(pd.Period(month,freq='M')+1),
                               'candidate':candidate,'weight':weight,'ae':errors.mean()})
    scores=pd.DataFrame(scores)
    scores.to_csv(ROOT/'results/adaptive_candidate_monthly_errors.csv',index=False)
    primary=pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    regional=pd.read_parquet(ROOT/'results/regional_seasonality_predictions.parquet')
    regional=regional[(regional.category=='Все категории')&regional.territory_id.isin(sample_ids)]
    rows=[];selections=[];future_checks=0
    for origin in sorted(primary[primary.horizon==1].origin.unique()):
        for policy in ['expanding','rolling_3m']:
            candidate,weight,loss,start,end=choose(scores,origin,policy)
            changed=scores.copy();changed.loc[changed.target>origin,'ae']+=1e6
            assert choose(changed,origin,policy)==(candidate,weight,loss,start,end)
            future_checks+=1
            selections.append({'origin':origin,'policy':policy,'candidate':candidate,'weight':weight,
                               'past_MAE':loss,'calibration_first':start,'calibration_last':end})
            p=primary[(primary.model=='prophet')&(primary.origin==origin)&primary.horizon.isin([1,3,6])]
            r=regional[(regional.candidate==candidate)&(regional.origin==origin)]
            keys=['territory_id','origin','target','horizon']
            m=p.merge(r,on=keys,validate='one_to_one',suffixes=('_prophet','_regional'))
            assert len(m)==len(p)
            np.testing.assert_allclose(m.actual_prophet,m.actual_regional)
            out=m[keys].copy();out['actual']=m.actual_prophet
            out['predicted']=weight*m.predicted_regional+(1-weight)*m.predicted_prophet
            out['model']='adaptive_'+policy;rows.append(out)
    result=pd.concat(rows,ignore_index=True)
    result.to_parquet(ROOT/'results/adaptive_forecast_predictions.parquet',index=False)
    pd.DataFrame(selections).to_csv(ROOT/'results/adaptive_forecast_selections.csv',index=False)
    compare=pd.concat([primary[primary.horizon.isin([1,3,6])],result],ignore_index=True)
    compare['ae']=(compare.actual-compare.predicted).abs()
    compare.groupby(['horizon','model']).agg(MAE=('ae','mean'),dates=('target','nunique'),municipalities=('territory_id','nunique')).to_csv(ROOT/'results/adaptive_forecast_comparison.csv')
    compare.groupby(['horizon','target','model']).ae.mean().to_csv(ROOT/'results/adaptive_forecast_monthly.csv')
    sensitivity=[]
    for h in [1,3,6]:
        part=compare[(compare.horizon==h)&compare.model.isin(['blend_75','adaptive_expanding','adaptive_rolling_3m'])]
        monthly=part.groupby(['target','model']).ae.mean().unstack()
        for policy in ['adaptive_expanding','adaptive_rolling_3m']:
            gains=monthly.blend_75-monthly[policy]
            for month in gains.index:
                remaining=gains.drop(month)
                sensitivity.append({'horizon':h,'model':policy,'excluded_target':month,
                    'remaining_dates':len(remaining),'mean_gain_rub':remaining.mean() if len(remaining) else np.nan})
    pd.DataFrame(sensitivity).to_csv(ROOT/'results/adaptive_forecast_sensitivity.csv',index=False)
    (ROOT/'results/adaptive_forecast_protocol.json').write_text(json.dumps({'policies':['expanding','rolling_3m'],
        'candidates':25,'selection_metric':'past one-step MAE on fixed 256 municipalities',
        'horizons':[1,3,6],'future_perturbation_checks':future_checks,'publication_lag_assumption':0,
        'limits':'Retrospective reused archive; neither policy selected by evaluation score. One-step tuning need not optimize longer horizons. Current-month observations assumed available.'},ensure_ascii=False,indent=2))
    print(pd.DataFrame(selections).to_string(index=False))
    print(compare[compare.model.isin(['adaptive_expanding','adaptive_rolling_3m','blend_75','seasonal_pooled','prophet'])].groupby(['horizon','model']).ae.mean().to_string())


if __name__=='__main__':main()
