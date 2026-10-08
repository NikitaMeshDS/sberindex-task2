"""Seasonally normalized multi-month anchors; early one-step selection only.

Four fixed anchor estimators × two fixed seasonal profiles, all six categories.
Uses 2023 seasonal profiles and observed history up to origin. No late tuning.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sberindex.detection.change_detection import CATEGORIES,panel_for
from sberindex.paths import ROOT
ANCHORS=['last','mean_3m','median_3m','weighted_3m']
PROFILES=['pooled','regional_shrink100']


def profiles(values,regions,kind):
    total=values[:,:12].sum(axis=0);global_profile=total/total.mean()
    result=np.tile(global_profile,(len(values),1))
    if kind=='regional_shrink100':
        for r in np.unique(regions):
            mask=regions==r;n=mask.sum()
            if n<10:continue
            series=values[mask,:12].sum(axis=0);profile=series/series.mean()
            w=n/(n+100)
            result[mask]=np.exp(w*np.log(profile)+(1-w)*np.log(global_profile))
    return result


def predict(values,profile,origin,horizon,anchor):
    months=np.arange(origin-2,origin+1)
    level=values[:,months]/profile[:,months%12]
    if anchor=='last':current=level[:,-1]
    elif anchor=='mean_3m':current=level.mean(axis=1)
    elif anchor=='median_3m':current=np.median(level,axis=1)
    elif anchor=='weighted_3m':current=level@np.array([.2,.3,.5])
    else:raise ValueError(anchor)
    return current*profile[:,(origin+horizon)%12]


def main():
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv');lookup=lookup[lookup.year==2024].set_index('territory_id')
    sample_ids=pd.read_parquet(ROOT/'results/primary_predictions.parquet').territory_id.unique()
    validation=[];records=[];selection={};temporal_error=0.
    for category in CATEGORIES:
        ids,values=panel_for(category);regions=lookup.loc[ids,'region_code'].to_numpy()
        cohort=np.isin(ids,sample_ids)
        for kind in PROFILES:
            profile=profiles(values,regions,kind)
            for anchor in ANCHORS:
                error=np.stack([np.abs(predict(values,profile,o,1,anchor)-values[:,o+1]) for o in range(12,17)])
                for name,mask in [('all',np.ones(len(ids),bool)),('primary256',cohort)]:
                    validation.append({'category':category,'cohort':name,'profile':kind,'anchor':anchor,'MAE':error[:,mask].mean(),'municipalities':int(mask.sum()),'dates':5})
                changed=values.copy();changed[:,18:]*=10
                old=predict(values,profile,17,3,anchor)
                new=predict(changed,profiles(changed,regions,kind),17,3,anchor)
                temporal_error=max(temporal_error,float(np.abs(old-new).max()))
                for h in [1,3,6]:
                    for origin in range(17,24-h):
                        prediction=predict(values,profile,origin,h,anchor)
                        origin_label=str(pd.Period("2023-01",freq="M")+origin)
                        target_label=str(pd.Period("2023-01",freq="M")+origin+h)
                        # Only validation-selected candidates will be persisted below.
                        for i,city in enumerate(ids):
                            records.append((category,int(city),origin_label,target_label,h,kind,anchor,values[i,origin+h],prediction[i]))
        print(category,flush=True)
    assert temporal_error==0
    val=pd.DataFrame(validation);val.to_csv(ROOT/'results/robust_anchor_validation.csv',index=False)
    frame=pd.DataFrame(records,columns=['category','territory_id','origin','target','horizon','profile','anchor','actual','predicted'])
    frame['ae']=(frame.actual-frame.predicted).abs()
    frame.groupby(['category','horizon','profile','anchor']).agg(MAE=('ae','mean'),dates=('target','nunique'),municipalities=('territory_id','nunique')).to_csv(ROOT/'results/robust_anchor_all_candidates.csv')
    selected=[]
    for category in CATEGORIES:
        for cohort in ['all','primary256']:
            v=val[(val.category==category)&(val.cohort==cohort)];best=v.loc[v.MAE.idxmin()]
            g=frame[(frame.category==category)&(frame.profile==best.profile)&(frame.anchor==best.anchor)].copy()
            if cohort=='primary256':g=g[g.territory_id.isin(sample_ids)]
            g['cohort']=cohort;selected.append(g)
            selection[category+'|'+cohort]={'profile':best.profile,'anchor':best.anchor,'validation_MAE':float(best.MAE)}
    selected=pd.concat(selected,ignore_index=True)
    selected.to_parquet(ROOT/'results/robust_anchor_selected.parquet',index=False)
    selected.groupby(['category','cohort','horizon']).agg(MAE=('ae','mean'),dates=('target','nunique'),municipalities=('territory_id','nunique')).to_csv(ROOT/'results/robust_anchor_summary.csv')
    keys=['category','territory_id','origin','target','horizon','profile']
    baseline=frame[frame.anchor=='last'][keys+['ae']].rename(columns={'ae':'baseline_ae'})
    paired=selected.merge(baseline,on=keys,validate='many_to_one')
    paired['gain']=paired.baseline_ae-paired.ae
    paired.groupby(['category','cohort','horizon','target']).agg(
        gain_rub=('gain','mean'),MAE=('ae','mean'),baseline_MAE=('baseline_ae','mean')).to_csv(ROOT/'results/robust_anchor_monthly_gains.csv')
    cities=paired.groupby(['category','cohort','horizon','territory_id']).gain.mean().reset_index()
    cities['win']=cities.gain>1e-8
    cities.groupby(['category','cohort','horizon']).agg(
        municipal_win_rate=('win','mean'),mean_gain_rub=('gain','mean'),municipalities=('territory_id','size')).to_csv(ROOT/'results/robust_anchor_municipal_gains.csv')
    (ROOT/'results/robust_anchor_protocol.json').write_text(json.dumps({'anchors':ANCHORS,'profiles':PROFILES,
        'weighted_anchor':[.2,.3,.5],'selection':selection,'validation':'February-June 2024, one-step errors',
        'evaluation':'origins June onward, horizons 1/3/6','future_perturbation_max_error':temporal_error,
        'limits':'Exploratory reused archive. Two selection cohorts reported separately. No post-evaluation parameter changes.'},ensure_ascii=False,indent=2))
    print(val[val.category=='Все категории'].to_string(index=False))
    print(selected[selected.category=='Все категории'].groupby(['cohort','horizon']).ae.mean().to_string())


if __name__=='__main__':main()
