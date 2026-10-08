"""Regional shrinkage experiment, selected on Feb-Jun one-month validation.

Candidate set fixed before first calculation; late targets are reused archive,
not independent validation. Geography is the retrospective 2024 dictionary.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sberindex.detection.change_detection import CATEGORIES,panel_for

from sberindex.paths import ROOT
CANDIDATES={'pooled':None,'regional':0,'shrink_5':5,'shrink_20':20,'shrink_100':100}


def ratios(values,regions,origin,target,candidate):
    # All seasonal information comes from 2023, before every allowed origin.
    a,b=origin%12,target%12
    global_ratio=values[:,:12][:,b].sum()/values[:,:12][:,a].sum()
    result=np.full(len(values),global_ratio)
    k=CANDIDATES[candidate]
    if k is not None:
        for region in np.unique(regions):
            mask=regions==region;n=int(mask.sum())
            if n<10:continue
            regional_ratio=values[mask,b].sum()/values[mask,a].sum()
            weight=n/(n+k)
            result[mask]=np.exp(weight*np.log(regional_ratio)+(1-weight)*np.log(global_ratio))
    return result


def main():
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv')
    lookup=lookup[lookup.year==2024].set_index('territory_id')
    validation=[];records=[];selection={}
    for category in CATEGORIES:
        ids,values=panel_for(category);regions=lookup.loc[ids,'region_code'].to_numpy()
        for candidate in CANDIDATES:
            errors=[]
            for origin in range(12,17):
                prediction=values[:,origin]*ratios(values,regions,origin,origin+1,candidate)
                errors.append(np.abs(prediction-values[:,origin+1]))
            validation.append({'category':category,'candidate':candidate,'MAE':np.concatenate(errors).mean(),
                               'dates':5,'municipalities':len(ids)})
        eligible=[r for r in validation if r['category']==category]
        chosen=min(eligible,key=lambda r:r['MAE'])['candidate'];selection[category]=chosen
        for candidate in CANDIDATES:
            for h in [1,3,6]:
                for origin in range(17,24-h):
                    target=origin+h
                    prediction=values[:,origin]*ratios(values,regions,origin,target,candidate)
                    for i,city in enumerate(ids):
                        records.append((category,int(city),int(regions[i]),str(pd.Period('2023-01',freq='M')+origin),
                            str(pd.Period('2023-01',freq='M')+target),h,candidate,candidate==chosen,
                            float(values[i,target]),float(prediction[i])))
        print(category,chosen,flush=True)
    validation=pd.DataFrame(validation);validation.to_csv(ROOT/'results/regional_seasonality_validation.csv',index=False)
    frame=pd.DataFrame(records,columns=['category','territory_id','region_code','origin','target','horizon','candidate','selected','actual','predicted'])
    frame.to_parquet(ROOT/'results/regional_seasonality_predictions.parquet',index=False)
    frame['ae']=(frame.actual-frame.predicted).abs()
    frame.groupby(['category','horizon','candidate']).agg(MAE=('ae','mean'),dates=('target','nunique'),
        municipalities=('territory_id','nunique')).to_csv(ROOT/'results/regional_seasonality_all_candidates.csv')
    # Keep diagnostic candidate scores distinct from the validation-selected model.
    chosen=frame[frame.selected].copy();chosen['model']='regional_selected'
    main=pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    sample=chosen[(chosen.category=='Все категории')&chosen.territory_id.isin(main.territory_id.unique())]
    compare=pd.concat([main[main.horizon.isin([1,3,6])],sample],ignore_index=True)
    compare['ae']=(compare.actual-compare.predicted).abs()
    compare.groupby(['horizon','model']).agg(MAE=('ae','mean'),dates=('target','nunique'),municipalities=('territory_id','nunique')).to_csv(ROOT/'results/regional_seasonality_primary_comparison.csv')
    picked=frame[frame.selected|frame.candidate.eq('pooled')]
    picked.groupby(['category','region_code','horizon','candidate']).agg(MAE=('ae','mean'),municipalities=('territory_id','nunique')).to_csv(ROOT/'results/regional_seasonality_regions.csv')
    protocol={'candidates':CANDIDATES,'regional_minimum_n':10,'ratio':'geometric mixing of regional and pooled 2023 spending ratios; weight n/(n+k)',
        'validation':'February-June 2024, one-month horizon, all complete municipalities per category',
        'evaluation':'origins June 2024 or later; horizons 1/3/6, dates 6/4/1',
        'selection':selection,'limits':'Repeated archive exploration. All-MO validation differs from original 256-MO blend validation. No independent holdout or proof of dictionary point-in-time availability.'}
    (ROOT/'results/regional_seasonality_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(validation.to_string(index=False))
    print(compare[compare.model.isin(['regional_selected','blend_75','seasonal_pooled','prophet'])].groupby(['horizon','model']).ae.mean().to_string())


if __name__=='__main__':main()
