"""Regional-seasonal/Prophet mixture, 25 candidates, early validation only.

Uses the existing five mixture weights and five regional candidates; all
results are retrospective because the later archive has already been explored.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sberindex.forecasting.regional_seasonality import ratios,CANDIDATES
from sberindex.detection.change_detection import panel_for
from sberindex.paths import ROOT
def main():
    ids,values=panel_for('Все категории')
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv')
    lookup=lookup[lookup.year==2024].set_index('territory_id');regions=lookup.loc[ids,'region_code'].to_numpy()
    raw=pd.read_parquet(ROOT/'results/rolling_predictions.parquet')
    prophet=raw[(raw.model=='prophet')&(raw.horizon==1)&raw.origin.between('2024-01','2024-05')]
    sample_ids=np.sort(prophet.territory_id.unique());mask=np.isin(ids,sample_ids)
    weights=json.loads((ROOT/'config.json').read_text())['blend_candidate_seasonal_weights']
    rows=[];regional_validation=[]
    for candidate in CANDIDATES:
        seasonal=[];base=[];actual=[]
        for origin in range(12,17):
            month=str(pd.Period('2023-01',freq='M')+origin)
            p=prophet[prophet.origin==month].set_index('territory_id').loc[sample_ids]
            np.testing.assert_allclose(p.actual,values[mask,origin+1])
            seasonal.append((values[:,origin]*ratios(values,regions,origin,origin+1,candidate))[mask])
            base.append(p.predicted.to_numpy());actual.append(p.actual.to_numpy())
        seasonal=np.concatenate(seasonal);base=np.concatenate(base);actual=np.concatenate(actual)
        regional_validation.append({'candidate':candidate,'MAE':np.abs(actual-seasonal).mean(),'municipalities':256,'dates':5})
        for weight in weights:
            rows.append({'candidate':candidate,'seasonal_weight':weight,'MAE':np.abs(actual-(weight*seasonal+(1-weight)*base)).mean()})
    val=pd.DataFrame(rows);val.to_csv(ROOT/'results/regional_blend_validation.csv',index=False)
    pd.DataFrame(regional_validation).to_csv(ROOT/'results/regional_seasonality_validation_256.csv',index=False)
    chosen=val.loc[val.MAE.idxmin()];candidate=chosen.candidate;weight=float(chosen.seasonal_weight)
    seasonal=pd.read_parquet(ROOT/'results/regional_seasonality_predictions.parquet')
    seasonal=seasonal[(seasonal.category=='Все категории')&(seasonal.candidate==candidate)&seasonal.territory_id.isin(sample_ids)]
    primary=pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    keys=['territory_id','origin','target','horizon']
    merged=seasonal.merge(primary[primary.model=='prophet'][keys+['actual','predicted']],on=keys,validate='one_to_one',suffixes=('_regional','_prophet'))
    np.testing.assert_allclose(merged.actual_regional,merged.actual_prophet)
    result=merged[keys].copy();result['actual']=merged.actual_regional
    result['predicted']=weight*merged.predicted_regional+(1-weight)*merged.predicted_prophet
    result['model']='regional_blend_selected'
    result.to_parquet(ROOT/'results/regional_blend_predictions.parquet',index=False)
    compare=pd.concat([primary[primary.horizon.isin([1,3,6])],result],ignore_index=True)
    compare['ae']=(compare.actual-compare.predicted).abs()
    compare.groupby(['horizon','model']).agg(MAE=('ae','mean'),dates=('target','nunique'),municipalities=('territory_id','nunique')).to_csv(ROOT/'results/regional_blend_comparison.csv')
    (ROOT/'results/regional_blend_protocol.json').write_text(json.dumps({'candidate':candidate,'seasonal_weight':weight,
        'validation_MAE':float(chosen.MAE),'candidate_count':len(val),'validation':'same 256 municipalities, February-June one-month targets',
        'evaluation':'origins June or later, 1/3/6-month horizons, unchanged selected parameters',
        'limits':'Exploratory expansion of candidate set after archive inspection; no untouched holdout, no claims of guaranteed superiority.'},ensure_ascii=False,indent=2))
    print(val.sort_values('MAE').head(8).to_string(index=False))
    print(compare[compare.model.isin(['regional_blend_selected','blend_75','prophet'])].groupby(['horizon','model']).ae.mean().to_string())


if __name__=='__main__':main()
