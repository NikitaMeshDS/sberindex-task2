"""Fixed 2023 deseasonalized HGB, recursive forecasts and early-only blending.

Seasonality estimated within the completed 2023 training set. Only forecasts
issued in/after December2023 are evaluated. Reused archive, not new holdout.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sberindex.forecasting.online_bias_correction import base_predictions

from sberindex.paths import ROOT
def fit_model(values,config):
    profile=values[:,:12].sum(axis=0)
    profile=profile/profile.mean()
    z=np.log(values[:,:12])-np.log(profile)[None,:]
    def x(a):return np.column_stack([a[:,-1],a[:,-1]-a[:,-2],a[:,-2]-a[:,-3]])
    X=np.concatenate([x(z[:,:t]) for t in range(3,12)])
    y=np.concatenate([z[:,t]-z[:,t-1] for t in range(3,12)])
    model=HistGradientBoostingRegressor(max_iter=config['hgb_max_iter'],max_leaf_nodes=config['hgb_max_leaf_nodes'],
        learning_rate=config['hgb_learning_rate'],min_samples_leaf=config['hgb_min_samples_leaf'],
        l2_regularization=config['hgb_l2_regularization'],random_state=config['random_seed'])
    model.fit(X,y)
    return model,profile


def forecast(model,profile,values,origin,horizon):
    assert origin>=11
    z=np.log(values[:,:origin+1])-np.log(profile[np.arange(origin+1)%12])[None,:]
    for step in range(1,horizon+1):
        X=np.column_stack([z[:,-1],z[:,-1]-z[:,-2],z[:,-2]-z[:,-3]])
        z=np.column_stack([z,z[:,-1]+np.clip(model.predict(X),-.5,.5)])
    return np.exp(z[:,-1])*profile[(origin+horizon)%12]


def main():
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    panel=raw[raw.category=='Все категории'].pivot(index='territory_id',columns='date',values='value').dropna().sort_index()
    values=panel.to_numpy(float)
    config=json.loads((ROOT/'config.json').read_text())
    model,profile=fit_model(values,config)
    baseline=base_predictions()
    sample_ids=np.sort(baseline.territory_id.unique())
    index=panel.index.get_indexer(sample_ids)
    rows=[]
    for (origin,horizon),group in baseline.groupby(['origin','horizon']):
        t=pd.Period(origin,freq='M').ordinal-pd.Period('2023-01',freq='M').ordinal
        predicted=forecast(model,profile,values,t,int(horizon))[index]
        group=group.set_index('territory_id').loc[sample_ids].reset_index()
        np.testing.assert_allclose(group.actual,values[index,t+horizon])
        group['hgb_predicted']=predicted;rows.append(group)
    combined=pd.concat(rows,ignore_index=True)
    early=combined[(combined.horizon==1)&combined.target.le('2024-06')]
    weights=[0.,.25,.5,.75,1.]
    validation=pd.DataFrame([dict(hgb_weight=w,MAE=np.abs(early.actual-((1-w)*early.predicted+w*early.hgb_predicted)).mean()) for w in weights])
    selected=float(validation.loc[validation.MAE.idxmin(),'hgb_weight'])
    late=combined[combined.origin.ge('2024-06')].copy()
    records=[]
    for w in weights:
        out=late.copy();out['predicted']=(1-w)*out.predicted+w*out.hgb_predicted;out['hgb_weight']=w
        out['ae']=(out.actual-out.predicted).abs();records.append(out)
    result=pd.concat(records,ignore_index=True)
    result.to_parquet(ROOT/'results/deseasonal_hgb_candidates.parquet',index=False)
    validation.to_csv(ROOT/'results/deseasonal_hgb_validation.csv',index=False)
    result.groupby(['horizon','hgb_weight']).agg(MAE=('ae','mean'),dates=('target','nunique')).to_csv(ROOT/'results/deseasonal_hgb_comparison.csv')
    result.groupby(['horizon','target','hgb_weight']).ae.mean().rename('MAE').to_csv(ROOT/'results/deseasonal_hgb_monthly.csv')
    chosen=result[result.hgb_weight==selected].copy();chosen['model']='deseasonal_hgb_blend_selected'
    keys=['territory_id','origin','target','horizon','actual','predicted','model']
    chosen[keys].to_parquet(ROOT/'results/deseasonal_hgb_predictions.parquet',index=False)
    h12=forecast(model,profile,values,11,12)[index]
    pd.DataFrame(dict(territory_id=sample_ids,origin='2023-12',target='2024-12',horizon=12,actual=values[index,23],predicted=h12)).to_csv(ROOT/'results/deseasonal_hgb_12m.csv',index=False)
    changed=values.copy();changed[:,18:]*=9
    np.testing.assert_array_equal(forecast(model,profile,values,17,3),forecast(model,profile,changed,17,3))
    protocol=dict(selected_hgb_weight=selected,weights=weights,seasonal_profile_2023=profile.tolist(),model_iterations=int(model.n_iter_),
        training='April-December2023 deseasonalized changes, 2016 complete municipalities, no calendar covariates after removing common profile.',
        profile='Pooled2023 monthly sums divided by annual monthly mean; training-set transformation uses complete2023, therefore earliest forecast origin is December2023.',
        selection='February-June2024 h1, fixed75/25 base previously chosen on these dates. Same reused archive, not independent validation.',
        recursive='Future z values generated recursively; growth clipped[-0.5,0.5]. Model frozen after2023.',
        horizon12='Separate exploratory fixed-model forecast at December2023; h1 blend weight is not applied retroactively at that origin.',
        limitation='Only one seasonal cycle, common seasonality may absorb trend. Internal HGB auto early stopping uses random split within training2023. Zero publication lag assumed.')
    (ROOT/'results/deseasonal_hgb_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(validation.to_string(index=False));print('selected',selected)
    print(result[result.hgb_weight.isin([0,selected,1])].groupby(['horizon','hgb_weight']).ae.mean().to_string())
    print('h12 MAE',float(np.abs(values[index,23]-h12).mean()))


if __name__=='__main__':main()
