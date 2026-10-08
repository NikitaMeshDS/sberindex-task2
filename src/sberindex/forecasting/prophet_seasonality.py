"""Central adapter and fixed stronger Prophet benchmark; separate appendix."""
import concurrent.futures
import hashlib
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sberindex.paths import ROOT


def build_model(variant, config):
    from prophet import Prophet
    model=Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False,uncertainty_samples=0)
    if variant=='prophet_yearly3':
        model.add_seasonality('yearly',period=config['yearly_period_days'],fourier_order=config['yearly_fourier_order'],
            prior_scale=config['seasonality_prior_scale'],mode=config['seasonality_mode'])
    elif variant=='prophet_pooled_profile':
        model.add_regressor('pooled_profile',prior_scale=config['regressor_prior_scale'],standardize=False,mode='multiplicative')
    elif variant!='prophet_disabled':raise ValueError('unknown Prophet variant')
    return model


def design_frames(history,profile,horizon):
    history=np.asarray(history,float);profile=np.asarray(profile,float)
    if horizon<1 or len(history)<2 or profile.shape!=(12,) or not np.isfinite(profile).all() or (profile<=0).any() or not np.isfinite(history).all() or (history<=0).any():
        raise ValueError('finite positive observed history and12-month profile required')
    dates=pd.date_range('2023-01-01',periods=len(history)+horizon,freq='MS')
    reg=profile[np.arange(len(dates))%12]-1
    train=pd.DataFrame({'ds':dates[:len(history)],'y':history,'pooled_profile':reg[:len(history)]})
    future=pd.DataFrame({'ds':dates[len(history):],'pooled_profile':reg[len(history):]})
    return train,future


def fit_predict(history,profile,horizon,variant,config):
    train,future=design_frames(history,profile,horizon)
    if variant!='prophet_pooled_profile':
        train=train.drop(columns='pooled_profile');future=future.drop(columns='pooled_profile')
    model=build_model(variant,config)
    model.fit(train,seed=config['seed'])
    return np.maximum(0.,model.predict(future).yhat.to_numpy(float))


def worker(job):
    city,origin,history,profile,horizon,config=job
    logging.getLogger('cmdstanpy').disabled=True
    result={}
    for variant in config['models']:result[variant]=fit_predict(history,profile,horizon,variant,config)
    return (city,origin),result


def main():
    started=time.monotonic()
    cfg=json.loads((ROOT/'configs/prophet_seasonality.json').read_text())
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    panel=raw[raw.category=='Все категории'].pivot(index='territory_id',columns='date',values='value').sort_index()
    panel=panel.reindex(columns=pd.period_range('2023-01','2024-12',freq='M').astype(str))
    train=panel.iloc[:,:12]
    eligible=train.index[(np.isfinite(train).all(axis=1)&(train>0).all(axis=1))]
    sums=train.loc[eligible].sum(axis=0).to_numpy(float);profile=sums/sums.mean()
    ids=json.loads((ROOT/'results/asof_cohort_protocol.json').read_text())['sample_ids']
    old=pd.read_parquet(ROOT/'reports/operational_workflow/delayed_predictions.parquet')
    old=old[(old.reporting_lag==0)&old.model.isin(['prophet','seasonal_pooled'])].copy()
    old=old.rename(columns={'business_horizon':'horizon'})
    keys=['territory_id','origin','target','horizon']
    base=old[old.model=='prophet'][keys+['actual','predicted']].sort_values(keys).reset_index(drop=True)
    season=old[old.model=='seasonal_pooled'][keys+['actual','predicted']].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(base[keys+['actual']],season[keys+['actual']])
    assert set(base.territory_id).issubset(ids) and set(base.horizon)==set(cfg['horizons'])
    jobs=[]
    for (city,origin),group in base.groupby(['territory_id','origin'],sort=True):
        index=(pd.Period(origin,'M')-pd.Period('2023-01','M')).n
        history=panel.loc[city].to_numpy(float)[:index+1]
        assert np.isfinite(history).all() and (history>0).all()
        jobs.append((int(city),origin,history,profile,int(group.horizon.max()),cfg))
    predictions={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            key,pred=future.result();predictions[key]=pred
            if n%256==0 or n==len(jobs):print(f'Prophet refits: {n}/{len(jobs)} ID/origins; {time.monotonic()-started:.1f}s',flush=True)
    rows=[]
    for row in base.itertuples(index=False):
        origin=(pd.Period(row.origin,'M')-pd.Period('2023-01','M')).n
        target=origin+row.horizon
        values=panel.loc[row.territory_id].to_numpy(float)
        assert values[target]==row.actual
        assert np.isclose(values[origin]*profile[target%12]/profile[origin%12],season.loc[len(rows)//(len(cfg['models'])+3),'predicted'])
        forecast=predictions[(row.territory_id,row.origin)]
        info=dict(territory_id=row.territory_id,origin=row.origin,target=row.target,horizon=row.horizon,actual=row.actual,
            year_ago=values[target-12],origin_actual=values[origin])
        for model,pred in [('prophet_disabled',row.predicted),('seasonal_pooled',values[origin]*profile[target%12]/profile[origin%12]),('seasonal_naive',values[target-12])]+[(model,float(forecast[model][row.horizon-1])) for model in cfg['models']]:
            rows.append(dict(**info,model=model,predicted=pred))
    result=pd.DataFrame(rows).sort_values(keys+['model']).reset_index(drop=True)
    result['ae']=(result.actual-result.predicted).abs()
    result['yoy_actual']=100*(result.actual/result.year_ago-1)
    result['yoy_predicted']=100*(result.predicted/result.year_ago-1)
    result['yoy_ae']=(result.yoy_actual-result.yoy_predicted).abs()
    result['mom_actual']=np.where(result.horizon==1,100*(result.actual/result.origin_actual-1),np.nan)
    result['mom_predicted']=np.where(result.horizon==1,100*(result.predicted/result.origin_actual-1),np.nan)
    stats=[]
    for (h,model),g in result.groupby(['horizon','model']):
        within=g.actual-g.groupby('territory_id').actual.transform('mean')
        denominator=float((within**2).sum())
        stats.append(dict(horizon=h,model=model,dates=g.target.nunique(),municipalities=g.territory_id.nunique(),observations=len(g),
            MAE=g.groupby('target').ae.mean().mean(),R2_pooled=r2_score(g.actual,g.predicted),
            R2_within_MO=(1-float(((g.actual-g.predicted)**2).sum())/denominator if denominator>0 else np.nan),
            YoY_MAE_pp=g.groupby('target').yoy_ae.mean().mean(),YoY_R2=r2_score(g.yoy_actual,g.yoy_predicted),
            MoM_R2=(r2_score(g.mom_actual,g.mom_predicted) if h==1 else np.nan)))
    summary=pd.DataFrame(stats)
    for reference in ['seasonal_naive','seasonal_pooled']:
        ref=summary[summary.model==reference].set_index('horizon').MAE
        summary['skill_vs_'+reference]=1-summary.MAE/summary.horizon.map(ref)
    monthly=result.groupby(['horizon','model','target']).agg(MAE=('ae','mean'),observations=('ae','size')).reset_index()
    out=ROOT/'reports/prophet_seasonality';out.mkdir(parents=True,exist_ok=True)
    result.to_parquet(out/'predictions.parquet',index=False)
    summary.to_csv(out/'summary.csv',index=False);monthly.to_csv(out/'monthly.csv',index=False)
    inputs=['data/consumption.parquet','results/asof_cohort_protocol.json','reports/operational_workflow/delayed_predictions.parquet',
        'configs/prophet_seasonality.json','docs/protocols/PROPHET_SEASONALITY_EXPERIMENT.md','src/sberindex/forecasting/prophet_seasonality.py']
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'audit.json').write_text(json.dumps(dict(status='passed',seconds=time.monotonic()-started,new_prophet_fits=len(jobs)*len(cfg['models']),
        sample_size_requested=len(ids),eligible_2023=len(eligible),profile=profile.tolist(),
        disabled_prophet_source='existing declared operational_workflow cache; not refitted in this extension',
        seasonal_control_matches=True,new_parameters_tuned=False,independent_time_validation=False,reporting_lag=0,
        input_sha256={name:sha(ROOT/name) for name in inputs},output_sha256={name:sha(out/name) for name in ['predictions.parquet','summary.csv','monthly.csv']},
        limits='Reviewed model choices after exploration2024.256-ID frozen cohort, not full2075; h12 one date. No selection on new results.'),ensure_ascii=False,indent=2)+'\n')
    print(summary[['horizon','model','dates','MAE','YoY_R2','skill_vs_seasonal_pooled']].to_string(index=False),flush=True)

if __name__=='__main__':main()
