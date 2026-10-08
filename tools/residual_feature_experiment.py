"""Frozen-2023 pooled correction to seasonal forecasts; standalone appendix."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/residual_features'
CATEGORIES=['Все категории','Продовольствие','Здоровье','Маркетплейсы','Общественное питание','Транспорт']
KEYS=['territory_id','origin','target','horizon']
INPUTS=['data/consumption.parquet','data/external/cbr_rate_decisions_2023_2024.csv','results/asof_cohort_protocol.json','results/asof_cohort_scored.parquet','configs/residual_features.json','docs/protocols/RESIDUAL_FEATURE_EXPERIMENT.md','tools/residual_feature_experiment.py','results/asof_cohort_predictions.parquet','data/external/new_sources_audit/rosstat_national_monthly_wage_2023_2024_snapshot2026.csv']


def feature_matrix(values,origin,horizon,profile,events,variant,macro=None,macro_lag=2):
    if origin<2 or horizon<1:raise ValueError('At least three history months and positive horizon required')
    total=values[:,0,:origin+1]
    scaled=total/profile[np.arange(origin+1)%12]
    month=(origin+horizon)%12
    n=len(values)
    own=np.column_stack([np.log(total[:,-1]),np.log(scaled[:,-2]/scaled[:,-1]),np.log(scaled[:,-3]/scaled[:,-1]),np.std(np.log(scaled[:,-3:]),axis=1),np.full(n,horizon),np.full(n,np.sin(2*np.pi*month/12)),np.full(n,np.cos(2*np.pi*month/12))])
    if variant=='own':return own
    with np.errstate(divide='ignore',invalid='ignore'):
        last=values[:,1:,origin];previous=values[:,1:,origin-1]
        categories=np.column_stack([np.log(last/total[:,-1,None]),np.log(last/previous)])
    categories[~np.isfinite(categories)]=np.nan
    result=np.column_stack([own,categories,np.isnan(categories).astype(float)])
    if variant=='categories':return result
    if variant not in ['policy','wage_snapshot']:raise ValueError(variant)
    end=pd.Period('2023-01',freq='M')+origin
    known=events[pd.to_datetime(events.available_from)<=end.end_time].sort_values('available_from')
    rate=7.5 if known.empty else float(known.iloc[-1].rate_after_pct)
    cutoff=(end-2).start_time
    delta=float(known.loc[pd.to_datetime(known.available_from)>=cutoff,'delta_bps'].sum())/100
    result=np.column_stack([result,np.full(n,rate),np.full(n,delta)])
    if variant=='wage_snapshot':
        if macro is None or macro_lag<1:raise ValueError('Snapshot macro requires series and positive hypothetical lag')
        at=origin-macro_lag
        if at<0:raise ValueError('Macro period before available series')
        value=float(macro[at]);growth=0. if at==0 else float(np.log(value/macro[at-1]))
        result=np.column_stack([result,np.full(n,np.log(value)),np.full(n,growth),np.full(n,at==0)])
    return result


def training_arrays(values,profile,events,variant,macro=None,macro_lag=2):
    xs=[];ys=[]
    for origin in range(2,11):
        for h in range(1,12-origin):
            valid=np.isfinite(values[:,0,:origin+1]).all(axis=1)&(values[:,0,:origin+1]>0).all(axis=1)&np.isfinite(values[:,0,origin+h])&(values[:,0,origin+h]>0)
            v=values[valid]
            base=v[:,0,origin]*profile[(origin+h)%12]/profile[origin%12]
            xs.append(feature_matrix(v,origin,h,profile,events,variant,macro,macro_lag));ys.append(np.log(v[:,0,origin+h]/base))
    return np.concatenate(xs),np.concatenate(ys)


def restore(last,origin,horizon,profile,correction,bound):
    return last*profile[(origin+horizon)%12]/profile[origin%12]*np.exp(np.clip(correction,-bound,bound))


def load():
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    months=[str(p) for p in pd.period_range('2023-01',periods=24,freq='M')]
    panels=[raw[raw.category.eq(c)].pivot(index='territory_id',columns='date',values='value').reindex(columns=months) for c in CATEGORIES]
    total=panels[0].sort_index();good=np.isfinite(total.iloc[:,:12]).all(axis=1)&total.iloc[:,:12].gt(0).all(axis=1)
    ids=total.index[good].to_numpy(int);values=np.stack([p.reindex(ids).to_numpy(float) for p in panels],axis=1)
    sums=values[:,0,:12].sum(axis=0);profile=sums/sums.mean()
    sample=json.loads((ROOT/'results/asof_cohort_protocol.json').read_text())['sample_ids']
    assert set(sample).issubset(ids) and len(ids)==2075
    return values,ids,profile,pd.read_csv(ROOT/'data/external/cbr_rate_decisions_2023_2024.csv')


def summarize(frame):
    frame=frame.copy();frame['absolute_error']=(frame.actual-frame.predicted).abs()
    frame['period']=np.where(frame.origin<'2024-06','early','late')
    # h12 has Dec2023 origin; it is a separate single-date extrapolation.
    frame.loc[frame.horizon.eq(12),'period']='h12_single_date'
    rows=[];monthly=[];contrasts=[];sensitivity=[]
    for (h,period,model),g in frame.groupby(['horizon','period','model']):
        rows.append(dict(horizon=h,period=period,model=model,MAE_date_balanced=g.groupby('target').absolute_error.mean().mean(),MAE_pooled=g.absolute_error.mean(),R2_pooled=r2_score(g.actual,g.predicted) if len(g)>1 else np.nan,WAPE_pct=100*g.absolute_error.sum()/g.actual.sum(),dates=g.target.nunique(),municipalities=g.territory_id.nunique(),observations=len(g)))
        for target,part in g.groupby('target'):monthly.append(dict(horizon=h,period=period,model=model,target=target,MAE=part.absolute_error.mean(),observations=len(part)))
        if model.startswith('residual_'):
            references=['seasonal_pooled','prophet','blend_selected']
            family=model.split('_')[1]
            if model.endswith('_categories'):references.append(f'residual_{family}_own')
            if model.endswith('_policy'):references.append(f'residual_{family}_categories')
            if model.endswith('_wage_snapshot'):references.append(f'residual_{family}_policy')
            for ref in references:
                base=frame[(frame.horizon==h)&(frame.period==period)&(frame.model==ref)]
                if base.empty:
                    assert h==12 and ref=='blend_selected', 'Required reference is missing'
                    continue
                paired=g.merge(base[KEYS+['actual','absolute_error']],on=KEYS,suffixes=('','_base'),validate='one_to_one')
                assert len(paired)==len(g);np.testing.assert_array_equal(paired.actual,paired.actual_base)
                paired['difference']=paired.absolute_error-paired.absolute_error_base
                labels=dict(horizon=h,period=period,model=model,reference=ref)
                contrasts.append({**labels,'MAE_difference':paired.groupby('target').difference.mean().mean(),'municipal_win_fraction':paired.groupby('territory_id').difference.mean().lt(0).mean(),'date_win_fraction':paired.groupby('target').difference.mean().lt(0).mean()})
                if g.target.nunique()>1:
                    for omitted in sorted(g.target.unique()):sensitivity.append({**labels,'excluded_target':omitted,'MAE_difference':paired[paired.target.ne(omitted)].groupby('target').difference.mean().mean()})
    combined=[]
    for model,g in frame[frame.horizon.eq(1)].groupby('model'):
        combined.append(dict(model=model,MAE_date_balanced=g.groupby('target').absolute_error.mean().mean(),dates=g.target.nunique(),observations=len(g)))
    return {name:pd.DataFrame(data) for name,data in [('summary',rows),('monthly',monthly),('paired',contrasts),('sensitivity',sensitivity),('h1_all_dates',combined)]}


def reference_frame():
    scored=pd.read_parquet(ROOT/'results/asof_cohort_scored.parquet')
    saved=scored[scored.model.isin(['seasonal_pooled','prophet','blend_selected'])][KEYS+['model','actual','predicted']].copy()
    early=pd.read_parquet(ROOT/'results/asof_cohort_predictions.parquet')
    early=early[early.origin.between('2024-01','2024-05') & early.horizon.eq(1)][KEYS+['model','actual','predicted']].copy()
    a=early[early.model.eq('seasonal_pooled')].sort_values(KEYS).reset_index(drop=True)
    b=early[early.model.eq('prophet')].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[KEYS],b[KEYS]);np.testing.assert_array_equal(a.actual,b.actual)
    blend=a.copy();blend['model']='blend_selected';blend['predicted']=.75*a.predicted+.25*b.predicted
    result=pd.concat([saved,early,blend],ignore_index=True)
    assert not result.duplicated(KEYS+['model']).any()
    return result


def inputs_hash():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}


def main():
    if '--verify' in sys.argv:
        protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['input_sha256']==inputs_hash()
        for name,digest in protocol['artifact_sha256'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
        f=pd.read_parquet(OUT/'predictions.parquet');assert not f.duplicated(KEYS+['model']).any()
        assert np.isfinite(f[['actual','predicted']]).all().all() and f.predicted.gt(0).all()
        ref=reference_frame()
        expected={f'residual_{family}_{variant}' for family in ['ridge','hgb'] for variant in ['own','categories','policy','wage_snapshot']}|set(ref.model)
        assert set(f.model)==expected
        for m in ref.model.unique():
            a=ref[ref.model.eq(m)].sort_values(KEYS).reset_index(drop=True);b=f[f.model.eq(m)].sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b,check_dtype=False)
        for m,g in f[f.model.str.startswith('residual_')].groupby('model'):
            base=ref[ref.model=='seasonal_pooled'].sort_values(KEYS).reset_index(drop=True);g=g.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(base[KEYS],g[KEYS]);np.testing.assert_array_equal(base.actual,g.actual)
        for name,table in summarize(f).items():pd.testing.assert_frame_equal(table,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-10,atol=1e-9)
        print('Residual experiment verified: keys, actuals, metrics and hashes');return
    OUT.mkdir(parents=True,exist_ok=True)
    hashes=inputs_hash();config=json.loads((ROOT/'configs/residual_features.json').read_text())
    values,ids,profile,events=load();index={city:i for i,city in enumerate(ids)}
    original=reference_frame();print('Reference models:',sorted(original.model.unique()),flush=True)
    macro_frame=pd.read_csv(ROOT/'data/external/new_sources_audit/rosstat_national_monthly_wage_2023_2024_snapshot2026.csv').sort_values('date')
    assert macro_frame.date.tolist()==[str(p) for p in pd.period_range('2023-01',periods=24,freq='M')]
    assert set(macro_frame.units)=={'nominal_rub_per_employee_month'}
    macro=macro_frame.value.to_numpy(float)
    assert len(macro)==24 and np.isfinite(macro).all() and (macro>0).all()
    original=original[original.model.isin(['seasonal_pooled','prophet','blend_selected'])][KEYS+['model','actual','predicted']]
    reference=original[original.model.eq('seasonal_pooled')].copy();records=[];fit=[]
    with threadpool_limits(limits=2):
        for variant in ['own','categories','policy','wage_snapshot']:
            x,y=training_arrays(values,profile,events,variant,macro,config['wage_hypothetical_lag_months'])
            # Direct validation of training-only fit against arbitrary future mutation.
            changed=values.copy();changed[:,:,12:]=np.nan
            xx,yy=training_arrays(changed,profile,events,variant,macro,config['wage_hypothetical_lag_months'])
            np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
            estimators={'ridge':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=config['ridge_alpha'])),
                'hgb':HistGradientBoostingRegressor(max_iter=config['hgb_max_iter'],max_leaf_nodes=config['hgb_max_leaf_nodes'],min_samples_leaf=config['hgb_min_samples_leaf'],l2_regularization=config['hgb_l2_regularization'],learning_rate=config['hgb_learning_rate'],random_state=config['seed'])}
            for family,model in estimators.items():
                model.fit(x,y);name=f'residual_{family}_{variant}';fit.append({'model':name,'training_rows':len(y),'features':x.shape[1],'training_max_horizon':9,'training_end':'2023-12'})
                for (origin,h),group in reference.groupby(['origin','horizon']):
                    i=(pd.Period(origin,freq='M')-pd.Period('2023-01',freq='M')).n
                    positions=[index[city] for city in group.territory_id];v=values[positions]
                    features=feature_matrix(v,i,int(h),profile,events,variant,macro,config['wage_hypothetical_lag_months'])
                    predicted=restore(v[:,0,i],i,int(h),profile,model.predict(features),config['max_absolute_log_correction'])
                    base=restore(v[:,0,i],i,int(h),profile,np.zeros(len(v)),config['max_absolute_log_correction'])
                    np.testing.assert_allclose(base,group.predicted,rtol=1e-12)
                    future=v.copy();future[:,:,i+1:]=1e12
                    np.testing.assert_array_equal(features,feature_matrix(future,i,int(h),profile,events,variant,macro,config['wage_hypothetical_lag_months']))
                    out=group.copy();out['model']=name;out['predicted']=predicted;records.append(out)
                print(name,'fit complete',flush=True)
    frame=pd.concat([original,*records],ignore_index=True).sort_values(KEYS+['model']).reset_index(drop=True)
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    for name,table in summarize(frame).items():table.to_csv(OUT/f'{name}.csv',index=False)
    pd.DataFrame(fit).to_csv(OUT/'training.csv',index=False)
    assert hashes==inputs_hash()
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in ['.csv','.parquet']}
    protocol={'input_sha256':hashes,'artifact_sha256':artifacts,'profile':profile.tolist(),'config':config,'rows':len(frame),'models':sorted(frame.model.unique()),'limits':'Retrospective2024, no independent test or promotion; fixed2023 fit; profile fitted on full training2023; h12 extrapolates outside fitted h1..9; lag0 design, historical expense vintages unknown; CBR frozen at origin, future decisions absent; wages2026 snapshot uses hypothetical2monthlag, historicalavailability NOTverified; earlyblend constructed with transferred.75weight chosen previously on earlywindow, so early notindependentselection.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
