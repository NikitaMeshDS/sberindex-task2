"""Frozen foundation/HGB comparison under hypothetical reporting delays."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sberindex.paths import ROOT
from sberindex.forecasting.foundation_seasonal_audit import known_2023_profile, restore_forecast
from sberindex.forecasting.asof_distribution_audit import paired_metrics, validate_actuals

OUT = ROOT/'results'
TARGETS = ['2024-09','2024-10','2024-11','2024-12']
MODELS = ['prophet','seasonal_pooled','blend_75','hgb_frozen','chronos_bolt_tiny',
          'chronos_2','chronos_bolt_tiny_seasonal','chronos_2_seasonal']
KEYS = ['territory_id','origin','target','horizon']
CACHE = ['foundation_delay_predictions.parquet','foundation_delay_protocol.json','foundation_delay_runtime.csv']
INPUTS = ['config.json','data/consumption.parquet','results/asof_cohort_protocol.json',
          'results/asof_reporting_delay_predictions.parquet','results/foundation_seasonal_predictions.parquet',
          'results/calendar_predictions.parquet','docs/protocols/FOUNDATION_DELAY_EXPERIMENT.md',
          'src/sberindex/forecasting/foundation_delay_audit.py',
          'src/sberindex/forecasting/foundation_seasonal_audit.py',
          'src/sberindex/forecasting/calendar_audit.py',
          'src/sberindex/forecasting/asof_distribution_audit.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing(target, delay):
    target = pd.Period(target, freq='M')
    return str(target-1), str(target-1-delay), 1+delay


def history_context(panel, origin, profile):
    history = panel.loc[:,panel.columns <= origin]
    valid = np.isfinite(history.to_numpy(float)).all(axis=1) & history.gt(0).all(axis=1)
    return history.loc[valid].div(np.asarray(profile)[np.arange(history.shape[1]) % 12],axis=1)


def restore_target(predictions, origin, profile):
    origin_index = pd.Period(origin, freq='M').ordinal-pd.Period('2023-01',freq='M').ordinal
    return restore_forecast(predictions,origin_index,np.asarray(profile))


def canonical(frame):
    frame = frame[KEYS+['model','actual','predicted']].copy()
    frame['territory_id'] = frame.territory_id.astype('int64')
    frame['horizon'] = frame.horizon.astype('int64')
    frame['reporting_delay_months'] = frame.horizon-1
    frame['decision_month'] = (pd.PeriodIndex(frame.target,freq='M')-1).astype(str)
    frame['business_horizon_months'] = 1
    return frame


def common_pairs(frame, models=MODELS):
    assert set(frame.model) == set(models)
    assert set(frame.reporting_delay_months) == {0,1,2}
    assert not frame.duplicated(['territory_id','target','model','reporting_delay_months']).any()
    counts = frame.groupby(['territory_id','target']).size()
    common = counts[counts.eq(3*len(models))].index
    selected = frame.set_index(['territory_id','target']).loc[common].reset_index()
    assert len(selected) and selected.groupby(['territory_id','target']).actual.nunique().eq(1).all()
    return selected.sort_values(['reporting_delay_months','model','target','territory_id']).reset_index(drop=True)


def score(frame, panel):
    frame = frame.copy()
    frame['absolute_error'] = (frame.actual-frame.predicted).abs()
    means = panel.loc[:,panel.columns.str.startswith('2023-')].mean(axis=1)
    frame['normalized_error_2023'] = frame.absolute_error/frame.territory_id.map(means)
    return frame


def summaries(frame):
    summary,monthly,paired,sensitivity = [],[],[],[]
    for (delay,model),group in frame.groupby(['reporting_delay_months','model'],sort=True):
        labels = {'reporting_delay_months':int(delay),'model':model,'business_horizon_months':1,'effective_horizon':int(delay)+1}
        summary.append({**labels,'MAE_date_balanced':float(group.groupby('target').absolute_error.mean().mean()),
            'MAE_pooled':float(group.absolute_error.mean()),'R2_pooled':float(r2_score(group.actual,group.predicted)),
            'WAPE_pct':float(100*group.absolute_error.sum()/group.actual.sum()),
            'NMAE_2023_pct':float(100*group.groupby('target').normalized_error_2023.mean().mean()),
            'dates':group.target.nunique(),'municipalities':group.territory_id.nunique(),'observations':len(group)})
        for target,part in group.groupby('target'):
            monthly.append({**labels,'target':target,'MAE':float(part.absolute_error.mean()),'municipalities':len(part)})
        references = {'seasonal_pooled','prophet'}
        if model.endswith('_seasonal'):references.add(model.removesuffix('_seasonal'))
        for reference in sorted(references-{model}):
            base = frame[frame.reporting_delay_months.eq(delay) & frame.model.eq(reference)].sort_values(KEYS).reset_index(drop=True)
            candidate = group.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(candidate[KEYS],base[KEYS])
            np.testing.assert_array_equal(candidate.actual,base.actual)
            candidate['difference'] = candidate.absolute_error-base.absolute_error
            paired.append({**labels,'reference':reference,**paired_metrics(candidate)})
            for omitted in TARGETS:
                sensitivity.append({**labels,'reference':reference,'excluded_target':omitted,
                                    **paired_metrics(candidate[candidate.target.ne(omitted)])})
    return {name:pd.DataFrame(rows) for name,rows in [('summary',summary),('monthly',monthly),('paired',paired),('sensitivity',sensitivity)]}


def panel_inputs(root=ROOT):
    raw = pd.read_parquet(root/'data/consumption.parquet')
    panel = raw[raw.category.eq('Все категории')].pivot(index='territory_id',columns='date',values='value').sort_index()
    profile,eligible = known_2023_profile(panel)
    assert len(eligible) == 2075
    ids = json.loads((root/'results/asof_cohort_protocol.json').read_text())['sample_ids']
    assert len(ids) == 256 and set(ids).issubset(eligible)
    return panel,profile,ids,raw


def reused(root=ROOT):
    base = pd.read_parquet(root/'results/asof_reporting_delay_predictions.parquet')
    foundation = pd.read_parquet(root/'results/foundation_seasonal_predictions.parquet')
    foundation = foundation[foundation.model.str.startswith('chronos') & foundation.target.isin(TARGETS) & foundation.horizon.isin([1,3])]
    hgb = pd.read_parquet(root/'results/calendar_predictions.parquet')
    hgb = hgb[hgb.model.eq('hgb_frozen') & hgb.target.isin(TARGETS) & hgb.horizon.isin([1,3])]
    return pd.concat([canonical(base),canonical(foundation),canonical(hgb)],ignore_index=True)


def infer_chronos(pipeline, name, context, horizon, origin):
    import torch
    if name == 'chronos_bolt_tiny':
        parts=[]
        for start in range(0,len(context),64):
            tensor=torch.from_numpy(context.iloc[start:start+64].to_numpy(np.float32,copy=True))
            parts.append(pipeline.predict(tensor,prediction_length=horizon)[:,4,:].detach().cpu().numpy())
        return np.concatenate(parts)
    long=context.stack().rename('value').reset_index()
    long.columns=['item_id','date','value'];long['timestamp']=pd.to_datetime(long.pop('date')+'-01')
    predicted=pipeline.predict_df(long,target='value',prediction_length=horizon,quantile_levels=[.5],batch_size=64,freq='MS')
    predicted['timestamp']=pd.to_datetime(predicted.timestamp)
    months=pd.date_range(pd.Period(origin,freq='M').start_time,periods=horizon+1,freq='MS')[1:]
    return np.column_stack([predicted[predicted.timestamp.eq(month)].set_index('item_id').loc[context.index,'0.5'].to_numpy(float) for month in months])


def infer():
    import torch
    from chronos import Chronos2Pipeline,ChronosBoltPipeline
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    from sberindex.forecasting.calendar_audit import features,predict_asof
    panel,profile,ids,raw = panel_inputs()
    sample=panel.loc[ids]
    config=json.loads((ROOT/'config.json').read_text())
    torch.set_num_threads(2)
    records=[];runtime=[];pipes={}
    for name,cls,m,r in [('chronos_bolt_tiny',ChronosBoltPipeline,'chronos_model','chronos_revision'),
                         ('chronos_2',Chronos2Pipeline,'chronos2_model','chronos2_revision')]:
        start=time.perf_counter()
        pipes[name]=cls.from_pretrained(config[m],revision=config[r],device_map='cpu',local_files_only=True)
        runtime.append({'model':name,'stage':'load','origin':'','seconds':time.perf_counter()-start,'inference_ids':0})
    months=pd.period_range('2023-01',periods=24,freq='M')
    _,eligible=known_2023_profile(panel)
    train=panel.loc[eligible].iloc[:,:12].to_numpy(float)
    with threadpool_limits(limits=2):
        x=np.concatenate([features(np.log(train[:,:t]),months[t],False) for t in range(3,12)])
        y=np.concatenate([np.log(train[:,t])-np.log(train[:,t-1]) for t in range(3,12)])
        hgb=HistGradientBoostingRegressor(max_iter=config['hgb_max_iter'],max_leaf_nodes=config['hgb_max_leaf_nodes'],
            learning_rate=config['hgb_learning_rate'],min_samples_leaf=config['hgb_min_samples_leaf'],
            l2_regularization=config['hgb_l2_regularization'],random_state=config['random_seed']).fit(x,y)
        for target in TARGETS:
            decision,origin,horizon=timing(target,1)
            origin_index=months.get_loc(pd.Period(origin,freq='M'))
            active=history_context(sample,origin,np.ones(12))
            for name,pipe in pipes.items():
                for normalized in [False,True]:
                    context=history_context(sample,origin,profile if normalized else np.ones(12))
                    start=time.perf_counter()
                    with torch.inference_mode():predictions=infer_chronos(pipe,name,context,horizon,origin)
                    predictions=restore_target(predictions,origin,profile) if normalized else np.maximum(0.,predictions)
                    assert np.isfinite(predictions).all()
                    model=name+('_seasonal' if normalized else '')
                    runtime.append({'model':model,'stage':'inference','origin':origin,'seconds':time.perf_counter()-start,'inference_ids':len(context)})
                    for city,prediction in zip(context.index,predictions[:,-1]):
                        actual=sample.loc[city,target]
                        if np.isfinite(actual):records.append({'territory_id':int(city),'origin':origin,'target':target,'horizon':horizon,'model':model,'actual':float(actual),'predicted':float(prediction)})
            spending=sample.loc[active.index].to_numpy(float)
            start=time.perf_counter()
            predicted=predict_asof(hgb,spending,origin_index,2,months,False)
            runtime.append({'model':'hgb_frozen','stage':'inference','origin':origin,'seconds':time.perf_counter()-start,'inference_ids':len(active)})
            for city,prediction in zip(active.index,predicted[:,-1]):
                actual=sample.loc[city,target]
                if np.isfinite(actual):records.append({'territory_id':int(city),'origin':origin,'target':target,'horizon':2,'model':'hgb_frozen','actual':float(actual),'predicted':float(prediction)})
            print(f'Fixed inference through {origin}; target {target}; {len(active)} IDs',flush=True)
        # Check the fitted HGB matches the previous h1/h3 cache on every reused pair.
        reference=reused();reference=reference[reference.model.eq('hgb_frozen')]
        for origin,group in reference.groupby('origin'):
            city_ids=group.territory_id.unique();index=months.get_loc(pd.Period(origin,freq='M'))
            predictions=predict_asof(hgb,sample.loc[city_ids].to_numpy(float),index,int(group.horizon.max()),months,False)
            positions={int(city):i for i,city in enumerate(city_ids)}
            computed=[predictions[positions[r.territory_id],r.horizon-1] for r in group.itertuples()]
            np.testing.assert_allclose(computed,group.predicted,rtol=1e-10)
    new=canonical(pd.DataFrame(records))
    comparison=common_pairs(pd.concat([reused(),new],ignore_index=True))
    comparison=score(comparison,panel)
    comparison.to_parquet(OUT/CACHE[0],index=False)
    pd.DataFrame(runtime).to_csv(OUT/CACHE[2],index=False)
    protocol={'input_sha256':{p:sha(ROOT/p) for p in INPUTS},
        'cached_artifact_sha256':{p:sha(OUT/p) for p in [CACHE[0],CACHE[2]]},
        'profile_factors':profile.tolist(),'profile_ids':eligible.astype(int).tolist(),'sample_ids':ids,
        'models':MODELS,'targets':TARGETS,'delays':[0,1,2],'business_horizon':1,
        'chronos_bolt_revision':config['chronos_revision'],'chronos_2_revision':config['chronos2_revision'],
        'batch_size':64,'torch_threads':2,'local_files_only':True,'hgb_training_rows':len(y),
        'new_inference':'h2 at reporting delay1; delays0/2 reused exactly; raw and seasonal contexts separate',
        'limits':'Hypothetical release lags; reused2024; four dates; no selection or independent test; pretrained historical overlap unknown.'}
    (OUT/CACHE[1]).write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    refresh()


def refresh(root=ROOT):
    protocol=json.loads((root/'results'/CACHE[1]).read_text())
    for path,digest in protocol['input_sha256'].items():
        assert sha(root/path)==digest,path
    for path,digest in protocol['cached_artifact_sha256'].items():
        assert sha(root/'results'/path)==digest,path
    frame=pd.read_parquet(root/'results'/CACHE[0])
    for name,table in summaries(frame).items():
        table.to_csv(root/'results'/f'foundation_delay_{name}.csv',index=False)


def verify():
    protocol=json.loads((OUT/CACHE[1]).read_text())
    for path,digest in protocol['input_sha256'].items():assert sha(ROOT/path)==digest,path
    assert set(protocol['cached_artifact_sha256']) == {CACHE[0],CACHE[2]}
    for path,digest in protocol['cached_artifact_sha256'].items():assert sha(OUT/path)==digest,path
    panel,profile,ids,raw=panel_inputs()
    np.testing.assert_allclose(profile,protocol['profile_factors'])
    np.testing.assert_array_equal(known_2023_profile(panel)[1],protocol['profile_ids'])
    assert ids==protocol['sample_ids']
    config=json.loads((ROOT/'config.json').read_text())
    assert protocol['chronos_bolt_revision']==config['chronos_revision']
    assert protocol['chronos_2_revision']==config['chronos2_revision']
    frame=pd.read_parquet(OUT/CACHE[0])
    assert set(frame.model)==set(MODELS) and frame.target.isin(TARGETS).all()
    assert np.isfinite(frame[['actual','predicted','absolute_error','normalized_error_2023']].to_numpy()).all()
    assert frame.predicted.ge(0).all()
    pd.testing.assert_frame_equal(frame,common_pairs(frame))
    for delay,group in frame.groupby('reporting_delay_months'):
        for target,part in group.groupby('target'):
            decision,origin,horizon=timing(target,int(delay))
            assert part.decision_month.eq(decision).all() and part.origin.eq(origin).all()
            assert part.horizon.eq(horizon).all() and part.business_horizon_months.eq(1).all()
        for model,part in group.groupby('model'):
            base=group[group.model.eq('seasonal_pooled')].sort_values(KEYS).reset_index(drop=True)
            part=part.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(part[KEYS],base[KEYS])
            np.testing.assert_array_equal(part.actual,base.actual)
            validate_actuals(part,raw[raw.category.eq('Все категории')])
    before=frame[frame.model.eq('prophet')][['territory_id','target']].drop_duplicates().sort_values(['territory_id','target']).reset_index(drop=True)
    old=pd.read_parquet(OUT/'asof_reporting_delay_predictions.parquet')[['territory_id','target']].drop_duplicates().sort_values(['territory_id','target']).reset_index(drop=True)
    pd.testing.assert_frame_equal(before,old,check_dtype=False)
    after=score(frame.drop(columns=['absolute_error','normalized_error_2023']),panel)
    np.testing.assert_array_equal(frame.absolute_error,after.absolute_error)
    np.testing.assert_array_equal(frame.normalized_error_2023,after.normalized_error_2023)
    old_predictions=reused()
    saved_reuse=frame[(frame.model.isin(['prophet','seasonal_pooled','blend_75'])) | frame.reporting_delay_months.isin([0,2])]
    compare=['territory_id','target','model','reporting_delay_months']
    old_predictions=old_predictions.set_index(compare).loc[pd.MultiIndex.from_frame(saved_reuse[compare])]
    np.testing.assert_array_equal(saved_reuse.predicted,old_predictions.predicted)
    for origin in sorted(frame.origin.unique()):
        changed=panel.loc[ids].copy();changed.loc[:,changed.columns>origin]=np.nan
        pd.testing.assert_frame_equal(history_context(panel.loc[ids],origin,profile),history_context(changed,origin,profile))
    for name,expected in summaries(frame).items():
        pd.testing.assert_frame_equal(pd.read_csv(OUT/f'foundation_delay_{name}.csv'),expected,check_dtype=False,rtol=1e-11,atol=1e-9)
    return True


if __name__=='__main__':
    import sys
    if '--refresh' in sys.argv:refresh()
    elif '--verify' in sys.argv:print(verify())
    else:infer()
