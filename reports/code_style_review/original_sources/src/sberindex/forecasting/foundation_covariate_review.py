"""Fixed actual foundation experiments with explicit municipal category groups."""
import hashlib
import json
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sberindex.paths import ROOT
from sberindex.detection.change_detection import CATEGORIES

OUT=ROOT/'reports/foundation_covariate_review'
KEYS=['category','territory_id','origin','target','horizon']
INPUTS=['data/consumption.parquet','results/asof_cohort_protocol.json',
 'configs/foundation_covariate_review.json','docs/protocols/FOUNDATION_COVARIATE_REVIEW_EXPERIMENT.md',
 'src/sberindex/forecasting/foundation_covariate_review.py','src/sberindex/detection/change_detection.py',
 'src/sberindex/paths.py','tests/test_foundation_covariate_review.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pooled_profiles(panels):
 profiles=[];eligible={}
 for category,panel in panels.items():
  train=panel.iloc[:,:12].to_numpy(float)
  valid=(np.isfinite(train)&(train>0)).all(axis=1)
  sums=train[valid].sum(axis=0)
  if not valid.any() or not (sums>0).all():raise ValueError('positive2023 history needed')
  profiles.append(sums/sums.mean());eligible[category]=panel.index[valid].tolist()
 return np.stack(profiles),eligible

def rss_gib(value,platform):
 """resource.ru_maxrss is bytes on macOS and KiB on Linux."""
 if platform=='darwin':return float(value)/1024**3
 if platform.startswith('linux'):return float(value)/1024**2
 raise ValueError('RSS units are defined here only for macOS and Linux')

def active_mask(values,origin):
 history=np.asarray(values)[:,:,:origin+1]
 return (np.isfinite(history)&(history>0)).all(axis=(1,2))

def make_inputs(values,origin,horizon,profiles,grouped,covariates):
 """Only historical expenses enter target; future covariates are fixed profiles."""
 history=np.asarray(values,dtype=np.float32)[:,:,:origin+1]
 profiles=np.asarray(profiles,dtype=np.float32)
 if profiles.shape!=(history.shape[1],12):raise ValueError('one12-month profile per category')
 tasks=[]
 for row in history:
  if grouped:
   item={'target':row.copy()}
   cats=range(len(profiles))
   if covariates:
    item['past_covariates']={f'profile_{k}':profiles[k,np.arange(origin+1)%12].copy() for k in cats}
    item['future_covariates']={f'profile_{k}':profiles[k,(origin+1+np.arange(horizon))%12].copy() for k in cats}
   tasks.append(item)
  else:
   for k,target in enumerate(row):
    item={'target':target.copy()}
    if covariates:
     item['past_covariates']={'profile':profiles[k,np.arange(origin+1)%12].copy()}
     item['future_covariates']={'profile':profiles[k,(origin+1+np.arange(horizon))%12].copy()}
    tasks.append(item)
 return tasks

def load_data():
 raw=pd.read_parquet(ROOT/'data/consumption.parquet')
 columns=pd.period_range('2023-01','2024-12',freq='M').astype(str).tolist()
 panels={c:raw[raw.category.eq(c)].pivot(index='territory_id',columns='date',values='value').sort_index().reindex(columns=columns) for c in CATEGORIES}
 profiles,eligible=pooled_profiles(panels)
 ids=np.array(json.loads((ROOT/'results/asof_cohort_protocol.json').read_text())['sample_ids'],int)
 values=np.stack([p.loc[ids].to_numpy(float) for p in panels.values()],axis=1)
 return ids,values,profiles,eligible

def metrics(g):
 error=(g.actual-g.predicted).abs()
 within=g.actual-g.groupby('territory_id').actual.transform('mean');den=float((within**2).sum())
 yoy_a=100*(g.actual/g.year_ago-1);yoy_p=100*(g.predicted/g.year_ago-1)
 return dict(MAE=float(error.groupby(g.target).mean().mean()),MAE_pooled=float(error.mean()),
  R2_pooled=float(r2_score(g.actual,g.predicted)),R2_within_MO=1-float(((g.actual-g.predicted)**2).sum())/den if den>0 else np.nan,
  WAPE_pct=float(100*error.sum()/g.actual.sum()),YoY_MAE_pp=float(abs(yoy_a-yoy_p).groupby(g.target).mean().mean()),
  YoY_R2=float(r2_score(yoy_a,yoy_p)),dates=int(g.target.nunique()),municipalities=int(g.territory_id.nunique()),observations=len(g))

def summaries(frame):
 summary=pd.DataFrame([dict(category=c,horizon=int(h),model=m,**metrics(g)) for (c,h,m),g in frame.groupby(['category','horizon','model'])])
 for name in ['seasonal_naive','seasonal_pooled']:
  reference=summary[summary.model.eq(name)].set_index(['category','horizon']).MAE
  summary['skill_vs_'+name]=1-summary.MAE/[reference.loc[c,h] for c,h in zip(summary.category,summary.horizon)]
 monthly=pd.DataFrame([dict(category=c,horizon=int(h),model=m,target=t,MAE=float(abs(g.actual-g.predicted).mean()),observations=len(g)) for (c,h,m,t),g in frame.groupby(['category','horizon','model','target'])])
 comparisons=[('chronos2_profile','chronos2_univariate'),('chronos2_mo_group','chronos2_univariate'),('chronos2_mo_group_profile','chronos2_mo_group'),('chronos2_mo_group_profile','chronos2_profile'),('bolt_base_univariate','chronos2_univariate')]
 paired=[]
 for model,reference in comparisons:
  left=monthly[monthly.model.eq(model)];right=monthly[monthly.model.eq(reference)]
  joined=left.merge(right,on=['category','horizon','target'],suffixes=('_new','_ref'),validate='one_to_one')
  for r in joined.itertuples():paired.append(dict(category=r.category,horizon=r.horizon,target=r.target,model=model,reference=reference,MAE_difference=r.MAE_new-r.MAE_ref))
 return dict(summary=summary,monthly=monthly,paired=pd.DataFrame(paired))

def group_check(tasks,grouped,covariates,horizon):
 from chronos.chronos2.dataset import Chronos2Dataset,DatasetMode
 n=6 if grouped else 1;nvars=n+(6 if grouped else 1 if covariates else 0) if covariates else n
 ds=Chronos2Dataset(tasks[:2],context_length=24,prediction_length=horizon,batch_size=96,output_patch_size=16,mode=DatasetMode.TEST)
 batch=next(iter(ds));groups=batch['group_ids'].numpy()
 unique,counts=np.unique(groups,return_counts=True)
 assert len(unique)==2 and np.array_equal(counts,[nvars,nvars]),(groups,counts,nvars)
 return dict(grouped=grouped,covariates=covariates,target_variates=n,total_variates_per_task=nvars,distinct_groups=2,group_ids=groups.tolist())

def run():
 import gc,resource,torch,chronos,inspect
 from importlib.metadata import version
 from chronos import Chronos2Pipeline,ChronosBoltPipeline
 from huggingface_hub import snapshot_download
 started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/'configs/foundation_covariate_review.json').read_text())
 frozen={p:sha(ROOT/p) for p in INPUTS}
 (OUT/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
 ids,values,profiles,eligible=load_data();assert len(ids)==cfg['sample_size']==256
 torch.set_num_threads(cfg['torch_threads']);torch.set_num_interop_threads(1)
 runtime=[];coverage=[];records=[];checks=[];artifacts={}
 def memory():return rss_gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sys.platform)
 def append_forecasts(predicted,active_ids,active_values,origin,horizons,model):
  assert predicted.shape==(len(active_ids),6,max(horizons)) and np.isfinite(predicted).all()
  for h in horizons:
   target=origin+h
   for k,c in enumerate(CATEGORIES):
    actual=active_values[:,k,target];valid=np.isfinite(actual)&(actual>0)
    for i,a,p,year,base in zip(active_ids[valid],actual[valid],predicted[valid,k,h-1],active_values[valid,k,target-12],active_values[valid,k,origin]):
     records.append(dict(category=c,territory_id=int(i),origin=str(pd.Period('2023-01','M')+origin),target=str(pd.Period('2023-01','M')+target),horizon=h,model=model,actual=float(a),predicted=float(max(0,p)),year_ago=float(year),origin_actual=float(base)))
 for stage in ['chronos2','bolt']:
  tick=time.monotonic();cls=Chronos2Pipeline if stage=='chronos2' else ChronosBoltPipeline
  cache=Path(snapshot_download(cfg[stage+'_model'],revision=cfg[stage+'_revision'],local_files_only=True))
  artifacts[stage]={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in cache.iterdir() if p.is_file()}
  pipeline=cls.from_pretrained(str(cache),device_map='cpu',local_files_only=True)
  runtime.append(dict(model=stage,stage='load',origin='',seconds=time.monotonic()-tick,peak_rss_GiB=memory(),municipalities=0,tasks=0))
  for origin in cfg['origin_indices']:
   horizons=[h for h in cfg['horizons'] if origin+h<24];horizon=max(horizons)
   active=active_mask(values,origin);active_ids=ids[active];v=values[active]
   if stage=='chronos2':
    coverage.append(dict(origin=str(pd.Period('2023-01','M')+origin),selected_ids=len(ids),history_eligible_ids=int(active.sum()),history_excluded_ids=int((~active).sum()),max_horizon=horizon))
    variants=[('chronos2_univariate',False,False),('chronos2_profile',False,True),('chronos2_mo_group',True,False),('chronos2_mo_group_profile',True,True)]
    for model,grouped,covariates in variants:
     tick=time.monotonic();tasks=make_inputs(v,origin,horizon,profiles,grouped,covariates)
     if origin==11:checks.append(group_check(tasks,grouped,covariates,horizon))
     with torch.inference_mode():pred=pipeline.predict(tasks,prediction_length=horizon,batch_size=cfg['batch_size_series'],cross_learning=False)
     q=pipeline.quantiles.index(cfg['quantile'])
     predicted=np.stack([p[:,q,:].detach().cpu().numpy() for p in pred]).reshape(len(v),6,horizon)
     append_forecasts(predicted,active_ids,v,origin,horizons,model)
     runtime.append(dict(model=model,stage='inference',origin=str(pd.Period('2023-01','M')+origin),seconds=time.monotonic()-tick,peak_rss_GiB=memory(),municipalities=len(v),tasks=len(tasks)))
     print(f'{model} {origin}: {runtime[-1]["seconds"]:.2f}s; peak{memory():.2f}GiB',flush=True)
    for model in ['seasonal_naive','seasonal_pooled']:
     predicted=np.stack([v[:,:,origin+h-12] if model=='seasonal_naive' else v[:,:,origin]*profiles[:,(origin+h)%12]/profiles[:,origin%12] for h in range(1,horizon+1)],axis=2)
     append_forecasts(predicted,active_ids,v,origin,horizons,model)
   else:
    tick=time.monotonic();history=v[:,:,:origin+1].reshape(-1,origin+1).astype('float32');parts=[]
    with torch.inference_mode():
     for start in range(0,len(history),cfg['bolt_batch_size']):
      parts.append(pipeline.predict(torch.from_numpy(history[start:start+cfg['bolt_batch_size']]),prediction_length=horizon)[:,4,:].detach().cpu().numpy())
    predicted=np.concatenate(parts).reshape(len(v),6,horizon)
    append_forecasts(predicted,active_ids,v,origin,horizons,'bolt_base_univariate')
    runtime.append(dict(model='bolt_base_univariate',stage='inference',origin=str(pd.Period('2023-01','M')+origin),seconds=time.monotonic()-tick,peak_rss_GiB=memory(),municipalities=len(v),tasks=len(history)))
    print(f'bolt base {origin}: {runtime[-1]["seconds"]:.2f}s; peak{memory():.2f}GiB',flush=True)
   # Save progress, without evaluating any new numeric performance until all variants complete.
   pd.DataFrame(records).to_parquet(OUT/'predictions.parquet',index=False)
   pd.DataFrame(runtime).to_csv(OUT/'runtime.csv',index=False)
  del pipeline;gc.collect()
 frame=pd.DataFrame(records).sort_values(KEYS+['model']).reset_index(drop=True)
 frame.to_parquet(OUT/'predictions.parquet',index=False)
 pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
 for name,table in summaries(frame).items():table.to_csv(OUT/f'{name}.csv',index=False)
 package=Path(chronos.__file__).parent
 installed={str(p.relative_to(package)):sha(p) for p in [package/'chronos2/pipeline.py',package/'chronos2/dataset.py',package/'chronos2/preprocess.py',package/'chronos_bolt.py']}
 audit=dict(status='complete',seconds=time.monotonic()-started,chronos_version=version('chronos-forecasting'),torch_version=torch.__version__,
  input_sha256=frozen,profiles=profiles.tolist(),profile_eligible_ids=eligible,sample_ids=ids.tolist(),model_artifacts=artifacts,installed_source_sha256=installed,
  api_predict_signature=str(inspect.signature(Chronos2Pipeline.predict)),explicit_dataset_groups=checks,torch_threads=cfg['torch_threads'],peak_rss_GiB=memory(),
  new_fit_count=0,platform=sys.platform,python_version=sys.version,model_revisions={k:cfg[k] for k in ['chronos2_revision','bolt_revision']},cross_learning=False,grouping='six categories within eachMO; separate task group_ids perMO',
  limits='Retrospective reused2024; later pretrained checkpoint, training contamination not excluded; zero reporting lag; h12 one target date; no model selection;256-ID sample not full2075.')
 audit['output_sha256']={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.csv','.parquet']}
 (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 print(verify(),flush=True)


def verify():
 audit=json.loads((OUT/'audit.json').read_text())
 for name,digest in audit['input_sha256'].items():assert sha(ROOT/name)==digest,name
 for name,digest in audit['output_sha256'].items():assert sha(OUT/name)==digest,name
 ids,values,profiles,eligible=load_data();np.testing.assert_array_equal(profiles,audit['profiles'])
 altered=values.copy();altered[:,:,12:]=np.nan
 for origin in range(11,23):
  modified=values.copy();modified[:,:,origin+1:]=1e30
  np.testing.assert_array_equal(active_mask(values,origin),active_mask(modified,origin))
  for grouped,covariates in [(False,False),(False,True),(True,False),(True,True)]:
   a=make_inputs(values,origin,24-origin-1,profiles,grouped,covariates);b=make_inputs(modified,origin,24-origin-1,profiles,grouped,covariates)
   for x,y in zip(a,b):
    np.testing.assert_array_equal(x['target'],y['target'])
    for section in ['past_covariates','future_covariates']:
     for k in x.get(section,{}):np.testing.assert_array_equal(x[section][k],y[section][k])
 frame=pd.read_parquet(OUT/'predictions.parquet');assert not frame.duplicated(KEYS+['model']).any()
 assert np.isfinite(frame[['actual','predicted','year_ago','origin_actual']]).all().all()
 ref=frame[frame.model.eq('seasonal_pooled')].sort_values(KEYS).reset_index(drop=True)
 for model,g in frame.groupby('model'):
  g=g.sort_values(KEYS).reset_index(drop=True)
  pd.testing.assert_frame_equal(g[KEYS+['actual']],ref[KEYS+['actual']])
 for r in ref.itertuples():
  i=int(np.flatnonzero(ids==r.territory_id)[0]);k=CATEGORIES.index(r.category);t=(pd.Period(r.target,'M')-pd.Period('2023-01','M')).n
  assert r.actual==values[i,k,t] and r.year_ago==values[i,k,t-12]
 for name,table in summaries(frame).items():pd.testing.assert_frame_equal(pd.read_csv(OUT/f'{name}.csv'),table,check_dtype=False,rtol=1e-10,atol=1e-10)
 return dict(saved_hashes=True,common_pairs=True,actuals=True,future_input_invariance=True,profile_invariance=True,saved_metrics=True)

if __name__=='__main__':
 import sys
 print(verify()) if '--verify' in sys.argv else run()
