"""Full posterior, boundary-before-observation BOCPD and matched detector audit."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from sberindex.paths import ROOT
from sberindex.detection.detector_settings import load_settings, validate_settings, SOURCES as SETTINGS_SOURCES
from sberindex.detection.asof_detector_audit import METHODS as OLD_METHODS, SEEDS, select_cohort, centered_signal, noise_multiplier, causal_scores, assigned
from sberindex.detection.change_detection import CATEGORIES
from sberindex.detection.event_attribution_audit import factors
from sberindex.detection.detector_delay_audit import LAGS, KEYS, calendar_metrics, release_month

METHODS=(*OLD_METHODS,'bocpd')
SOURCES=['data/consumption.parquet','docs/protocols/BOCPD_EXPERIMENT.md',
 'src/sberindex/detection/bocpd_audit.py','src/sberindex/detection/asof_detector_audit.py',
 'src/sberindex/detection/change_detection.py','src/sberindex/detection/event_attribution_audit.py',
 'src/sberindex/detection/detector_delay_audit.py',*SETTINGS_SOURCES]

def bocpd_scores(signal,sigma,return_posterior=False,settings=None):
    """p(boundary immediately before x_t | x_1:t); no comparison after gaps."""
    settings=load_settings() if settings is None else validate_settings(settings)
    mu0=settings["bocpd_mu"];k0=settings["bocpd_kappa"];a0=settings["bocpd_alpha"]
    x=np.asarray(signal,float)
    if x.ndim!=2 or not np.isfinite(sigma) or sigma<=0:raise ValueError('2D signal and positive finite sigma required')
    n,tmax=x.shape;shape=(n,tmax+1);hazard=settings["bocpd_hazard"]
    logw=np.full(shape,-np.inf);logw[:,0]=0
    mu=np.full(shape,mu0);kappa=np.full(shape,k0);alpha=np.full(shape,a0);beta=np.full(shape,sigma**2)
    known=np.zeros(n,bool);scores=np.full_like(x,np.nan);trace=[]
    for t in range(tmax):
        valid=np.isfinite(x[:,t]);value=np.where(valid,x[:,t],0)[:,None]
        scale2=beta*(kappa+1)/(alpha*kappa);df=2*alpha
        lp=gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*np.pi*scale2)-(df+1)/2*np.log1p((value-mu)**2/(df*scale2))
        reset=np.log(hazard)+lp[:,0]
        grow=logw+np.log1p(-hazard)+lp
        nextw=np.full(shape,-np.inf)
        nextw[:,1]=np.logaddexp(reset,grow[:,0])
        if tmax>1:nextw[:,2:]=grow[:,1:-1]
        norm=logsumexp(nextw,axis=1)
        scores[valid&known,t]=np.exp(reset[valid&known]-norm[valid&known])
        nextw-=norm[:,None]
        nextk=kappa+1;nextmu=(kappa*mu+value)/nextk
        nextbeta=beta+.5*kappa/nextk*(value-mu)**2;nextalpha=alpha+.5
        # State 0 retains the prior; state 1 always updates the prior with x_t.
        mu=np.column_stack((np.full(n,mu0),nextmu[:,0],nextmu[:,1:-1]))
        kappa=np.column_stack((np.full(n,k0),nextk[:,0],nextk[:,1:-1]))
        alpha=np.column_stack((np.full(n,a0),nextalpha[:,0],nextalpha[:,1:-1]))
        beta=np.column_stack((np.full(n,sigma**2),nextbeta[:,0],nextbeta[:,1:-1]))
        nextw[~valid]=-np.inf;nextw[~valid,0]=0
        mu[~valid]=mu0;kappa[~valid]=k0;alpha[~valid]=a0;beta[~valid]=sigma**2
        logw=nextw;known=valid
        if return_posterior:trace.append(np.exp(logw))
    return (scores,np.stack(trace,axis=1)) if return_posterior else scores

def prior_sigma(z,settings=None):
    settings=load_settings() if settings is None else validate_settings(settings)
    calibration=z[:,:6]
    mad=np.median(np.abs(calibration-np.median(calibration,axis=1)[:,None]),axis=1)
    return float(max(settings["bocpd_mad_factor"]*np.median(mad),settings["bocpd_sigma_floor"]))

def score(z,method,sigma,settings=None):return bocpd_scores(z,sigma,settings=settings) if method=='bocpd' else causal_scores(z,method,settings)

def build(settings=None):
    settings=load_settings() if settings is None else validate_settings(settings)
    raw=pd.read_parquet(ROOT/'data/consumption.parquet');runs=[];cal=[];individual=[];calendar=[]
    for category in CATEGORIES:
        panel=raw[raw.category.eq(category)].pivot(index='territory_id',columns='date',values='value').sort_index()
        table=select_cohort(panel);values=table.to_numpy(float);ids=table.index.to_numpy(int)
        z=centered_signal(values);observed=np.isfinite(z[:,6:])
        for delay in LAGS:
            for t in range(6):
                source=f'2024-{t+7:02d}';release=release_month(source,delay)
                calendar.append(dict(category=category,reporting_delay_months=delay,source_month=source,
                    release_month=release,calibration_ready_month=release_month('2024-06',delay),
                    observed_source_ids=int(observed[:,t].sum()),missing_source_ids=int((~observed[:,t]).sum()),
                    pending_by_december_ids=int(observed[:,t].sum()) if release>'2024-12' else 0))
        for scaling,scale in [('raw',np.ones(len(table))),('regularized_noise',noise_multiplier(z,settings))]:
            signal=z*scale[:,None];sigma=prior_sigma(signal,settings);frozen={}
            for method in METHODS:
                scores=score(signal,method,sigma,settings)
                threshold=float(np.quantile(np.max(scores[:,1:6],axis=1),settings["threshold_quantile"]))
                original=scores[:,6:]>threshold;frozen[method]=(threshold,original)
                cal.append(dict(category=category,scaling=scaling,method=method,threshold=threshold,prior_sigma=sigma,
                    eligible_ids=len(ids),calibration_alarm_rate=float((scores[:,1:6]>threshold).any(axis=1).mean())))
                individual.append(pd.DataFrame(dict(category=category,scaling=scaling,method=method,
                    territory_id=np.repeat(ids,12),target=np.tile([f'2024-{i:02d}' for i in range(1,13)],len(ids)),
                    score=scores.ravel(),threshold=threshold,alarm=(scores>threshold).ravel(),
                    source_observed=np.isfinite(signal).ravel(),score_observed=np.isfinite(scores).ravel())))
            for seed in SEEDS:
                treated=assigned(ids,seed)
                for shift in [-20,20]:
                    for shape in ['step','pulse','ramp']:
                        changed=values.copy();changed[treated,18:]*=factors(shape,shift/100)
                        shifted=centered_signal(changed)*scale[:,None];onset=int(shape=='ramp')
                        np.testing.assert_array_equal(shifted[:,:6],signal[:,:6])
                        for method,(threshold,original) in frozen.items():
                            alarms=score(shifted,method,sigma,settings)[:,6:]>threshold
                            new=alarms&~original&observed
                            for delay in LAGS:
                                runs.append(dict(category=category,scaling=scaling,seed=seed,shift_pct=shift,shape=shape,
                                    method=method,reporting_delay_months=delay,**calendar_metrics(new,original,observed,treated,onset,delay)))
        print(f'BOCPD matched audit: {category}',flush=True)
    runs=pd.DataFrame(runs);metrics=[c for c in runs if c not in KEYS+['seed']]
    return dict(runs=runs,summary=runs.groupby(KEYS,sort=True)[metrics].mean().reset_index(),
        calibration=pd.DataFrame(cal),calendar=pd.DataFrame(calendar),scores=pd.concat(individual,ignore_index=True))

def main():
    settings=load_settings();tables=build(settings)
    assert load_settings()==settings, 'Settings changed during audit'
    for name,table in tables.items():
        if name=='scores':table.to_parquet(ROOT/'results/bocpd_scores.parquet',index=False)
        else:table.to_csv(ROOT/'results'/f'bocpd_{name}.csv',index=False)
    p=dict(input_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
        rows=len(tables['runs']),detector_settings=settings,hazard=settings["bocpd_hazard"],calibration=f'Feb-Jun 2024, matched five-month maximum q{100*settings["threshold_quantile"]:g}',
        limits='Reused2024; synthetic changes; hypothetical uniform lag; no independent event truth or method promotion.')
    (ROOT/'results/bocpd_protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))

def verify():
    p=json.loads((ROOT/'results/bocpd_protocol.json').read_text());assert set(p['input_sha256'])==set(SOURCES)
    for s,h in p['input_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h,s
    assert p['detector_settings']==load_settings()
    expected=build()
    for name,table in expected.items():
        saved=pd.read_parquet(ROOT/'results/bocpd_scores.parquet') if name=='scores' else pd.read_csv(ROOT/f'results/bocpd_{name}.csv')
        pd.testing.assert_frame_equal(saved,table,check_dtype=False,rtol=1e-10,atol=1e-12)
    runs=expected['runs'];assert len(runs)==p['rows']==10800
    assert not runs.duplicated(KEYS+['seed']).any()
    pairs=[c for c in KEYS if c not in ['method','reporting_delay_months']]+['seed']
    for _,g in runs.groupby(pairs):
        assert g.evaluated_first_three_n.nunique()==g.treated_n.nunique()==1
    for _,g in runs.groupby([c for c in KEYS if c!='reporting_delay_months']+['seed']):
        for col in ['new_by_first_calendar_month','new_by_second_calendar_month','new_by_third_calendar_month','new_by_december']:
            assert (np.diff(g.sort_values('reporting_delay_months')[col])<=1e-12).all()
    return True

if __name__=='__main__':
    import sys
    print(verify()) if '--verify' in sys.argv else main()
