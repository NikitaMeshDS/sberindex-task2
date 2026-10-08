"""Calibration-available detector cohort and regularized noise sensitivity.

Missing scores are unobserved, sequential states reset after a gap. Synthetic
assignments are ID-stable across cohorts. No late threshold/model selection.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.detection.detector_settings import load_settings, validate_settings, SOURCES as SETTINGS_SOURCES
from sberindex.detection.change_detection import CATEGORIES
from sberindex.detection.event_attribution_audit import factors

METHODS = ('spike','rolling_3m','ewma','cusum')
SEEDS = range(20261001,20261011)
SOURCES=['data/consumption.parquet','src/sberindex/detection/asof_detector_audit.py',
    'src/sberindex/detection/change_detection.py','src/sberindex/detection/event_attribution_audit.py',*SETTINGS_SOURCES]


def select_cohort(panel):
    return panel.loc[panel.iloc[:,:18].notna().all(axis=1)].sort_index()


def centered_signal(values):
    growth = np.log(values[:,12:24]/values[:,:12])
    centered = growth-np.nanmedian(growth,axis=0,keepdims=True)
    return centered-np.median(centered[:,:6],axis=1,keepdims=True)


def noise_multiplier(signal,settings=None):
    settings=load_settings() if settings is None else validate_settings(settings)
    weight=settings["noise_individual_weight"];floor=settings["noise_floor"]
    calibration = signal[:,:6]
    individual = np.median(abs(calibration-np.median(calibration,axis=1,keepdims=True)),axis=1)
    pooled = max(float(np.median(individual)),floor)
    return pooled/np.maximum(weight*individual+(1-weight)*pooled,floor)


def causal_scores(signal,method,settings=None):
    settings=load_settings() if settings is None else validate_settings(settings)
    alpha=settings["ewma_alpha"];drift=settings["cusum_drift"]
    if method == 'spike':
        return abs(signal)
    n,months = signal.shape
    out = np.full_like(signal,np.nan)
    state = np.zeros(n);pos = np.zeros(n);neg = np.zeros(n);run = np.zeros(n,dtype=int)
    for t in range(months):
        valid = np.isfinite(signal[:,t]);value = np.where(valid,signal[:,t],0.)
        run = np.where(valid,run+1,0)
        if method == 'ewma':
            state = np.where(valid,alpha*value+(1-alpha)*state,0.)
            out[valid,t] = abs(state[valid])
        elif method == 'cusum':
            pos = np.where(valid,np.maximum(0,pos+value-drift),0.)
            neg = np.where(valid,np.maximum(0,neg-value-drift),0.)
            out[valid,t] = np.maximum(pos,neg)[valid]
        elif method == 'rolling_3m':
            total = np.zeros(n)
            for lag in range(min(t+1,3)):
                total += np.where(run>lag,signal[:,t-lag],0.)
            out[valid,t] = abs(total[valid]/np.minimum(run[valid],3))
        else:
            raise ValueError(method)
    return out


def assigned(ids,seed):
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}:{int(i)}'.encode()).digest()[:8],'big')/2**64 < .25 for i in ids])


def event_metrics(original,changed,observed,treated,scope,onset):
    new = changed & ~original
    active = treated & scope;control = ~treated & scope
    result = dict(treated_n=int(active.sum()),control_n=int(control.sum()))
    for length,label in [(1,'onset'),(2,'second'),(3,'third')]:
        complete = observed[:,onset:onset+length].all(axis=1)
        evaluated = active & complete
        result['observed_'+label+'_n'] = int(evaluated.sum())
        result['new_by_'+label] = float(new[evaluated,onset:onset+length].any(axis=1).mean()) if evaluated.any() else np.nan
    eligible = active & observed[:,onset:].any(axis=1)
    first = np.where(new[eligible,onset:].any(axis=1),new[eligible,onset:].argmax(axis=1),np.nan)
    result['new_any_observed'] = float(np.isfinite(first).mean())
    result['median_delay_if_new'] = float(np.nanmedian(first)) if np.isfinite(first).any() else np.nan
    result['unobserved_treated_months'] = int((~observed[active,onset:]).sum())
    result['new_control_per100_observed_months'] = float(100*new[control,onset:].sum()/observed[control,onset:].sum())
    result['original_control_per100_observed_months'] = float(100*original[control,onset:].sum()/observed[control,onset:].sum())
    return result


def verify():
    from sberindex.detection.change_detection import score
    reference_settings=load_settings();reference_settings.update(ewma_alpha=.45,cusum_drift=.025)
    signal = np.random.default_rng(12).normal(size=(8,12))
    for method in METHODS:
        np.testing.assert_allclose(causal_scores(signal,method,reference_settings),score(signal,method))
    signal[:,6:] = np.nan
    for method in METHODS:
        assert np.isnan(causal_scores(signal,method)[:,6:]).all()
    values = np.arange(1,97,dtype=float).reshape(4,24)+100
    old = centered_signal(values)
    altered = values.copy();altered[:2,18:] *= 20
    new = centered_signal(altered)
    np.testing.assert_array_equal(old[:,:6],new[:,:6])
    np.testing.assert_array_equal(noise_multiplier(old),noise_multiplier(new))
    ids = np.arange(100)
    np.testing.assert_array_equal(assigned(ids,20261001)[::2],assigned(ids[::2],20261001))


def main():
    verify()
    settings=load_settings()
    raw = pd.read_parquet(ROOT/'data/consumption.parquet')
    rows=[];coverage=[];calibration=[];membership=[]
    for category in CATEGORIES:
        panel = raw[raw.category == category].pivot(index='territory_id',columns='date',values='value').sort_index()
        asof = select_cohort(panel);balanced = panel.dropna()
        for cohort,table in [('asof_june',asof),('balanced_future_reference',balanced)]:
            values = table.to_numpy(float);ids = table.index.to_numpy(int)
            z = centered_signal(values);observed = np.isfinite(z[:,6:]);common = np.isin(ids,balanced.index)
            multiplier = noise_multiplier(z,settings)
            for i in ids:
                membership.append(dict(category=category,cohort=cohort,territory_id=int(i)))
            for month in range(6):
                coverage.append(dict(category=category,cohort=cohort,target=f'2024-{month+7:02d}',
                    eligible_ids=len(ids),observed_ids=int(observed[:,month].sum()),missing_ids=int((~observed[:,month]).sum())))
            for scaling,scale in [('raw',np.ones(len(ids))),('regularized_noise',multiplier)]:
                frozen={}
                for method in METHODS:
                    original_score = causal_scores(z*scale[:,None],method,settings)
                    threshold = float(np.quantile(original_score[:,:6].max(axis=1),settings["threshold_quantile"]))
                    frozen[method] = threshold,original_score[:,6:]>threshold
                    calibration.append(dict(category=category,cohort=cohort,scaling=scaling,method=method,
                        threshold=threshold,calibration_alarm_rate=float((original_score[:,:6]>threshold).any(axis=1).mean()),
                        scale_min=float(scale.min()),scale_max=float(scale.max())))
                for seed in SEEDS:
                    treated = assigned(ids,seed)
                    for shift in [-20,20]:
                        for shape in ['step','pulse','ramp']:
                            changed = values.copy();changed[treated,18:] *= factors(shape,shift/100)
                            changed_z = centered_signal(changed)*scale[:,None]
                            onset = 1 if shape == 'ramp' else 0
                            for method,(threshold,original) in frozen.items():
                                altered_alarm = causal_scores(changed_z,method,settings)[:,6:]>threshold
                                for scope,mask in [('all_eligible',np.ones(len(ids),bool)),('common_balanced_ids',common)]:
                                    rows.append(dict(category=category,cohort=cohort,scaling=scaling,seed=seed,
                                        shift_pct=shift,shape=shape,method=method,scope=scope,
                                        **event_metrics(original,altered_alarm,observed,treated,mask,onset)))
        print(f'Detector cohort/noise: {category} complete',flush=True)
    out=ROOT/'results'
    frame=pd.DataFrame(rows);frame.to_csv(out/'asof_detector_runs.csv',index=False)
    keys=['category','cohort','scaling','shift_pct','shape','method','scope']
    metric_columns=[c for c in frame.columns if c not in keys+['seed']]
    summary=frame.groupby(keys)[metric_columns].mean().reset_index()
    summary.to_csv(out/'asof_detector_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'asof_detector_coverage.csv',index=False)
    pd.DataFrame(calibration).to_csv(out/'asof_detector_calibration.csv',index=False)
    pd.DataFrame(membership).to_csv(out/'asof_detector_membership.csv',index=False)
    assert load_settings()==settings, 'Settings changed during audit'
    protocol=dict(input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},calibration='Jan-Jun2024, all18observed training months',evaluation='Jul-Dec2024',
        cohort_rule='Observed history Jan2023-Jun2024 only; balanced24month cohort is explicitly retrospective reference',
        missing_state='Score NaN, no scored alarm; EWMA/CUSUM reset at missing; rolling contiguous observations only',
        noise_rule='Configured individual/pooled calibration MAD weight and floor; pooled/regularized MAD',
        detector_settings=settings,threshold_quantile=settings["threshold_quantile"],seeds=list(SEEDS),assignment='ID-stable SHA256(seed:ID) < 0.25',
        scenarios=len(frame),selection='No method/scaling selected on evaluated shocks',
        denominators='First1/2/3active-month rates use fully observed windows; missing windows shown separately; load per observed MO-month',
        limits='Repeated2024, synthetic assignments not independent histories, no real false-alarm labels, MAD calibrated on only6months.')
    (out/'asof_detector_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))


def verify_artifacts():
    verify()
    out=ROOT/'results'
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    membership=pd.read_csv(out/'asof_detector_membership.csv')
    coverage=pd.read_csv(out/'asof_detector_coverage.csv')
    for category in CATEGORIES:
        panel=raw[raw.category == category].pivot(index='territory_id',columns='date',values='value').sort_index()
        selected=select_cohort(panel)
        future_removed=panel.copy();future_removed.iloc[:,18:]=np.nan
        assert select_cohort(future_removed).index.equals(selected.index)
        for cohort,table in [('asof_june',selected),('balanced_future_reference',panel.dropna())]:
            ids=membership[(membership.category == category)&(membership.cohort == cohort)].territory_id
            assert ids.tolist() == table.index.tolist()
            for row in coverage[(coverage.category == category)&(coverage.cohort == cohort)].itertuples():
                assert row.eligible_ids == len(table)
                assert row.observed_ids == table[row.target].notna().sum()
                assert row.missing_ids == row.eligible_ids-row.observed_ids
    runs=pd.read_csv(out/'asof_detector_runs.csv')
    protocol=json.loads((out/'asof_detector_protocol.json').read_text())
    assert protocol['detector_settings']==load_settings()
    assert set(protocol['input_sha256'])==set(SOURCES)
    for p,h in protocol['input_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert len(runs) == protocol['scenarios'] == 11520
    keys=['category','cohort','scaling','shift_pct','shape','method','scope']
    columns=[c for c in runs.columns if c not in keys+['seed']]
    expected=runs.groupby(keys)[columns].mean().reset_index()
    pd.testing.assert_frame_equal(expected,pd.read_csv(out/'asof_detector_summary.csv'),check_dtype=False,rtol=1e-10)
    for row in pd.read_csv(out/'asof_detector_calibration.csv').itertuples():
        assert 0 <= row.calibration_alarm_rate <= 1-load_settings()['threshold_quantile']+1/len(membership[(membership.category==row.category)&(membership.cohort==row.cohort)])+1e-12
        assert row.scale_min > 0 and row.scale_max <= 1/(1-load_settings()['noise_individual_weight'])+1e-9
    # Unobserved onset is excluded, not silently scored as a missed event.
    original=np.zeros((4,6),bool);changed=original.copy();changed[0,0]=True;changed[1,0]=True
    observed=np.ones((4,6),bool);observed[1,0]=False
    result=event_metrics(original,changed & observed,observed,np.array([1,1,0,0],bool),np.ones(4,bool),0)
    assert result['observed_onset_n']==1 and result['new_by_onset']==1
    return True


if __name__ == '__main__':
    main()
