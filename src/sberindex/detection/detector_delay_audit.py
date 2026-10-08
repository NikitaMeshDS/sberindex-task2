"""Calendar availability of fixed synthetic detector alarms under uniform lag."""
import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.detection.detector_settings import load_settings, validate_settings, SOURCES as SETTINGS_SOURCES
from sberindex.detection.asof_detector_audit import (
    METHODS, SEEDS, select_cohort, centered_signal, noise_multiplier,
    causal_scores, assigned,
)
from sberindex.detection.change_detection import CATEGORIES
from sberindex.detection.event_attribution_audit import factors

LAGS=(0,1,2)
SOURCES=['data/consumption.parquet','docs/protocols/DETECTOR_DELAY_EXPERIMENT.md',
    'src/sberindex/detection/detector_delay_audit.py',
    'src/sberindex/detection/asof_detector_audit.py',
    'src/sberindex/detection/change_detection.py',
    'src/sberindex/detection/event_attribution_audit.py',*SETTINGS_SOURCES]
KEYS=['category','scaling','shift_pct','shape','method','reporting_delay_months']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def release_month(source,delay):
    assert delay in LAGS
    return str(pd.Period(source,freq='M')+delay)


def calendar_metrics(new,original,observed,treated,onset,delay):
    """New is already paired in source time; no reattribution after shifting."""
    assert delay in LAGS and onset in (0,1)
    assert new.shape==original.shape==observed.shape and new.shape[1]==6
    assert len(treated)==len(new) and treated.any() and (~treated).any()
    assert new.dtype==original.dtype==observed.dtype==treated.dtype==bool
    evaluated=treated&observed[:,onset:onset+3].all(axis=1)
    result={'treated_n':int(treated.sum()),'evaluated_first_three_n':int(evaluated.sum()),
        'incomplete_first_three_n':int((treated&~evaluated).sum())}
    source=np.arange(6);arrival=source+delay
    new=new&observed
    for length,label in [(1,'first'),(2,'second'),(3,'third')]:
        window=(source>=onset)&(arrival<=onset+length-1)
        result[f'new_by_{label}_calendar_month']=float(new[evaluated][:,window].any(axis=1).mean()) if evaluated.any() else np.nan
    delivered=(source>=onset)&(arrival<=5)
    alarms=new[evaluated][:,delivered]
    detected=alarms.any(axis=1)
    if detected.any():
        first=arrival[delivered][alarms[detected].argmax(axis=1)]-onset
        median=float(np.median(first))
    else:median=np.nan
    result.update(new_detected_by_december_n=int(detected.sum()),
        new_by_december=float(detected.mean()) if evaluated.any() else np.nan,
        median_calendar_delay_if_new=median,
        missing_treated_source_months=int((~observed[treated,onset:]).sum()),
        pending_observed_treated_months=int(observed[treated][:,(source>=onset)&(arrival>5)].sum()))
    control=~treated
    denominator=int(observed[control][:,delivered].sum())
    result['delivered_control_mo_months']=denominator
    result['new_control_per100_delivered_months']=float(100*new[control][:,delivered].sum()/denominator) if denominator else np.nan
    result['original_control_per100_delivered_months']=float(100*(original&observed)[control][:,delivered].sum()/denominator) if denominator else np.nan
    return result


def build(settings=None):
    settings=load_settings() if settings is None else validate_settings(settings)
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    rows=[];calibration=[];coverage=[]
    for category in CATEGORIES:
        panel=raw[raw.category.eq(category)].pivot(index='territory_id',columns='date',values='value').sort_index()
        table=select_cohort(panel);values=table.to_numpy(float);ids=table.index.to_numpy(int)
        z=centered_signal(values);observed=np.isfinite(z[:,6:])
        for delay in LAGS:
            for t in range(6):
                source=f'2024-{t+7:02d}';release=release_month(source,delay);within=release<='2024-12'
                coverage.append(dict(category=category,reporting_delay_months=delay,source_month=source,
                    release_month=release,calibration_ready_month=release_month('2024-06',delay),
                    eligible_ids=len(table),observed_source_ids=int(observed[:,t].sum()),
                    missing_source_ids=int((~observed[:,t]).sum()),
                    delivered_by_december_ids=int(observed[:,t].sum()) if within else 0,
                    pending_observed_ids=0 if within else int(observed[:,t].sum())))
        for scaling,scale in [('raw',np.ones(len(table))),('regularized_noise',noise_multiplier(z,settings))]:
            frozen={}
            for method in METHODS:
                scores=causal_scores(z*scale[:,None],method,settings)
                threshold=float(np.quantile(scores[:,:6].max(axis=1),settings["threshold_quantile"]))
                frozen[method]=(threshold,scores[:,6:]>threshold)
                calibration.append(dict(category=category,scaling=scaling,method=method,threshold=threshold,
                    calibration_alarm_rate=float((scores[:,:6]>threshold).any(axis=1).mean()),eligible_ids=len(table)))
            for seed in SEEDS:
                treated=assigned(ids,seed)
                for shift in [-20,20]:
                    for shape in ['step','pulse','ramp']:
                        changed=values.copy();changed[treated,18:]*=factors(shape,shift/100)
                        shifted=centered_signal(changed)*scale[:,None];onset=1 if shape=='ramp' else 0
                        np.testing.assert_allclose(shifted[:,:6],z[:,:6]*scale[:,None],rtol=0,atol=0)
                        for method,(threshold,original) in frozen.items():
                            changed_alarm=causal_scores(shifted,method,settings)[:,6:]>threshold
                            new=changed_alarm&~original&observed
                            for delay in LAGS:
                                rows.append(dict(category=category,scaling=scaling,seed=seed,shift_pct=shift,
                                    shape=shape,method=method,reporting_delay_months=delay,
                                    onset_month=f'2024-{7+onset:02d}',calibration_ready_month=release_month('2024-06',delay),
                                    **calendar_metrics(new,original,observed,treated,onset,delay)))
        print(f'Detector calendar delay: {category}',flush=True)
    runs=pd.DataFrame(rows)
    metrics=[c for c in runs if c not in KEYS+['seed','onset_month','calibration_ready_month']]
    summary=runs.groupby(KEYS,sort=True)[metrics].mean().reset_index()
    return {'runs':runs,'summary':summary,'coverage':pd.DataFrame(coverage),'calibration':pd.DataFrame(calibration)}


def main():
    settings=load_settings();tables=build(settings)
    assert load_settings()==settings, 'Settings changed during audit'
    for name,table in tables.items():table.to_csv(ROOT/'results'/f'detector_delay_{name}.csv',index=False)
    protocol={'input_sha256':{p:sha(ROOT/p) for p in SOURCES},'rows':len(tables['runs']),
        'detector_settings':settings,'lags':list(LAGS),'decision_end':'2024-12','calibration':'Jan-Jun 2024; ready June+lag; no recalibration by lag',
        'denominator':'Assigned MO with all first three source months observed; fixed across methods and lags',
        'summary':'Equal weight over ten ID-stable assignments; conditional medians averaged across assignments',
        'selection':'No new selection of detector, noise scale, threshold or lag',
        'limits':'Reused2024, synthetic, uniform hypothetical lag; no real vintages, revisions, event truth or independent histories.'}
    (ROOT/'results/detector_delay_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))


def verify():
    out=ROOT/'results';protocol=json.loads((out/'detector_delay_protocol.json').read_text())
    assert set(protocol['input_sha256'])==set(SOURCES)
    for p,h in protocol['input_sha256'].items():assert sha(ROOT/p)==h,p
    assert protocol['detector_settings']==load_settings()
    expected=build()
    for name,table in expected.items():
        pd.testing.assert_frame_equal(pd.read_csv(out/f'detector_delay_{name}.csv'),table,check_dtype=False,rtol=1e-10,atol=1e-12)
    runs=expected['runs'];assert len(runs)==protocol['rows']==8640
    assert not runs.duplicated(KEYS+['seed']).any()
    grouping=[c for c in KEYS if c!='reporting_delay_months']+['seed']
    for _,group in runs.groupby(grouping):
        group=group.sort_values('reporting_delay_months')
        assert group.evaluated_first_three_n.nunique()==group.treated_n.nunique()==1
        for column in ['new_by_first_calendar_month','new_by_second_calendar_month','new_by_third_calendar_month','new_by_december']:
            assert (np.diff(group[column])<=1e-12).all(),column
        np.testing.assert_allclose(group[group.reporting_delay_months.eq(1)].new_by_third_calendar_month,
            group[group.reporting_delay_months.eq(0)].new_by_second_calendar_month,rtol=0,atol=0)
        np.testing.assert_allclose(group[group.reporting_delay_months.eq(2)].new_by_third_calendar_month,
            group[group.reporting_delay_months.eq(0)].new_by_first_calendar_month,rtol=0,atol=0)
    old=pd.read_csv(out/'asof_detector_calibration.csv')
    old=old[old.cohort.eq('asof_june')].sort_values(['category','scaling','method']).reset_index(drop=True)
    current=expected['calibration'].sort_values(['category','scaling','method']).reset_index(drop=True)
    pd.testing.assert_frame_equal(current[['category','scaling','method']],old[['category','scaling','method']])
    np.testing.assert_allclose(current.threshold,old.threshold,rtol=1e-12,atol=1e-12)
    old_runs=pd.read_csv(out/'asof_detector_runs.csv')
    old_runs=old_runs[old_runs.cohort.eq('asof_june')&old_runs.scope.eq('all_eligible')].sort_values(grouping).reset_index(drop=True)
    zero=runs[runs.reporting_delay_months.eq(0)].sort_values(grouping).reset_index(drop=True)
    pd.testing.assert_frame_equal(zero[grouping],old_runs[grouping])
    np.testing.assert_array_equal(zero.evaluated_first_three_n,old_runs.observed_third_n)
    np.testing.assert_allclose(zero.new_by_third_calendar_month,old_runs.new_by_third,rtol=1e-12,atol=1e-12)
    return True


if __name__=='__main__':
    import sys
    if '--verify' in sys.argv:print(verify())
    else:main()
