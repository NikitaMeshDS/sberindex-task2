"""Sequential trigger/confirmation experiment; parameters fixed before first run.

Trigger on a one-step seasonal error; freeze the pre-trigger deseasonalized
level and require three consecutive observations with same-signed deviations.
November/December triggers cannot be fully classified in this archive.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sberindex.detection.change_detection import CATEGORIES, panel_for

from sberindex.paths import ROOT
PROTOCOL = {
    'seasonality': '2023 pooled sums, frozen', 'calibration': 'January-June 2024',
    'evaluation': 'July-December 2024', 'trigger': 'absolute one-step log error exceeds threshold',
    'threshold': 'municipal calibration-max q99; aggregate own maximum; minimum log(1.05)',
    'confirmation': 'three consecutive same-direction deviations above the trigger threshold against pre-trigger level',
    'state': 'one active candidate per series, then restart next month; no retrospective backdating',
    'censoring': 'unresolved end-of-archive candidates reported separately',
    'shifts_pct': [-20,20], 'shapes': ['permanent','one_month','ramp'],
    'status': 'exploratory sequential confirmation; reused archive; not independent validation',
}


def transformed(values):
    seasonal = values[:, :12].sum(axis=0)
    return np.log(values[:, 11:]) - np.log(seasonal[np.arange(11,24)%12])[None,:]


def levels(x, regions, eligible):
    return {'local': x, 'national': np.median(x,axis=0,keepdims=True),
            'regional': np.stack([np.median(x[regions==r],axis=0) for r in eligible])}


def monitor(x, threshold):
    """x includes preceding June at col0. Outputs event timestamps, never onset guesses."""
    n, length = x.shape
    threshold = np.broadcast_to(np.asarray(threshold), (n,))
    trigger = np.zeros((n,length-1),bool); confirm = trigger.copy(); reject = trigger.copy()
    signed_trigger = np.zeros_like(trigger,dtype=float); signed_confirm = signed_trigger.copy()
    active = np.zeros(n,bool); count = np.zeros(n,int)
    anchor = np.zeros(n); direction = np.zeros(n)
    for t in range(1,length):
        # Every candidate is judged only on values already observed at t.
        was_active = active.copy()
        error = x[:,t]-x[:,t-1]
        start = ~was_active & (np.abs(error)>threshold)
        trigger[start,t-1] = True
        signed_trigger[start,t-1] = np.sign(error[start])
        active[start] = True; count[start] = 0
        anchor[start] = x[start,t-1]; direction[start] = np.sign(error[start])
        valid = active & ((x[:,t]-anchor)*direction > threshold)
        failed = active & ~valid
        reject[failed,t-1] = True; active[failed] = False; count[failed] = 0
        count[valid] += 1
        done = active & (count>=3)
        signed_confirm[done,t-1] = direction[done]
        confirm[done,t-1] = True; active[done] = False; count[done] = 0
    return trigger,confirm,reject,active,signed_trigger,signed_confirm


def main():
    lookup = pd.read_csv(ROOT/'results/municipal_lookup.csv')
    lookup = lookup[lookup.year==2024].set_index('territory_id')
    records,background,event_rows = [],[],[]
    prefix_error = 0
    for category in CATEGORIES:
        ids,values = panel_for(category)
        regions = lookup.loc[ids,'region_code'].to_numpy()
        counts = pd.Series(regions).value_counts().sort_index()
        eligible = counts[counts>=10].index.to_numpy()
        largest = sorted(eligible,key=lambda r:(-counts[r],r))[:3]
        originals = levels(transformed(values),regions,eligible)
        unit_ids = {'local':ids,'regional':eligible,'national':['panel']}
        thresholds,base = {},{}
        for level,x in originals.items():
            maxima = np.abs(np.diff(x[:,:7],axis=1)).max(axis=1)
            thresholds[level] = np.maximum(np.quantile(maxima,.99) if level=='local' else maxima,np.log(1.05))
            base[level] = monitor(x[:,6:],thresholds[level])
            trigger,confirm,reject,pending,trigger_sign,confirm_sign = base[level]
            background.append({'category':category,'level':level,'units':len(x),
                'trigger_rate':trigger.any(axis=1).mean(),'confirmation_rate':confirm.any(axis=1).mean(),
                'rejection_rate':reject.any(axis=1).mean(),'pending_rate':pending.mean()})
            for i,unit in enumerate(unit_ids[level]):
                for kind,events in [('trigger',trigger),('confirmation',confirm),('rejection',reject)]:
                    for t in np.flatnonzero(events[i]):
                        event_rows.append({'category':category,'level':level,'unit':str(unit),'event':kind,
                                           'date':str(pd.Period('2024-07',freq='M')+int(t))})
            future = x[:,6:].copy(); future[:len(x)//2,3:] += .4
            future_result = monitor(future,thresholds[level])
            for original,changed in zip(base[level][:3],future_result[:3]):
                prefix_error += int((original[:,:2]!=changed[:,:2]).sum())
        scenarios = [('national','all',np.ones(len(ids),bool))]
        scenarios += [('regional',str(r),regions==r) for r in largest]
        for seed in (20261001,20261002,20261003):
            selected = np.zeros(len(ids),bool)
            selected[np.random.default_rng(seed).choice(len(ids),len(ids)//4,replace=False)] = True
            scenarios.append(('local',str(seed),selected))
        for scope,scenario,treated in scenarios:
            target = treated if scope=='local' else (np.array([True]) if scope=='national' else eligible==int(scenario))
            for shift in PROTOCOL['shifts_pct']:
                for shape in PROTOCOL['shapes']:
                    factor = np.full(6,1+shift/100)
                    if shape=='one_month': factor[1:]=1
                    if shape=='ramp': factor = 1+np.linspace(0,shift/100,6)
                    altered = values.copy(); altered[treated,18:] *= factor
                    changed = levels(transformed(altered),regions,eligible)[scope]
                    np.testing.assert_allclose(changed[:,:7],originals[scope][:,:7],rtol=0,atol=0)
                    trigger,confirm,reject,pending,trigger_sign,confirm_sign = monitor(changed[:,6:],thresholds[scope])
                    for method,events,bg,signs in [('trigger_only',trigger,base[scope][0],trigger_sign),('confirmed_3m',confirm,base[scope][1],confirm_sign)]:
                        detected = events.any(axis=1); baseline = bg.any(axis=1)
                        delay = events[target & detected].argmax(axis=1)
                        records.append({'category':category,'scope':scope,'scenario':scenario,'shift_pct':shift,'shape':shape,
                            'method':method,'target_units':int(target.sum()),'target_rate':detected[target].mean(),
                            'baseline_target_rate':baseline[target].mean(),
                            'same_direction_target_rate':(signs[target]==np.sign(shift)).any(axis=1).mean(),
                            'opposite_direction_target_rate':(signs[target]==-np.sign(shift)).any(axis=1).mean(),'new_target_rate':(detected & ~baseline)[target].mean(),
                            'control_rate':detected[~target].mean() if (~target).any() else np.nan,
                            'baseline_control_rate':baseline[~target].mean() if (~target).any() else np.nan,
                            'pending_target_rate':pending[target].mean(),
                            'median_delay_months':float(np.median(delay)) if len(delay) else np.nan})
    assert prefix_error==0
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT/'results/confirmation_stress.csv',index=False)
    summary=frame.groupby(['category','scope','shift_pct','shape','method'])[
        ['target_rate','baseline_target_rate','same_direction_target_rate','opposite_direction_target_rate','new_target_rate','control_rate','baseline_control_rate','pending_target_rate','median_delay_months']].mean()
    summary.to_csv(ROOT/'results/confirmation_summary.csv')
    pd.DataFrame(background).to_csv(ROOT/'results/confirmation_background.csv',index=False)
    pd.DataFrame(event_rows).to_csv(ROOT/'results/confirmation_events.csv',index=False)
    (ROOT/'results/confirmation_audit.json').write_text(json.dumps({'protocol':PROTOCOL,'rows':len(frame),
        'future_prefix_mismatches':prefix_error},ensure_ascii=False,indent=2))
    print(summary.loc[('Все категории',slice(None),20,slice(None),slice(None)),:].to_string())
    print(pd.DataFrame(background).query("category == 'Все категории'").to_string(index=False))


if __name__=='__main__': main()
