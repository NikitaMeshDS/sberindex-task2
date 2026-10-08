"""Exploratory three-level monitoring with separated fit/calibration periods.

All aggregates are unweighted medians of complete municipalities, not national
expenditure volumes. National/regional alarms are never broadcast as local ones.
The protocol is fixed in code before the first run; no post-June optimization.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sberindex.detection.change_detection import CATEGORIES, panel_for, score

from sberindex.paths import ROOT
METHODS = ('spike', 'rolling_3m', 'ewma', 'cusum')
PROTOCOL = {
    'fit': '2024-01 through 2024-03', 'calibrate': '2024-04 through 2024-06',
    'evaluate': '2024-07 through 2024-12', 'min_region_municipalities': 10,
    'threshold': 'per-unit max calibration score, floored at log(1.05)',
    'floor_note': 'Fixed exploratory minimum effect, not optimized or a significance level',
    'shifts_pct': [-20, 20], 'shapes': ['permanent', 'ramp'],
    'regional_scenarios': 'three largest eligible regions by complete-panel municipality count',
    'local_scenarios': '25% of eligible municipalities, three fixed seeds',
    'status': 'exploratory reused archive, no independent temporal holdout',
}


def signals(values, regions, eligible):
    growth = np.log(values[:, 12:] / values[:, :12])
    national = np.median(growth, axis=0, keepdims=True)
    regional = np.stack([np.median(growth[regions == r], axis=0) for r in eligible])
    local = growth.copy()
    for r, aggregate in zip(eligible, regional):
        local[regions == r] -= aggregate
    valid = np.isin(regions, eligible)
    return {'national': national, 'regional': regional, 'local': local[valid]}


def fit(signals_by_level):
    fitted = {}
    for level, raw in signals_by_level.items():
        center = np.median(raw[:, :3], axis=1, keepdims=True)
        for method in METHODS:
            sc = score(raw - center, method)
            threshold = np.maximum(sc[:, 3:6].max(axis=1), np.log(1.05))
            fitted[level, method] = (center, threshold)
    return fitted


def evaluate(raw, center, threshold, method):
    return score(raw - center, method)[:, 6:] > threshold[:, None]


def main():
    lookup = pd.read_csv(ROOT / 'results/municipal_lookup.csv')
    lookup = lookup[lookup.year == 2024].set_index('territory_id')
    rows, background, geography = [], [], []
    max_future_error = max_common_error = 0.0
    for category in CATEGORIES:
        ids, values = panel_for(category)
        regions = lookup.loc[ids, 'region_code'].to_numpy()
        counts = pd.Series(regions).value_counts().sort_index()
        eligible = counts[counts >= PROTOCOL['min_region_municipalities']].index.to_numpy()
        valid = np.isin(regions, eligible)
        largest = sorted(eligible, key=lambda r: (-counts[r], r))[:3]
        units = {'national': ['panel'], 'regional': eligible, 'local': ids[valid]}
        for r in counts.index:
            geography.append({'category': category, 'region_code': r,
                              'municipalities': int(counts[r]), 'eligible': bool(r in eligible),
                              'regional_stress_target': bool(r in largest)})
        raw = signals(values, regions, eligible)
        fitted = fit(raw)
        baseline = {}
        for (level, method), (center, threshold) in fitted.items():
            baseline[level, method] = evaluate(raw[level], center, threshold, method)
            for i, unit in enumerate(units[level]):
                alarms = baseline[level, method][i]
                background.append({'category': category, 'level': level, 'method': method,
                                   'unit': str(unit), 'threshold': threshold[i],
                                   'any_alarm': bool(alarms.any()), 'alarm_months': int(alarms.sum()),
                                   'first_alarm': str(pd.Period('2024-07', freq='M') + int(alarms.argmax())) if alarms.any() else ''})
        # Prefix invariance and fit invariance under post-July perturbations.
        future = values.copy(); future[:len(values)//2, 19:] *= 1.5
        fraw = signals(future, regions, eligible)
        ffitted = fit(fraw)
        for key, (center, threshold) in fitted.items():
            level, method = key
            np.testing.assert_allclose(center, ffitted[key][0], rtol=0, atol=0)
            np.testing.assert_allclose(threshold, ffitted[key][1], rtol=0, atol=0)
            err = np.abs(score(raw[level]-center, method)[:, 6] - score(fraw[level]-center, method)[:, 6]).max()
            max_future_error = max(max_future_error, float(err))
        scenarios = [('national', 'all', np.ones(len(ids), bool))]
        scenarios += [('regional', str(r), regions == r) for r in largest]
        for seed in (20261001, 20261002, 20261003):
            treated = np.zeros(len(ids), bool)
            candidates = np.flatnonzero(valid)
            treated[np.random.default_rng(seed).choice(candidates, len(candidates)//4, replace=False)] = True
            scenarios.append(('local', str(seed), treated))
        for scope, scenario, treated in scenarios:
            for shift in PROTOCOL['shifts_pct']:
                for shape in PROTOCOL['shapes']:
                    factor = np.full(6, 1+shift/100) if shape == 'permanent' else 1 + np.linspace(0, shift/100, 6)
                    altered = values.copy(); altered[treated, 18:] *= factor
                    changed = signals(altered, regions, eligible)
                    if scope == 'national':
                        err = np.abs((changed['national']-raw['national'])[0, 6:] - np.log(factor)).max()
                        max_common_error = max(max_common_error, float(err))
                    # Score every channel; compare rates only at its own unit level.
                    targets = {'national': np.array([scope == 'national']),
                               'regional': np.array([treated[regions == r].all() for r in eligible]),
                               'local': treated[valid]}
                    for (level, method), (center, threshold) in fitted.items():
                        alarms = evaluate(changed[level], center, threshold, method)
                        detected = alarms.any(axis=1)
                        base = baseline[level, method].any(axis=1)
                        target = targets[level]
                        control = ~target
                        delay = alarms[target & detected].argmax(axis=1)
                        rows.append({'category': category, 'scope': scope, 'scenario': scenario,
                                     'shift_pct': shift, 'shape': shape, 'level': level, 'method': method,
                                     'intended_channel': scope == level, 'target_units': int(target.sum()),
                                     'control_units': int(control.sum()),
                                     'target_alarm_rate': detected[target].mean() if target.any() else np.nan,
                                     'target_baseline_alarm_rate': base[target].mean() if target.any() else np.nan,
                                     'target_new_alarm_rate': (detected & ~base)[target].mean() if target.any() else np.nan,
                                     'control_alarm_rate': detected[control].mean() if control.any() else np.nan,
                                     'control_baseline_alarm_rate': base[control].mean() if control.any() else np.nan,
                                     'median_delay_detected': float(np.median(delay)) if len(delay) else np.nan})
    assert max_future_error < 1e-12 and max_common_error < 1e-12
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'results/hierarchical_stress.csv', index=False)
    bg = pd.DataFrame(background)
    bg.to_csv(ROOT / 'results/hierarchical_background.csv', index=False)
    bg.groupby(['category', 'level', 'method']).agg(units=('unit','size'), alarm_rate=('any_alarm','mean'),
        mean_alarm_months=('alarm_months','mean')).to_csv(ROOT / 'results/hierarchical_background_summary.csv')
    pd.DataFrame(geography).to_csv(ROOT / 'results/hierarchical_coverage.csv', index=False)
    metrics = ['target_alarm_rate', 'target_baseline_alarm_rate', 'target_new_alarm_rate',
               'control_alarm_rate', 'control_baseline_alarm_rate', 'median_delay_detected']
    summary = results[results.intended_channel].groupby(['category','scope','shift_pct','shape','method'])[metrics].mean()
    summary.to_csv(ROOT / 'results/hierarchical_summary.csv')
    audit = {'protocol': PROTOCOL, 'scenario_channel_rows': len(results),
             'future_perturbation_max_error': max_future_error, 'common_shift_preservation_max_error': max_common_error,
             'limits': 'Alarm rates on unlabelled real data are not false positive rates; three calibration months cannot establish reliability.'}
    (ROOT / 'results/hierarchical_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(summary.loc[('Все категории', slice(None), 20, 'permanent', slice(None)), :].to_string())
    print(bg[bg.category == 'Все категории'].groupby(['level','method']).any_alarm.agg(['mean','size']).to_string())


if __name__ == '__main__':
    main()
