"""Paired shock study: adaptive one-step versus frozen June seasonal forecasts.

Seasonal factors use only 2023. Jan-Jun is threshold calibration; Jul-Dec
is retrospective evaluation. Scores reset at July for equal six-month windows.
No threshold is tuned using evaluation or injected shocks.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sberindex.detection.change_detection import CATEGORIES, panel_for, score

from sberindex.paths import ROOT
METHODS = ('spike', 'rolling_3m', 'ewma', 'cusum')
PROTOCOL = {
    'models': ['adaptive_1m', 'frozen_6m'],
    'seasonality': 'pooled 2023 monthly sums, frozen before January 2024',
    'adaptive': 'previous observed month times known seasonal ratio',
    'frozen': 'December 2023 anchor for January-June; June 2024 anchor for July-December',
    'calibration': 'January-June 2024', 'evaluation': 'July-December 2024',
    'score_state': 'reset at start of each six-month window',
    'threshold_local': '99th percentile of municipal calibration maxima',
    'threshold_aggregate': 'own calibration maximum',
    'minimum_threshold': 'log(1.05), fixed exploratory floor for all methods',
    'shifts_pct': [-20, 20], 'shapes': ['permanent', 'one_month', 'ramp'],
    'scope': 'whole panel, three largest eligible regions, 25% municipalities with three fixed seeds',
    'limits': 'reused archive; full-panel eligibility is retrospective; no real change labels or false-alarm guarantees',
}


def predict(values, model):
    seasonal = values[:, :12].sum(axis=0)
    out = np.empty((len(values), 12))
    for j, target in enumerate(range(12, 24)):
        anchor = target - 1 if model == 'adaptive_1m' else (11 if target < 18 else 17)
        out[:, j] = values[:, anchor] * seasonal[target % 12] / seasonal[anchor % 12]
    return out


def channels(errors, regions, eligible):
    return {'local': errors, 'national': np.median(errors, axis=0, keepdims=True),
            'regional': np.stack([np.median(errors[regions == r], axis=0) for r in eligible])}


def main():
    lookup = pd.read_csv(ROOT / 'results/municipal_lookup.csv')
    lookup = lookup[lookup.year == 2024].set_index('territory_id')
    rows, backgrounds, trajectories, accuracy = [], [], [], []
    future_error = calibration_error = 0.0
    for category in CATEGORIES:
        ids, values = panel_for(category)
        regions = lookup.loc[ids, 'region_code'].to_numpy()
        counts = pd.Series(regions).value_counts().sort_index()
        eligible = counts[counts >= 10].index.to_numpy()
        largest = sorted(eligible, key=lambda r: (-counts[r], r))[:3]
        scenarios = [('national', 'all', np.ones(len(ids), bool))]
        scenarios += [('regional', str(r), regions == r) for r in largest]
        for seed in (20261001, 20261002, 20261003):
            selected = np.zeros(len(ids), bool)
            selected[np.random.default_rng(seed).choice(len(ids), len(ids)//4, replace=False)] = True
            scenarios.append(('local', str(seed), selected))
        for model in PROTOCOL['models']:
            prediction = predict(values, model)
            errors = np.log(values[:, 12:] / prediction)
            original = channels(errors, regions, eligible)
            accuracy.append({'category': category, 'model': model,
                             'MAE_Jul_Dec': np.abs(values[:, 18:]-prediction[:, 6:]).mean(),
                             'note': 'Different forecast horizons; not a like-for-like forecast ranking'})
            thresholds, baseline = {}, {}
            for level, signal in original.items():
                for method in METHODS:
                    maximum = score(signal[:, :6], method).max(axis=1)
                    threshold = np.quantile(maximum, .99) if level == 'local' else maximum
                    thresholds[level, method] = np.maximum(threshold, np.log(1.05))
                    th = np.atleast_1d(thresholds[level, method])[:, None]
                    alarms = score(signal[:, 6:], method) > th
                    baseline[level, method] = alarms.any(axis=1)
                    backgrounds.append({'category': category, 'model': model, 'level': level, 'method': method,
                                        'units': len(signal), 'alarm_rate': alarms.any(axis=1).mean(),
                                        'mean_alarm_months': alarms.sum(axis=1).mean()})
            future = values.copy(); future[:len(values)//2, 19:] *= 1.5
            future_error = max(future_error, float(np.abs(predict(future, model)[:, :7]-prediction[:, :7]).max()))
            for scope, scenario, treated in scenarios:
                targets = {'local': treated, 'national': np.array([scope == 'national']),
                           'regional': np.array([treated[regions == r].all() for r in eligible])}
                for shift in PROTOCOL['shifts_pct']:
                    for shape in PROTOCOL['shapes']:
                        factor = np.full(6, 1+shift/100)
                        if shape == 'one_month': factor[1:] = 1
                        if shape == 'ramp': factor = 1 + np.linspace(0, shift/100, 6)
                        altered = values.copy(); altered[treated, 18:] *= factor
                        changed_prediction = predict(altered, model)
                        changed_errors = np.log(altered[:, 12:] / changed_prediction)
                        calibration_error = max(calibration_error, float(np.abs(changed_errors[:, :6]-errors[:, :6]).max()))
                        changed = channels(changed_errors, regions, eligible)
                        if scope == 'national':
                            for month in range(6):
                                trajectories.append({'category': category, 'model': model, 'shift_pct': shift,
                                    'shape': shape, 'month_after': month,
                                    'induced_median_log_error': np.median(changed_errors[:, 6+month]-errors[:, 6+month])})
                        # Keep the level matched to the injection's unit: no regional alarm broadcast.
                        signal = changed[scope]
                        target = targets[scope]
                        for method in METHODS:
                            th = np.atleast_1d(thresholds[scope, method])[:, None]
                            alarms = score(signal[:, 6:], method) > th
                            detected = alarms.any(axis=1); base = baseline[scope, method]
                            delay = alarms[target & detected].argmax(axis=1)
                            rows.append({'category': category, 'model': model, 'scope': scope, 'scenario': scenario,
                                'shift_pct': shift, 'shape': shape, 'method': method,
                                'target_units': int(target.sum()), 'control_units': int((~target).sum()),
                                'target_alarm_rate': detected[target].mean(),
                                'target_baseline_alarm_rate': base[target].mean(),
                                'new_alarm_rate': (detected & ~base)[target].mean(),
                                'control_alarm_rate': detected[~target].mean() if (~target).any() else np.nan,
                                'control_baseline_alarm_rate': base[~target].mean() if (~target).any() else np.nan,
                                'median_delay_detected': float(np.median(delay)) if len(delay) else np.nan})
    assert future_error < 1e-12 and calibration_error < 1e-12
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'results/forecast_residual_stress.csv', index=False)
    summary = frame.groupby(['category','model','scope','shift_pct','shape','method'])[
        ['target_alarm_rate','target_baseline_alarm_rate','new_alarm_rate','control_alarm_rate',
         'control_baseline_alarm_rate','median_delay_detected']].mean()
    summary.to_csv(ROOT / 'results/forecast_residual_summary.csv')
    pd.DataFrame(backgrounds).to_csv(ROOT / 'results/forecast_residual_background.csv', index=False)
    pd.DataFrame(trajectories).to_csv(ROOT / 'results/forecast_residual_trajectories.csv', index=False)
    pd.DataFrame(accuracy).to_csv(ROOT / 'results/forecast_residual_accuracy.csv', index=False)
    audit = {'protocol': PROTOCOL, 'rows': len(frame), 'future_prediction_max_error': future_error,
             'calibration_error_after_injection': calibration_error}
    (ROOT / 'results/forecast_residual_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(summary.loc[('Все категории',slice(None),slice(None),20,'permanent','rolling_3m'),:].to_string())
    print(pd.DataFrame(backgrounds).query("category == 'Все категории'").to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
