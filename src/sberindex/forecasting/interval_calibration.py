"""Past-only normalized intervals and a panel feedback heuristic.

Separate research appendix; no theoretical coverage guarantee, model selection,
or changes to point forecasts. Parameters in configs/interval_calibration.json.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.forecasting.operational_audit import finite_quantile
from sberindex.paths import ROOT


def calibrate_panel(frame, scale_2023, config):
    p = frame.copy().sort_values(['model', 'target', 'territory_id'])
    keys = ['model', 'target', 'territory_id']
    if p.duplicated(keys).any():
        raise ValueError('duplicate forecast key')
    p['scale_2023'] = p.territory_id.map(scale_2023)
    values = p[['actual', 'predicted', 'scale_2023']].to_numpy(float)
    if not np.isfinite(values).all() or (p.scale_2023 <= 0).any() or (p.predicted < 0).any() or (p.actual < 0).any():
        raise ValueError('missing/nonfinite data or invalid scale/negative expense')
    rows, audit = [], []
    for model, part in p.groupby('model', sort=True):
        for strategy in config['strategies']:
            if strategy not in ['raw_3m', 'scaled_2023_3m', 'scaled_forecast_3m', 'adaptive_scaled_2023_3m']:
                raise ValueError('unknown interval strategy')
            normalizer = (pd.Series(1., index=part.index) if strategy == 'raw_3m'
                else part.predicted.abs().clip(lower=config['normalizer_floor_rub']) if strategy == 'scaled_forecast_3m'
                else part.scale_2023.clip(lower=config['normalizer_floor_rub']))
            for level in config['levels']:
                nominal_alpha = 1 - level
                alpha = nominal_alpha
                for target in config['evaluation_targets']:
                    start = max(config['calibration_start'], str(pd.Period(target, 'M') - config['window_months']))
                    cal = part[(part.target >= start) & (part.target < target)]
                    test = part[part.target == target].copy()
                    expected_dates = set(pd.period_range(start, pd.Period(target,'M')-1, freq='M').astype(str))
                    if cal.empty or test.empty or set(cal.target) != expected_dates:
                        raise ValueError('missing calibration/evaluation month')
                    # Calendar-balanced comparisons require equal ID coverage.
                    ids = set(test.territory_id)
                    if any(set(g.territory_id) != ids for _, g in cal.groupby('target')):
                        raise ValueError('different calibration/evaluation ID coverage')
                    scores = (cal.actual - cal.predicted).abs() / normalizer.loc[cal.index]
                    q = finite_quantile(scores, 1-alpha)
                    half_width = q * normalizer.loc[test.index]
                    test['lower'] = (test.predicted-half_width).clip(lower=0)
                    test['upper'] = (test.predicted+half_width).clip(lower=0)
                    test['covered'] = (test.actual >= test.lower) & (test.actual <= test.upper)
                    test['width'] = test.upper - test.lower
                    test['relative_width_pct'] = 100 * test.width / test.scale_2023
                    test['interval_score'] = test.width + 2/nominal_alpha * (
                        (test.lower-test.actual).clip(lower=0) + (test.actual-test.upper).clip(lower=0))
                    test['strategy'], test['nominal_coverage'] = strategy, level
                    # Feedback arrives only AFTER this month's interval was issued.
                    after = (float(np.clip(alpha + config['gamma']*(nominal_alpha - (1-test.covered.mean())),
                        config['alpha_min'], config['alpha_max'])) if strategy.startswith('adaptive_') else alpha)
                    audit.append(dict(model=model, strategy=strategy, nominal_coverage=level, target=target,
                        origin=str(pd.Period(target,'M')-1), calibration_first=cal.target.min(),
                        calibration_last=cal.target.max(), calibration_dates=cal.target.nunique(),
                        calibration_rows=len(cal), q=q, alpha_before=alpha, alpha_after=after,
                        observed_miscoverage=1-test.covered.mean()))
                    rows.append(test)
                    alpha = after
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit)


def main():
    config_path = ROOT/'configs/interval_calibration.json'
    cfg = json.loads(config_path.read_text())
    source = pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    p = source[(source.horizon == cfg['horizon']) & source.model.isin(cfg['models'])
        & (source.target >= cfg['calibration_start'])].copy()
    assert (p.origin == (pd.PeriodIndex(p.target,freq='M')-1).astype(str)).all()
    source_data = pd.read_parquet(ROOT/'data/consumption.parquet')
    train = source_data[(source_data.category == 'Все категории') & (source_data.date.astype(str).str[:4] == '2023')]
    counts = train.groupby('territory_id').date.nunique()
    assert (counts.reindex(p.territory_id.unique()) == 12).all()
    scale = train.groupby('territory_id').value.mean()
    result, cal = calibrate_panel(p, scale, cfg)
    strata = pd.qcut(scale.reindex(sorted(p.territory_id.unique())), 4, labels=['Q1','Q2','Q3','Q4'])
    result['expense_quartile'] = result.territory_id.map(strata).astype(str)
    group = ['model','strategy','nominal_coverage']
    monthly = result.groupby(group+['target']).agg(coverage=('covered','mean'), mean_width_rub=('width','mean'),
        relative_width_pct=('relative_width_pct','mean'),mean_interval_score=('interval_score','mean'), observations=('covered','size')).reset_index()
    summary = monthly.groupby(group).agg(coverage=('coverage','mean'), minimum_date_coverage=('coverage','min'),
        mean_width_rub=('mean_width_rub','mean'),relative_width_pct=('relative_width_pct','mean'),
        mean_interval_score=('mean_interval_score','mean'), dates=('target','nunique'),observations=('observations','sum')).reset_index()
    old = pd.read_csv(ROOT/'results/online_interval_summary.csv')
    baseline = summary[summary.strategy == 'raw_3m'].set_index(['model','nominal_coverage']).sort_index()
    old = old[(old.strategy == 'rolling_3m') & old.model.isin(cfg['models'])].set_index(['model','nominal_coverage']).sort_index()
    assert baseline.index.equals(old.index)
    assert np.allclose(baseline[['coverage','mean_width_rub','mean_interval_score']],
        old[['coverage','mean_width_rub','mean_interval_score']], rtol=0,atol=1e-8)
    for _, g in result.groupby(group):
        expected=p[(p.model==g.model.iloc[0]) & p.target.isin(cfg['evaluation_targets'])].set_index(['territory_id','target'])
        actual=g.set_index(['territory_id','target'])
        assert set(expected.index)==set(actual.index)
        assert np.array_equal(expected.sort_index()[['actual','predicted']],actual.sort_index()[['actual','predicted']])
    out=ROOT/'reports/interval_calibration';out.mkdir(parents=True,exist_ok=True)
    result.to_parquet(out/'predictions.parquet',index=False)
    cal.to_csv(out/'calibration.csv',index=False)
    monthly.to_csv(out/'monthly.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    result.groupby(group+['expense_quartile'],observed=True).agg(coverage=('covered','mean'),
        mean_width_rub=('width','mean'),relative_width_pct=('relative_width_pct','mean'),
        mean_interval_score=('interval_score','mean'),observations=('covered','size')).to_csv(out/'quartiles.csv')
    # Fixed comparison figure, not a strategy-selection dashboard.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    methods=cfg['strategies'];labels=['Рубли','Масштаб 2023','Масштаб прогноза','Адаптивный\nмасштаб 2023']
    for model,color in zip(cfg['models'],['#147D64','#4E65A5','#BD6A38']):
        data=summary[(summary.model==model)&(summary.nominal_coverage==.95)].set_index('strategy').reindex(methods)
        for ax,col in zip(axes,['coverage','mean_width_rub','mean_interval_score']):
            ax.plot(range(4),data[col]*(100 if col=='coverage' else 1),marker='o',label=model,color=color)
    axes[0].axhline(95,color='#555555',linestyle='--',linewidth=1,label='Номинал 95%')
    for ax,title in zip(axes,['Покрытие, %','Средняя ширина, руб.','Interval score, руб. ↓']):
        ax.set_title(title);ax.set_xticks(range(4),labels,rotation=15);ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Калибровка h1: сентябрь–декабрь 2024, лаг 0',fontsize=13)
    fig.text(.5,.015,'4 изученных месяца; параметры фиксированы; независимая гарантия покрытия отсутствует.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.95])
    fig.savefig(out/'comparison.png',dpi=180)
    with matplotlib.rc_context({'svg.hashsalt':'interval_calibration_fixed'}):fig.savefig(out/'comparison.svg',metadata={'Date':None})
    plt.close(fig)
    inputs=['results/primary_predictions.parquet','data/consumption.parquet','results/online_interval_summary.csv',
        'configs/interval_calibration.json','docs/protocols/INTERVAL_CALIBRATION_EXPERIMENT.md',
        'src/sberindex/forecasting/interval_calibration.py','src/sberindex/forecasting/operational_audit.py']
    outputs=['predictions.parquet','calibration.csv','monthly.csv','summary.csv','quartiles.csv','comparison.png','comparison.svg']
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'audit.json').write_text(json.dumps(dict(status='passed',baseline_matches_previous=True,
        point_predictions_and_targets_unchanged=True,calibration_strictly_past=True,models_refitted=False,
        independent_validation=False,coverage_guarantee=False,selection='none',reporting_lag=0,
        input_sha256={name:sha(ROOT/name) for name in inputs},output_sha256={name:sha(out/name) for name in outputs},
        limits='Four studied target months; normalized pooled scores and panel-average clipped adaptation are heuristics. No ACI theorem asserted; actual expenditure publication vintages unknown.'),ensure_ascii=False,indent=2)+'\n')
    print(summary[summary.model=='blend_75'].to_string(index=False))

if __name__=='__main__':main()
