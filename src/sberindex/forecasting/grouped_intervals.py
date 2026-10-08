"""Fixed quartile calibration audit; no selection or coverage theorem."""
import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.forecasting.interval_calibration import calibrate_panel


def calibrate_by_group(frame, scale, groups, config):
    labels = frame.territory_id.map(groups)
    if labels.isna().any():
        raise ValueError('missing fixed municipality group')
    rows, audit = [], []
    for label in sorted(labels.unique()):
        part = frame[labels == label].copy()
        result, cal = calibrate_panel(part, scale, config)
        result['expense_quartile'] = label
        cal['expense_quartile'] = label
        rows.append(result); audit.append(cal)
    return pd.concat(rows, ignore_index=True), pd.concat(audit, ignore_index=True)


def main():
    setup = json.loads((ROOT/'configs/grouped_intervals.json').read_text())
    cfg = json.loads((ROOT/setup['base_config']).read_text())
    cfg['strategies'] = setup['strategies']
    assert setup['groups'] == 4 and setup['scopes'] == ['global','quartile']
    saved = pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    p = saved[(saved.horizon == cfg['horizon']) & saved.model.isin(cfg['models'])
        & saved.target.ge(cfg['calibration_start'])].copy()
    raw = pd.read_parquet(ROOT/'data/consumption.parquet')
    train = raw[(raw.category == 'Все категории') & raw.date.astype(str).str.startswith('2023')]
    scale = train.groupby('territory_id').value.mean().reindex(sorted(p.territory_id.unique()))
    assert (train.groupby('territory_id').date.nunique().reindex(scale.index) == 12).all()
    strata, cuts = pd.qcut(scale, setup['groups'], labels=['Q1','Q2','Q3','Q4'], retbins=True)
    assert strata.value_counts().tolist() == [64]*4
    rows, calibrations = [], []
    for scope in setup['scopes']:
        if scope == 'global':
            r,c = calibrate_panel(p,scale,cfg)
            r['expense_quartile'] = r.territory_id.map(strata).astype(str)
            c['expense_quartile'] = 'ALL'
        else:
            r,c = calibrate_by_group(p,scale,strata.astype(str),cfg)
        r['scope'],c['scope'] = scope,scope
        rows.append(r);calibrations.append(c)
    r = pd.concat(rows,ignore_index=True)
    c = pd.concat(calibrations,ignore_index=True)
    keys=['model','scope','strategy','nominal_coverage']
    # Exact row-level compatibility of the two global controls.
    old = pd.read_parquet(ROOT/'reports/interval_calibration/predictions.parquet')
    for strategy in cfg['strategies']:
        columns=['actual','predicted','lower','upper','covered','width','interval_score']
        index=['model','nominal_coverage','target','territory_id']
        a=r[(r.scope=='global') & (r.strategy==strategy)].set_index(index).sort_index()[columns]
        b=old[old.strategy==strategy].set_index(index).sort_index()[columns]
        pd.testing.assert_frame_equal(a,b)
    reference = r[(r.scope=='global') & (r.strategy=='raw_3m')].set_index(['model','nominal_coverage','target','territory_id'])[['actual','predicted']].sort_index()
    for _,part in r.groupby(['scope','strategy']):
        check=part.set_index(['model','nominal_coverage','target','territory_id'])[['actual','predicted']].sort_index()
        pd.testing.assert_frame_equal(reference,check)
    monthly=r.groupby(keys+['target']).agg(coverage=('covered','mean'),width_rub=('width','mean'),interval_score=('interval_score','mean'),observations=('covered','size')).reset_index()
    quartiles=r.groupby(keys+['expense_quartile']).agg(coverage=('covered','mean'),width_rub=('width','mean'),interval_score=('interval_score','mean'),observations=('covered','size')).reset_index()
    group_monthly=r.groupby(keys+['expense_quartile','target']).agg(coverage=('covered','mean'),width_rub=('width','mean'),observations=('covered','size')).reset_index()
    summary=monthly.groupby(keys).agg(coverage=('coverage','mean'),minimum_date_coverage=('coverage','min'),mean_width_rub=('width_rub','mean'),mean_interval_score=('interval_score','mean'),dates=('target','nunique'),observations=('observations','sum')).reset_index()
    worst=quartiles.groupby(keys).coverage.min().rename('minimum_quartile_coverage').reset_index()
    summary=summary.merge(worst,on=keys,validate='one_to_one')
    out=ROOT/'reports/grouped_intervals';out.mkdir(parents=True,exist_ok=True)
    r.to_parquet(out/'predictions.parquet',index=False)
    for name,frame in [('calibration.csv',c),('monthly.csv',monthly),('quartiles.csv',quartiles),('group_monthly.csv',group_monthly),('summary.csv',summary)]:frame.to_csv(out/name,index=False)
    pd.DataFrame({'territory_id':scale.index,'mean_2023':scale.to_numpy(),'expense_quartile':strata.astype(str).to_numpy()}).to_csv(out/'groups.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    methods=[('global','raw_3m'),('global','scaled_forecast_3m'),('quartile','raw_3m'),('quartile','scaled_forecast_3m')]
    labels=['Общая: рубли','Общая: относительные','По группам: рубли','По группам: относительные']
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=True)
    for ax,model,title in zip(axes,cfg['models'],['Смесь 75/25','Сезонная','Prophet']):
        part=quartiles[(quartiles.model==model)&(quartiles.nominal_coverage==.95)]
        data=np.array([[100*part[(part.scope==scope)&(part.strategy==strategy)&(part.expense_quartile==group)].coverage.iloc[0] for group in ['Q1','Q2','Q3','Q4']] for scope,strategy in methods])
        heat=ax.imshow(data,vmin=50,vmax=100,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(4),['Q1','Q2','Q3','Q4']);ax.set_yticks(range(4),labels);ax.set_title(title)
        for (i,j),value in np.ndenumerate(data):ax.text(j,i,f'{value:.1f}%',ha='center',va='center',color='white' if value>=82 else 'black')
    fig.suptitle('Покрытие 95% интервалов по уровню расходов МО',fontsize=13)
    fig.subplots_adjust(left=.2,right=.91,bottom=.22,top=.8,wspace=.12)
    bar=fig.colorbar(heat,cax=fig.add_axes([.93,.24,.015,.54]));bar.set_label('Покрытие, %')
    fig.text(.5,.06,'Группы по 2023; Q1 — низкие расходы, Q4 — высокие. 4 изученных месяца 2024, h1, лаг 0.',ha='center',fontsize=9)
    fig.savefig(out/'coverage.png',dpi=180)
    with matplotlib.rc_context({'svg.hashsalt':'fixed_grouped_intervals'}):fig.savefig(out/'coverage.svg',metadata={'Date':None})
    plt.close(fig)
    inputs=['configs/grouped_intervals.json',setup['base_config'],'docs/protocols/GROUPED_INTERVAL_EXPERIMENT.md',
        'src/sberindex/forecasting/grouped_intervals.py','src/sberindex/forecasting/interval_calibration.py',
        'src/sberindex/forecasting/operational_audit.py','results/primary_predictions.parquet','data/consumption.parquet',
        'reports/interval_calibration/predictions.parquet']
    names=['predictions.parquet','calibration.csv','monthly.csv','quartiles.csv','group_monthly.csv','summary.csv','groups.csv','coverage.png','coverage.svg']
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'audit.json').write_text(json.dumps(dict(status='passed',global_controls_row_identical=True,
        point_predictions_and_targets_unchanged=True,group_cutpoints_2023=cuts.tolist(),models_refitted=False,
        strategy_selected=False,independent_validation=False,coverage_guarantee=False,reporting_lag=0,
        input_sha256={n:sha(ROOT/n) for n in inputs},output_sha256={n:sha(out/n) for n in names},
        limits='Posthoc hypothesis after inspecting previous interval errors; four reused calendar months. No independent temporal/group coverage theorem. Original forecasting cohort selection is unchanged.'),ensure_ascii=False,indent=2)+'\n')
    print(summary[(summary.model=='blend_75') & (summary.nominal_coverage==.95)].to_string(index=False))

if __name__=='__main__':main()
