"""Descriptive tradeoffs from the frozen matched BOCPD experiment; no fitting.

Run with ../venv/bin/python tools/detector_tradeoffs.py [--verify].
This standalone research appendix does not change the reference refresh pipeline.
"""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/detector_tradeoffs'
INPUT = ROOT / 'results/bocpd_summary.csv'
LABELS = {'spike':'Разовый сигнал', 'rolling_3m':'Среднее 3м', 'ewma':'EWMA', 'cusum':'CUSUM', 'bocpd':'BOCPD'}
SHAPES = {'step':'Ступень', 'pulse':'Импульс', 'ramp':'Плавный рост'}


def tables():
    source = pd.read_csv(INPUT)
    data = source[(source.category == 'Все категории') & (source.scaling == 'regularized_noise')].copy()
    keys = ['shift_pct','shape','reporting_delay_months','method']
    assert len(data) == 90 and not data.duplicated(keys).any()
    assert set(data.method) == set(LABELS)
    assert set(data.shift_pct) == {-20,20}
    assert set(data.reporting_delay_months) == {0,1,2}
    rows = []
    for _, group in data.groupby(['shift_pct','shape','reporting_delay_months'], sort=True):
        for horizon in [1,3]:
            hit = f'new_by_{"first" if horizon == 1 else "third"}_calendar_month'
            for _, row in group.iterrows():
                burden = float(row.original_control_per100_delivered_months)
                rate = float(row[hit])
                dominates = (group.original_control_per100_delivered_months <= burden) & (group[hit] >= rate) & ((group.original_control_per100_delivered_months < burden) | (group[hit] > rate))
                rows.append({**{key:row[key] for key in keys}, 'deadline_months':horizon, 'new_detection_rate':rate, 'original_control_per100_delivered_months':burden, 'pareto':not dominates.any(), 'dominated_by':','.join(sorted(group.loc[dominates,'method']))})
    frontier = pd.DataFrame(rows)
    robust = data.groupby(['shape','method'], sort=True).agg(
        third_month_min=('new_by_third_calendar_month','min'),
        third_month_max=('new_by_third_calendar_month','max'),
        original_control_burden_max=('original_control_per100_delivered_months','max'),
        scenario_count=('shift_pct','size')).reset_index()
    assert (robust.scenario_count == 6).all()
    illustration=[]
    for shape, group in robust.groupby('shape', sort=True):
        for cap in [.1,2.,10.]:
            eligible=group[group.original_control_burden_max <= cap]
            best=eligible.third_month_min.max() if len(eligible) else None
            tied=eligible[eligible.third_month_min == best] if best is not None else eligible
            illustration.append({'shape':shape,'illustrative_capacity_per100_months':cap,'eligible_methods':','.join(sorted(eligible.method)),'highest_minimum_methods':','.join(sorted(tied.method)),'third_month_min':best})
    return frontier,robust,pd.DataFrame(illustration)


def render(frontier):
    selected=frontier[(frontier.shift_pct == 20)&(frontier.reporting_delay_months == 0)&(frontier.deadline_months == 3)]
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'sberindex-detector-tradeoffs','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    colors={'spike':'#3264ad','rolling_3m':'#a26318','ewma':'#20836b','cusum':'#a94054','bocpd':'#72499b'}
    for ax,(shape,title) in zip(axes,SHAPES.items()):
        group=selected[selected['shape']==shape]
        for _,row in group.iterrows():
            ax.scatter(row.original_control_per100_delivered_months,100*row.new_detection_rate,c=colors[row.method],s=68,marker='o' if row.pareto else 'x')
            offset = (5,-16) if (shape == 'step' and row.method == 'ewma') or (shape == 'ramp' and row.method == 'spike') else (5,5)
            ax.annotate(LABELS[row.method],(row.original_control_per100_delivered_months,100*row.new_detection_rate),xytext=offset,textcoords='offset points',fontsize=8)
        ax.set_title(title);ax.set_xlim(-.7,19);ax.set_ylim(-5,105);ax.grid(alpha=.18)
        ax.set_xlabel('Исходные тревоги / 100 МО-месяцев')
    axes[0].set_ylabel('Новое обнаружение к 3-му месяцу, %')
    fig.suptitle('Компромисс при +20% и лаге 0 · «Все категории», regularized_noise',fontsize=12)
    fig.text(.5,.01,'● На границе Парето   × Доминируется другим методом · Исходные тревоги не являются доказанными ложными',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    for ext in ['png','svg']:
        fig.savefig(DEST/f'tradeoffs.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else {'Software':'SberIndex research appendix'})
    plt.close(fig)


def main():
    outputs = dict(zip(['scenario_frontiers.csv','scenario_envelopes.csv','capacity_illustrations.csv'],tables()))
    digests={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [INPUT,Path(__file__).resolve(),ROOT/'configs/detectors.json',ROOT/'results/bocpd_protocol.json']}
    if '--verify' in sys.argv:
        audit=json.loads((DEST/'audit.json').read_text())
        assert audit['input_sha256']==digests
        for name,frame in outputs.items():
            pd.testing.assert_frame_equal(frame,pd.read_csv(DEST/name,keep_default_na=False),check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
        for name,digest in audit['output_sha256'].items():assert hashlib.sha256((DEST/name).read_bytes()).hexdigest()==digest,name
        print('Verified 180 frontier rows, 15 envelopes and 9 capacity illustrations; no fitting.')
        return
    DEST.mkdir(parents=True,exist_ok=True)
    for name,frame in outputs.items():frame.to_csv(DEST/name,index=False)
    render(outputs['scenario_frontiers.csv'])
    artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.iterdir()) if p.suffix in ['.csv','.png','.svg']}
    audit={'input_sha256':digests,'output_sha256':artifact_hashes,'rows':{name:len(frame) for name,frame in outputs.items()},'scope':'Descriptive retrospective appendix: 10-seed means, two signs and three hypothetical lags. Pareto is scenario-specific. Capacities are illustrations, not externally justified operating costs. No fitting, threshold tuning, promotion, independent validation or false-positive labels.'}
    (DEST/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(audit['rows'])


if __name__=='__main__':main()
