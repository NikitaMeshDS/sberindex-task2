"""Scientific comparison of delayed residual forecasts, with actual horizons."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/operational_workflow'
f=pd.read_csv(OUT/'summary.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'sberindex-operational20261005'})
fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
colors={'prophet':'#a4a8ad','seasonal_pooled':'#455766','blend_75':'#697998','residual_hgb_policy':'#278379'}
labels={'prophet':'Prophet','seasonal_pooled':'Сезонный','blend_75':'Смесь 75/25','residual_hgb_policy':'HGB + категории/ЦБ'}
for row,period in enumerate(['early','late']):
    for lag in [0,1,2]:
        ax=axes[row,lag];part=f[(f.reporting_lag==lag)&(f.business_horizon==1)&(f.period==period)]
        for model,color in colors.items():
            g=part[part.model==model]
            if len(g):ax.bar(labels[model],g.MAE_date_balanced.iloc[0],color=color)
        n=part.dates.max();ax.set_title(f'Лаг {lag} · эффективный горизонт {lag+1}м · {n} дат')
        ax.tick_params(axis='x',rotation=25,labelsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.set_ylabel(('Ранние цели' if row==0 else 'Поздние цели')+' · MAE, руб.')
fig.suptitle('Бизнес-горизонт 1 месяц · фиксированное обучение2023 · архив2024',fontsize=12)
fig.text(.5,.015,'Сравнение моделей внутри каждого сценария на одинаковых парах. Между лагами число доступных дат различается.\nЗадержки гипотетические; это ретроспективный анализ, HGB не повышен до основной модели.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.94))
for ext in ['png','svg']:fig.savefig(OUT/f'delayed_comparison.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else {'Software':'SberIndex research'})
plt.close(fig)
inputs=[OUT/'summary.csv',Path(__file__).resolve()]
(OUT/'figure_provenance.json').write_text(json.dumps({'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'output_sha256':{f'delayed_comparison.{ext}':hashlib.sha256((OUT/f'delayed_comparison.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']}},indent=2)+'\n')
