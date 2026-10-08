"""Scientific figure for the standalone residual experiment."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/residual_features'
df=pd.read_csv(OUT/'summary.csv')
models=['prophet','seasonal_pooled','blend_selected','residual_hgb_own','residual_hgb_categories','residual_hgb_policy','residual_hgb_wage_snapshot']
labels=['Prophet','Сезонный','Смесь 75/25','Поправка HGB','+ категории','+ решения ЦБ','+ зарплата*']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'residual-features20261005','font.size':10})
fig,axes=plt.subplots(1,3,figsize=(13,5))
for ax,(h,period,title) in zip(axes,[(1,'early','1 месяц · ранние даты'),(1,'late','1 месяц · поздние даты'),(3,'late','3 месяца · поздние даты')]):
    part=df[(df.horizon==h)&(df.period==period)].set_index('model').loc[models]
    bars=ax.barh(range(len(models)),part.MAE_date_balanced,color=['#a4a8ad','#455766','#697998','#278379','#278379','#278379','#c0974a'])
    bars[-1].set_hatch('//');ax.set_yticks(range(len(models)),labels);ax.invert_yaxis()
    for b,value in zip(bars,part.MAE_date_balanced):ax.text(value+22,b.get_y()+b.get_height()/2,f'{value:.0f}',va='center',fontsize=9)
    ax.set_xlim(0,2800 if h==3 else 2200);ax.set_title(title);ax.set_xlabel('MAE, руб. · одинаковые пары');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Поправка к сезонному прогнозу · обучение2023, изученный архив2024',fontsize=12)
fig.text(.5,.015,'* Зарплата: снимок2026, гипотетический лаг2; историческая доступность неизвестна. Новая модель не выбрана.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.93))
for ext in ['png','svg']:fig.savefig(OUT/f'comparison.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else {'Software':'SberIndex research'})
