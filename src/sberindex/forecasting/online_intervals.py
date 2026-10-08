"""Predeclared fixed/expanding/three-month calibration diagnostic, no winner selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from sberindex.forecasting.operational_audit import finite_quantile
from sberindex.paths import ROOT
def main():
    p=pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    p=pd.concat([p,pd.read_parquet(ROOT/'results/adaptive_forecast_predictions.parquet')],ignore_index=True)
    p=p[(p.horizon==1)&p.model.isin(['blend_75','prophet','seasonal_pooled','adaptive_expanding','adaptive_rolling_3m'])&p.target.ge('2024-07')].copy()
    rows=[];calibration=[]
    for model,part in p.groupby('model'):
        for strategy in ['fixed_jul_aug','expanding','rolling_3m']:
            for month in ['2024-09','2024-10','2024-11','2024-12']:
                origin=str(pd.Period(month,freq='M')-1)
                cal=part[(part.target<month)&part.target.ge('2024-07')]
                if strategy=='fixed_jul_aug': cal=cal[cal.target.le('2024-08')]
                if strategy=='rolling_3m': cal=cal[cal.target.ge(str(pd.Period(month,freq='M')-3))]
                assert len(cal)>0 and cal.target.max()<=origin
                test=part[part.target==month]
                for level in [.8,.95]:
                    q=finite_quantile((cal.actual-cal.predicted).abs(),level)
                    calibration.append({'model':model,'strategy':strategy,'target':month,'origin':origin,
                        'nominal_coverage':level,'q':q,'calibration_last':cal.target.max(),'calibration_first':cal.target.min(),
                        'calibration_dates':cal.target.nunique(),'calibration_rows':len(cal)})
                    r=test.copy();r['lower']=(r.predicted-q).clip(lower=0);r['upper']=(r.predicted+q).clip(lower=0)
                    r['covered']=(r.actual>=r.lower)&(r.actual<=r.upper);r['width']=r.upper-r.lower
                    r['interval_score']=r.width+2/(1-level)*((r.lower-r.actual).clip(lower=0)+(r.actual-r.upper).clip(lower=0))
                    r['strategy']=strategy;r['nominal_coverage']=level;rows.append(r)
    r=pd.concat(rows,ignore_index=True)
    r.to_parquet(ROOT/'results/online_intervals.parquet',index=False)
    pd.DataFrame(calibration).to_csv(ROOT/'results/online_interval_calibration.csv',index=False)
    keys=['model','strategy','nominal_coverage']
    summary=r.groupby(keys).agg(coverage=('covered','mean'),mean_width_rub=('width','mean'),mean_interval_score=('interval_score','mean'))
    summary.to_csv(ROOT/'results/online_interval_summary.csv')
    r.groupby(keys+['target']).agg(coverage=('covered','mean'),width_rub=('width','mean')).to_csv(ROOT/'results/online_interval_monthly.csv')
    print(summary.loc['blend_75'].to_string())


if __name__=='__main__':main()
