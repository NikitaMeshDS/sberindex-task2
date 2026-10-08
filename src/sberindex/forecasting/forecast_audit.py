"""Frozen-model forecast audit: heterogeneity, concentration, scale and coverage.

No model choice or parameter is updated from these retrospective diagnostics.
Expense strata and scaling denominators use 2023 only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
def main():
    p=pd.read_parquet(ROOT/'results/primary_predictions.parquet')
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    panel=raw[raw.category=='Все категории'].pivot(index='territory_id',columns='date',values='value').dropna().sort_index()
    training=panel.iloc[:,:12]
    meta=pd.DataFrame({'expense_2023':training.mean(axis=1),
                       'scale_2023':training.diff(axis=1).abs().iloc[:,1:].mean(axis=1)})
    meta['expense_quartile']=pd.qcut(meta.expense_2023,4,labels=['Q1','Q2','Q3','Q4'])
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv')
    lookup=lookup[lookup.year==2024].set_index('territory_id')
    meta=meta.join(lookup[['region_code','region_name','municipal_district_name_short']],validate='one_to_one')
    p=p.join(meta,on='territory_id',validate='many_to_one')
    assert p.region_code.notna().all() and (p.scale_2023>0).all()
    p['ae']=(p.actual-p.predicted).abs();p['signed_error']=p.predicted-p.actual
    p['scaled_ae']=p.ae/p.scale_2023
    rows=[]
    for (h,model),g in p.groupby(['horizon','model']):
        municipal=g.groupby('territory_id').ae.mean()
        within=g.actual-g.groupby('territory_id').actual.transform('mean')
        denominator=np.square(within).sum()
        rows.append({'horizon':h,'model':model,'MAE':g.ae.mean(),'median_AE':g.ae.median(),
            'p90_AE':g.ae.quantile(.9),'bias_rub':g.signed_error.mean(),'MASE_nonseasonal':g.scaled_ae.mean(),
            'median_municipal_MAE':municipal.median(),'top5pct_municipal_error_share':
                municipal.nlargest(int(np.ceil(.05*len(municipal)))).sum()/municipal.sum(),
            'R2_within_municipality':1-np.square(g.signed_error).sum()/denominator if denominator>0 else np.nan})
    overview=pd.DataFrame(rows);overview.to_csv(ROOT/'results/forecast_audit_overview.csv',index=False)
    groups=[]
    for dimension in ['region_name','expense_quartile','target']:
        for (h,model,label),g in p.groupby(['horizon','model',dimension],observed=True):
            groups.append({'dimension':dimension,'group':str(label),'horizon':h,'model':model,
                'municipalities':g.territory_id.nunique(),'observations':len(g),'MAE':g.ae.mean(),
                'MASE_nonseasonal':g.scaled_ae.mean(),'bias_rub':g.signed_error.mean()})
    pd.DataFrame(groups).to_csv(ROOT/'results/forecast_audit_groups.csv',index=False)
    selected_ids=p.territory_id.unique()
    coverage=meta.assign(selected=meta.index.isin(selected_ids)).groupby(['region_code','region_name']).agg(
        complete_municipalities=('selected','size'),sample_municipalities=('selected','sum'))
    coverage['fraction_sampled']=coverage.sample_municipalities/coverage.complete_municipalities
    coverage.to_csv(ROOT/'results/forecast_sample_coverage.csv')
    paired=[];units=[];loo=[]
    for h in (1,3,6,12):
        chosen='global_hgb' if h==12 else 'blend_75'
        part=p[(p.horizon==h)&p.model.isin([chosen,'prophet'])]
        wide=part.pivot(index=['territory_id','origin','target','region_name','expense_quartile'],columns='model',values='ae').reset_index()
        wide['gain']=wide.prophet-wide[chosen]
        city=wide.groupby('territory_id').gain.mean()
        paired.append({'horizon':h,'model':chosen,'mean_gain_rub':wide.gain.mean(),
            'observation_win_rate':(wide.gain>0).mean(),'municipal_win_rate':(city>0).mean(),
            'municipalities':len(city),'dates':wide.target.nunique()})
        for city_id,gain in city.items():units.append({'horizon':h,'territory_id':city_id,'gain_rub':gain})
        for dimension in ['region_name','target']:
            for label in wide[dimension].unique():
                remaining=wide[wide[dimension]!=label]
                if len(remaining):loo.append({'horizon':h,'dimension':dimension,'excluded':str(label),
                    'remaining_observations':len(remaining),'mean_gain_rub':remaining.gain.mean()})
    paired=pd.DataFrame(paired);paired.to_csv(ROOT/'results/forecast_paired_audit.csv',index=False)
    pd.DataFrame(units).to_csv(ROOT/'results/forecast_municipal_gains.csv',index=False)
    pd.DataFrame(loo).to_csv(ROOT/'results/forecast_leave_group_out.csv',index=False)
    audit={'strata_and_scale_period':'2023 only; quartiles on all complete municipalities',
        'sample_municipalities':len(selected_ids),'complete_municipalities':len(panel),
        'regions_complete_panel':len(coverage),'regions_sampled':int((coverage.sample_municipalities>0).sum()),
        'regions_without_sample':coverage[coverage.sample_municipalities==0].reset_index().region_name.tolist(),
        'limits':'No refit or selection. Nonseasonal MASE uses 11 monthly differences, not 12-month seasonal scaling. Within-municipality R2 is descriptive and undefined at one target per municipality. Leave-group-out gains are sensitivity diagnostics, not confidence intervals.'}
    (ROOT/'results/forecast_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(paired.to_string(index=False))
    print(overview[overview.model.isin(['blend_75','prophet','global_hgb'])].to_string(index=False))


if __name__=='__main__':main()
