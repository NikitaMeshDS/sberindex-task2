"""Sparse as-of features from explicit, manually reviewed seasonal PDF claims.

Absence is unknown, not a forecast of normal temperatures. No model is trained.
"""
from pathlib import Path
import json
import pandas as pd

from sberindex.paths import ROOT
def available_claims(claims,origin,horizon):
    origin=pd.Timestamp(origin)
    target=(origin+pd.DateOffset(months=horizon)).strftime('%Y-%m')
    return claims[(claims.available_from<=origin+pd.offsets.MonthEnd(0))&(claims.target==target)].copy()


def main():
    claims=pd.read_csv(ROOT/'data/external/seasonal_weather/claims.csv',parse_dates=['published_date','available_from'])
    assert not claims.duplicated(['bulletin','target','region_code','comparison_baseline']).any()
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv')
    lookup=lookup[(lookup.year==2024)&lookup.territory_id.isin(raw.territory_id)].copy()
    full=raw[raw.category=='Все категории'].pivot(index='territory_id',columns='date',values='value').dropna().index
    lookup=lookup[lookup.territory_id.isin(full)]
    counts=lookup.groupby('region_code').territory_id.nunique()
    records=[]
    for origin in pd.date_range('2023-01-01','2024-11-01',freq='MS'):
        for horizon in [1,3,6,12]:
            for r in available_claims(claims,origin,horizon).itertuples():
                records.append(dict(origin=origin.strftime('%Y-%m'),horizon=horizon,target=r.target,
                    region_code=r.region_code,comparison_baseline=r.comparison_baseline,
                    direction=r.temperature_direction,bulletin=r.bulletin,page=r.page,
                    available_from=r.available_from.strftime('%Y-%m-%d'),municipalities=int(counts.get(r.region_code,0)),
                    has_observed_target=r.target<='2024-12'))
    frame=pd.DataFrame(records)
    frame.to_csv(ROOT/'results/seasonal_news_available_features.csv',index=False)
    eligible=frame[(frame.comparison_baseline=='climatology_1991_2020')&frame.has_observed_target]
    summary=[]
    for h in [1,3,6,12]:
        group=eligible[eligible.horizon==h]
        summary.append(dict(horizon=h,region_origin_target_rows=len(group),origins=group.origin.nunique(),
            target_months=group.target.nunique(),regions=group.region_code.nunique(),municipality_origin_target_rows=int(group.municipalities.sum())))
    pd.DataFrame(summary).to_csv(ROOT/'results/seasonal_news_coverage.csv',index=False)
    audit=dict(claim_rows=len(claims),climatology_claims=int((claims.comparison_baseline=='climatology_1991_2020').sum()),
        previous_year_claims=int((claims.comparison_baseline=='same_month_previous_year').sum()),
        known_h1_training_target_months_2023=sorted(eligible[(eligible.horizon==1)&(eligible.target<'2024-01')].target.unique()),
        next_day_rule='2024-09-30 release becomes available 2024-10-01; not usable at September month-end.',
        interpretation='Sparse explicit qualitative regional claims, not exhaustive weather forecasts. Missing directions stay absent, not zero. Region-wide context is not identical municipality exposure.',
        decision='Two releases and only two h1 training target months in 2023; no standalone forecasting fit or claimed MAE improvement.',
        sources='See data/external/seasonal_weather/protocol.json; current PDFs may differ from historical vintages.')
    (ROOT/'results/seasonal_news_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
