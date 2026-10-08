"""Source-frozen local event illustrations; events are not spending-shift labels."""
import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.detection.change_detection import CATEGORIES

DIRECTORY='data/external/real_event_registry'
LINKS=[('real_2024_001',1600,'Обь','Новосибирская область'),
 ('real_2024_002',627,'Новороссийск','Краснодарский край'),
 ('real_2024_003',2278,'Карабашский','Челябинская область'),
 ('real_2024_004',221,'Сегежский','Республика Карелия'),
 ('real_2024_005',1798,'Новочеркасск','Ростовская область'),
 ('real_2024_005',1829,'Октябрьский','Ростовская область'),
 ('real_2024_006',478,'Чебоксары','Чувашская Республика')]
SOURCES=['data/consumption.parquet','results/municipal_lookup.csv','results/bocpd_scores.parquet',
 'docs/protocols/REAL_EVENT_MATCHING.md','docs/protocols/REAL_EVENT_REGISTRY_EXPERIMENT.md',
 'src/sberindex/external/real_event_registry.py','src/sberindex/detection/change_detection.py',*[f'{DIRECTORY}/{n}' for n in ['registry.csv','rejected_unknown.csv','selection_freeze.json','capture_manifest.json']]]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate_registry():
    freeze=json.loads((ROOT/f'{DIRECTORY}/selection_freeze.json').read_text())
    assert sha(ROOT/f'{DIRECTORY}/registry.csv')==freeze['registry_sha256']
    registry=pd.read_csv(ROOT/f'{DIRECTORY}/registry.csv').fillna('')
    assert registry.event_id.tolist()==freeze['event_ids'] and registry.event_id.is_unique
    assert not freeze['expense_data_accessed'] and not freeze['expense_forecasts_or_alarms_accessed']
    manifest=json.loads((ROOT/f'{DIRECTORY}/capture_manifest.json').read_text())
    for entry in manifest:assert sha(ROOT/entry['snapshot_path'])==entry['sha256'],entry['snapshot_path']
    assert set(registry.snapshot_path)<=set(e['snapshot_path'] for e in manifest)
    return registry,manifest

def build():
    registry,_=validate_registry();lookup=pd.read_csv(ROOT/'results/municipal_lookup.csv');geo=[]
    for event_id,tid,name,region in LINKS:
        match=lookup[lookup.year.eq(2024)&lookup.territory_id.eq(tid)]
        assert len(match)==1
        r=match.iloc[0];assert r.municipal_district_name_short==name and r.region_name==region
        geo.append(dict(event_id=event_id,territory_id=tid,municipality=name,region=region,
            official_name=r.municipal_district_name,year=2024,matching='explicit alias + unique official ID/name/region'))
    geo=pd.DataFrame(geo);assert set(geo.event_id)==set(registry.event_id)
    raw=pd.read_parquet(ROOT/'data/consumption.parquet');cases=[]
    for link in geo.itertuples():
        event=registry.set_index('event_id').loc[link.event_id];onset=pd.Period(event.event_start,freq='M')
        for category in CATEGORIES:
            facts=raw[raw.territory_id.eq(link.territory_id)&raw.category.eq(category)].set_index('date').value
            for target in [f'2024-{m:02d}' for m in range(1,13)]:
                value=facts.get(target,np.nan);previous=facts.get(target.replace('2024','2023'),np.nan)
                cases.append(dict(event_id=link.event_id,territory_id=link.territory_id,municipality=link.municipality,
                    category=category,target=target,spending_rub=value,yoy_pct=100*(value/previous-1),
                    relative_month=pd.Period(target,freq='M').ordinal-onset.ordinal,
                    published_at=event.published_at,event_start=event.event_start,event_end=event.event_end,
                    source_url=event.source_url,source_observed=bool(np.isfinite(value)),
                    publication_month_known_by_target=event.published_at[:7]<=target))
    cases=pd.DataFrame(cases)
    scores=pd.read_parquet(ROOT/'results/bocpd_scores.parquet')
    assert not scores.duplicated(['territory_id','category','target','method','scaling']).any()
    alarms=cases.merge(scores,on=['territory_id','category','target'],how='left',validate='many_to_many',suffixes=('','_signal'))
    alarms['cohort_available']=alarms.method.notna()
    alarms=alarms[alarms.relative_month.between(-1,1)].copy().reset_index(drop=True)
    rows=[]
    for link in geo.itertuples():
        part=alarms[alarms.event_id.eq(link.event_id)&alarms.territory_id.eq(link.territory_id)&alarms.category.eq('Все категории')&alarms.scaling.eq('regularized_noise')]
        for method in ['spike','rolling_3m','ewma','cusum','bocpd']:
            g=part[part.method.eq(method)]
            record=dict(event_id=link.event_id,territory_id=link.territory_id,municipality=link.municipality,method=method)
            for offset,label in [(-1,'before'),(0,'event'),(1,'after')]:
                row=g[g.relative_month.eq(offset)]
                record[f'{label}_alarm']=bool(row.alarm.iloc[0]) if len(row) and pd.notna(row.alarm.iloc[0]) else np.nan
                record[f'{label}_score_observed']=bool(row.score_observed.iloc[0]) if len(row) and pd.notna(row.score_observed.iloc[0]) else False
            rows.append(record)
    return dict(geography=geo,cases=cases,alarms=alarms,summary=pd.DataFrame(rows))

def input_paths():
    _,manifest=validate_registry();return SOURCES+[e['snapshot_path'] for e in manifest]

def main():
    tables=build()
    for name,t in tables.items():t.to_csv(ROOT/f'results/real_registry_{name}.csv',index=False)
    p=dict(input_sha256={s:sha(ROOT/s) for s in input_paths()},events=6,municipal_links=7,
        limits='Source-selected convenience sample; reused2024; event context not causal or spending-shift truth; unknown publication revisions; monthly aggregation.')
    (ROOT/'results/real_registry_protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))

def verify():
    p=json.loads((ROOT/'results/real_registry_protocol.json').read_text());assert set(p['input_sha256'])==set(input_paths())
    for s,h in p['input_sha256'].items():assert sha(ROOT/s)==h,s
    tables=build()
    for name,t in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(ROOT/f'results/real_registry_{name}.csv').fillna(''),t.fillna(''),check_dtype=False,rtol=1e-10,atol=1e-12)
    assert len(tables['geography'])==7 and len(tables['cases'])==504 and len(tables['summary'])==35
    assert not tables['cases'].duplicated(['event_id','territory_id','category','target']).any()
    return True

if __name__=='__main__':
    import sys
    print(verify()) if '--verify' in sys.argv else main()
