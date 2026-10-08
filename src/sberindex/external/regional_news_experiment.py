"""Frozen regional-news eligibility audit; refuse fitting an underidentified corpus.

This module deliberately never calculates spending errors when the predeclared
release-diversity gate fails. Current snapshots do not establish historical
attachment vintages. Regional directions are context, not municipality exposure.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from sberindex.paths import ROOT
from sberindex.external.news_corpus_audit import select_training_cohort

OUT = ROOT / 'data/external/regional_news_expansion'
PREFIX = ROOT / 'results/regional_news_experiment'
BASELINES = ['climatology_1991_2020', 'source_climate_normal_period_unspecified']


def select_claims(registry):
    return registry[(registry.event_kind.isin(['heating_forecast', 'warm_forecast'])) &
                    registry.comparison_baseline.isin(BASELINES)].copy()


def feature_vector(claims, region, origin, target):
    """Separate unknown, explicitly negative, and nonnegative claims."""
    use = claims[(claims.region_code == region) & (claims.target == target) &
                 (claims.available_from <= pd.Timestamp(origin) + pd.offsets.MonthEnd(0))]
    vector = []
    for baseline in BASELINES:
        for variable in ['temperature', 'precipitation']:
            group = use[(use.comparison_baseline == baseline) & (use.variable == variable)]
            directions = group.value.astype(float).unique()
            if len(directions) > 1:
                raise ValueError('Conflicting same-baseline regional directions')
            known = bool(len(directions))
            vector.extend([float(directions[0]) if known else np.nan, int(known),
                           int(known and directions[0] < 0)])
    return vector


def load_claims():
    registry = pd.read_csv(ROOT / 'results/news_claim_registry.csv')
    claims = select_claims(registry)
    claims['region_code'] = claims.region_codes.map(lambda x: json.loads(x)[0])
    claims['target'] = claims.event_start.str[:7]
    claims['available_from'] = pd.to_datetime(claims.available_from)
    return claims


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    raw = pd.read_parquet(ROOT / 'data/consumption.parquet')
    panel = select_training_cohort(raw[raw.category == 'Все категории'].pivot(
        index='territory_id', columns='date', values='value'))
    assert len(panel) == 2075
    lookup = pd.read_csv(ROOT / 'results/municipal_lookup.csv')
    claims = load_claims()
    rows, feature_rows, eligible_claim_rows = [], [], []
    for stage, origins in [('training', pd.date_range('2023-03-01', '2023-11-01', freq='MS')),
                           ('evaluation', pd.date_range('2024-01-01', '2024-11-01', freq='MS'))]:
        for origin in origins:
            origin_key = origin.strftime('%Y-%m')
            target = (origin + pd.DateOffset(months=1)).strftime('%Y-%m')
            position = list(panel.columns).index(origin_key)
            observed = panel.iloc[:, position-2:position+2].notna().all(axis=1)
            mapping = lookup[lookup.year == origin.year].set_index('territory_id').region_code
            assert mapping.index.is_unique
            regions = mapping.reindex(panel.index)
            assert regions[observed].notna().all()
            # Retired training IDs have no 2024 lookup and no scored targets.
            regions = regions.fillna(-1).astype(int)
            use = claims[(claims.target == target) &
                         (claims.available_from <= origin + pd.offsets.MonthEnd(0)) &
                         claims.region_code.isin(regions[observed])]
            covered = regions.isin(use.region_code)
            rows.append(dict(stage=stage, origin=origin_key, target=target,
                eligible_ids=len(panel), observed_pairs=int(observed.sum()),
                news_covered_pairs=int((covered & observed).sum()),
                news_unknown_pairs=int((~covered & observed).sum()),
                claim_count=len(use), regions=use.region_code.nunique(),
                release_dates=use.published_date.nunique(), source_documents=use.source_url.nunique()))
            for r in use.itertuples():
                eligible_claim_rows.append(dict(stage=stage,origin=origin_key,target=target,
                    claim_id=r.claim_id,region_code=r.region_code,published_date=r.published_date,
                    available_from=r.available_from.strftime('%Y-%m-%d'),source_url=r.source_url))
            for region in sorted(regions.unique()):
                feature_rows.append([stage,origin_key,target,int(region),*feature_vector(claims,region,origin,target)])
    coverage = pd.DataFrame(rows)
    eligible = pd.DataFrame(eligible_claim_rows)
    coverage.to_csv(str(PREFIX)+'_coverage.csv',index=False)
    eligible.to_csv(str(PREFIX)+'_eligible_claims.csv',index=False)
    columns = ['stage','origin','target','region_code']
    for baseline in BASELINES:
        for variable in ['temperature','precipitation']:
            columns.extend([f'{baseline}_{variable}_{kind}' for kind in ['direction','known','negative']])
    pd.DataFrame(feature_rows,columns=columns).to_csv(str(PREFIX)+'_features.csv',index=False)
    counts = {}
    for stage in ['training','evaluation']:
        group=eligible[eligible.stage==stage]
        counts[f'distinct_{stage}_publication_dates']=int(group.published_date.nunique())
        counts[f'covered_{stage}_target_months']=int(group.target.nunique())
        counts[f'{stage}_source_documents']=int(group.source_url.nunique())
        counts[f'{stage}_news_covered_pairs']=int(coverage[coverage.stage==stage].news_covered_pairs.sum())
    failures=[dict(requirement=k,required=v,observed=counts[k]) for k,v in protocol['minimum_gate'].items() if counts[k]<v]
    result=dict(protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        cohort_ids=len(panel),counts=counts,gate_passed=not failures,failed_requirements=failures,
        fit_status='rejected_before_spending_error_inspection' if failures else 'eligible_for_separate_fixed_HGB_fit',
        spending_errors_calculated=False, independent_holdout=False,
        warnings='No existing dated warning overlaps a scored h1 target while available by origin month end; 2023-12 releases target Jan2024 outside evaluated Feb-Dec2024.',
        unknown_semantics='NaN direction/known0/negative0; no direction0 manufactured',
        rejection='Insufficient distinct publication dates in training and evaluation. No empirical MAE advantage claimed.',
        limitations=['Single reviewer','Current snapshot, historical vintage unverified','2024 repeatedly reused','Regional context not municipal exposure','Gate is minimal diversity rule, not power calculation'])
    type(ROOT)(str(PREFIX)+'_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


def verify():
    """Offline source integrity, point-in-time and unknown/negative checks."""
    checks=json.loads((OUT/'source_checks.json').read_text())
    for record in checks:
        if record['status']=='downloaded':
            p=ROOT/record['path']
            assert p.stat().st_size==record['bytes']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
    audit=json.loads(type(ROOT)(str(PREFIX)+'_audit.json').read_text())
    assert audit['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    assert audit['cohort_ids']==2075 and not audit['spending_errors_calculated']
    assert audit['failed_requirements'] and not audit['gate_passed']
    c=pd.read_csv(str(PREFIX)+'_eligible_claims.csv')
    assert (pd.to_datetime(c.available_from)>pd.to_datetime(c.published_date)).all()
    assert (pd.to_datetime(c.available_from)<=pd.to_datetime(c.origin)+pd.offsets.MonthEnd(0)).all()
    f=pd.read_csv(str(PREFIX)+'_features.csv')
    for col in [c for c in f.columns if c.endswith('_direction')]:
        stem=col.removesuffix('_direction')
        assert f.loc[f[stem+'_known']==0,col].isna().all()
        assert f.loc[f[stem+'_known']==1,col].isin([-1,1]).all()
        assert (f[stem+'_negative']==(f[col]<0).astype(int)).all()
    coverage=pd.read_csv(str(PREFIX)+'_coverage.csv')
    assert (coverage.news_covered_pairs+coverage.news_unknown_pairs==coverage.observed_pairs).all()
    assert len(coverage)==20
    events=pd.read_csv(OUT/'event_registry.csv')
    assert events.event_id.is_unique
    assert events.allowed_role.eq('context_only_not_spending_shift_ground_truth').all()
    assert events.period_semantics.eq('reported_observation_day_not_onset_or_duration').all()
    assert (pd.to_datetime(events.available_from)>pd.to_datetime(events.published_date)).all()
    unavailable=pd.read_csv(OUT/'unavailable_claims.csv')
    assert unavailable.published_date.isna().all() and unavailable.available_from.isna().all()
    assert unavailable.allowed_role.eq('excluded_publication_unverified').all()
    return {'regional_news_experiment':'passed','fit_rejected':True,'source_snapshots':sum(r['status']=='downloaded' for r in checks)}


if __name__=='__main__':
    main()
