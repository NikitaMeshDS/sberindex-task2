"""Known-2023 seasonal normalization audit of fixed foundation models."""
import numpy as np
import pandas as pd


def verify_foundation_seasonal_invariants():
    """Small analytic and future-perturbation checks, safe without model loading."""
    columns = [str(p) for p in pd.period_range('2023-01', periods=24, freq='M')]
    factor = np.arange(1, 13, dtype=float)
    values = np.tile(np.r_[factor, factor], (3, 1)) * np.array([2., 3., 5.])[:, None]
    panel = pd.DataFrame(values, index=[1, 2, 3], columns=columns)
    profile, ids = known_2023_profile(panel)
    np.testing.assert_allclose(profile, factor / factor.mean())
    np.testing.assert_array_equal(ids, [1, 2, 3])
    normalized = normalized_history(panel, 11, profile)
    np.testing.assert_allclose(normalized.loc[1], np.full(12, 2 * factor.mean()))
    restored = restore_forecast(np.full((3, 12), 2 * factor.mean()), 11, profile)
    np.testing.assert_allclose(restored, np.tile(2 * factor, (3, 1)))
    changed = panel.copy()
    changed.iloc[:, 12:] = np.nan
    changed.iloc[0, 12:17] = 1e15
    p2, ids2 = known_2023_profile(changed)
    np.testing.assert_array_equal(ids, ids2)
    np.testing.assert_allclose(profile, p2)
    pd.testing.assert_frame_equal(normalized, normalized_history(changed, 11, p2))
    missing = panel.copy()
    missing.iloc[1, 12] = np.nan
    np.testing.assert_array_equal(normalized_history(missing, 11, profile).index, [1, 2, 3])
    np.testing.assert_array_equal(normalized_history(missing, 12, profile).index, [1, 3])
    assert (restore_forecast(np.array([[-1., 2.]]), 11, profile) >= 0).all()
    return {'analytic_restore': True, 'future_profile_invariance': True,
            'origin_history_invariance': True, 'complete_history_only': True}


def known_2023_profile(panel):
    """Eligibility and pooled factors depend strictly on the first 12 months."""
    assert list(panel.columns[:12]) == [str(p) for p in pd.period_range('2023-01', periods=12, freq='M')]
    known = np.isfinite(panel.iloc[:, :12].to_numpy(float)).all(axis=1)
    ids = panel.index[known].to_numpy(int)
    totals = panel.loc[ids].iloc[:, :12].sum().to_numpy(float)
    if not (np.isfinite(totals).all() and (totals > 0).all()):
        raise ValueError('Seasonal factors require positive finite monthly totals')
    return totals / totals.mean(), ids


def normalized_history(panel, origin, profile):
    history = panel.iloc[:, :origin+1]
    valid = np.isfinite(history.to_numpy(float)).all(axis=1)
    return history.loc[valid].div(profile[np.arange(origin+1) % 12], axis=1)


def restore_forecast(prediction, origin, profile):
    prediction = np.asarray(prediction, dtype=float)
    factors = profile[(origin + 1 + np.arange(prediction.shape[-1])) % 12]
    return np.maximum(0., prediction * factors)


def score_metrics(group):
    from sklearn.metrics import r2_score
    errors = (group.actual-group.predicted).abs()
    return dict(MAE_date_balanced=float(errors.groupby(group.target).mean().mean()),
                MAE_pooled=float(errors.mean()), R2_pooled=float(r2_score(group.actual, group.predicted)),
                WAPE_pct=float(100*errors.sum()/group.actual.sum()),
                dates=int(group.target.nunique()), municipalities=int(group.territory_id.nunique()),
                observations=len(group))


def summarize(predictions, out):
    rows = [dict(horizon=int(h), model=m, **score_metrics(g))
            for (h,m),g in predictions.groupby(['horizon','model'])]
    pd.DataFrame(rows).sort_values(['horizon','MAE_date_balanced']).to_csv(
        out/'foundation_seasonal_summary.csv', index=False)
    monthly = pd.DataFrame([dict(horizon=int(h),model=m,target=t,**score_metrics(g))
        for (h,m,t),g in predictions.groupby(['horizon','model','target'])])
    monthly.to_csv(out/'foundation_seasonal_monthly.csv',index=False)
    contrasts = []
    for name in ['chronos_bolt_tiny','chronos_2']:
        for h,g in monthly[monthly.model.eq(name+'_seasonal')].groupby('horizon'):
            for reference in [name,'seasonal_pooled','regional_asof_selected']:
                ref = monthly[monthly.model.eq(reference) & monthly.horizon.eq(h)]
                if ref.empty:
                    continue
                joined = g.merge(ref,on=['horizon','target'],suffixes=('_normalized','_reference'),validate='one_to_one')
                assert len(joined)==len(g)
                for r in joined.itertuples():
                    contrasts.append(dict(horizon=int(h),model=name+'_seasonal',reference=reference,
                        target=r.target,MAE_difference=r.MAE_pooled_normalized-r.MAE_pooled_reference))
    pd.DataFrame(contrasts).to_csv(out/'foundation_seasonal_paired_monthly.csv',index=False)


def main():
    import hashlib
    import json
    import time
    import torch
    from chronos import Chronos2Pipeline, ChronosBoltPipeline
    from sberindex.paths import ROOT
    out=ROOT/'results'
    config=json.loads((ROOT/'config.json').read_text())
    cohort=json.loads((out/'asof_cohort_protocol.json').read_text())
    checks=verify_foundation_seasonal_invariants()
    raw=pd.read_parquet(ROOT/'data/consumption.parquet')
    panel=raw[raw.category.eq('Все категории')].pivot(index='territory_id',columns='date',values='value').sort_index()
    profile,eligible_ids=known_2023_profile(panel)
    assert len(eligible_ids)==2075
    ids=np.array(cohort['sample_ids'],dtype=int)
    assert set(ids).issubset(eligible_ids)
    panel=panel.loc[ids]
    months=pd.date_range('2023-01-01',periods=24,freq='MS')
    keys=['territory_id','origin','target','horizon']
    source=pd.read_parquet(out/'asof_cohort_scored.parquet')
    selected=source[(source.horizon.eq(12)) | source.origin.ge('2024-06')].copy()
    saved_raw=pd.read_parquet(out/'asof_foundation_predictions.parquet')
    assert set(saved_raw.model)=={'chronos_bolt_tiny','chronos_2'}
    raw_protocol=json.loads((out/'asof_foundation_protocol.json').read_text())
    assert raw_protocol['chronos_bolt_revision']==config['chronos_revision']
    assert raw_protocol['chronos_2_revision']==config['chronos2_revision']
    reference=selected[selected.model.eq('seasonal_pooled')][keys+['actual']].sort_values(keys).reset_index(drop=True)
    for name in ['chronos_bolt_tiny','chronos_2']:
        match=saved_raw[(saved_raw.model.eq(name)) & ((saved_raw.horizon.eq(12)) | saved_raw.origin.ge('2024-06'))]
        match=match.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference[keys],match[keys])
        np.testing.assert_allclose(reference.actual,match.actual)
    torch.set_num_threads(2)
    runtimes=[]
    pipelines={}
    for name,cls,model_key,revision_key in [
        ('chronos_bolt_tiny',ChronosBoltPipeline,'chronos_model','chronos_revision'),
        ('chronos_2',Chronos2Pipeline,'chronos2_model','chronos2_revision')]:
        start=time.perf_counter()
        pipelines[name]=cls.from_pretrained(config[model_key],revision=config[revision_key],device_map='cpu',local_files_only=True)
        runtimes.append(dict(model=name,stage='load',origin='',seconds=time.perf_counter()-start,municipalities=0,prediction_length=0))
    records=[]
    for origin in [11,*range(17,23)]:
        horizons=[12] if origin==11 else [h for h in [1,3,6] if origin+h<24]
        max_h=max(horizons)
        active=normalized_history(panel,origin,profile)
        active_ids=active.index.to_numpy(int)
        for name,pipeline in pipelines.items():
            start=time.perf_counter()
            with torch.inference_mode():
                if name=='chronos_bolt_tiny':
                    parts=[]
                    for batch in range(0,len(active),64):
                        context=torch.from_numpy(active.iloc[batch:batch+64].to_numpy(np.float32))
                        parts.append(pipeline.predict(context,prediction_length=max_h)[:,4,:].detach().cpu().numpy())
                    predicted=np.concatenate(parts)
                else:
                    long=active.stack().rename('target').reset_index()
                    long.columns=['item_id','date','target']
                    long['timestamp']=pd.to_datetime(long.pop('date')+'-01')
                    frame=pipeline.predict_df(long,prediction_length=max_h,quantile_levels=[0.5],batch_size=64,freq='MS')
                    frame['timestamp']=pd.to_datetime(frame.timestamp)
                    predicted=np.column_stack([frame[frame.timestamp.eq(months[origin+h])].set_index('item_id').loc[active_ids,'0.5'].to_numpy(float) for h in range(1,max_h+1)])
            predicted=restore_forecast(predicted,origin,profile)
            assert np.isfinite(predicted).all()
            runtimes.append(dict(model=name,stage='inference',origin=str(months[origin].to_period('M')),seconds=time.perf_counter()-start,municipalities=len(active),prediction_length=max_h))
            for h in horizons:
                actual=panel.loc[active_ids].iloc[:,origin+h].to_numpy(float)
                valid=np.isfinite(actual)
                records.extend(dict(territory_id=int(i),origin=str(months[origin].to_period('M')),target=str(months[origin+h].to_period('M')),horizon=h,model=name+'_seasonal',actual=float(a),predicted=float(p)) for i,a,p in zip(active_ids[valid],actual[valid],predicted[valid,h-1]))
            print(f'{name} normalized origin {months[origin]:%Y-%m}: {runtimes[-1]["seconds"]:.2f}s',flush=True)
    normalized=pd.DataFrame(records)
    for name,g in normalized.groupby('model'):
        ordered=g.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference[keys],ordered[keys])
        np.testing.assert_allclose(reference.actual,ordered.actual)
    regional=pd.read_parquet(out/'asof_regional_predictions.parquet')
    raw_late=saved_raw[saved_raw.horizon.eq(12) | saved_raw.origin.ge('2024-06')]
    baselines=selected[~selected.model.isin(['chronos_bolt_tiny','chronos_2'])]
    comparison=pd.concat([baselines[keys+['model','actual','predicted']],raw_late,normalized,regional],ignore_index=True)
    assert not comparison.duplicated(keys+['model']).any()
    for (_,name),g in comparison.groupby(['horizon','model']):
        ref=reference[reference.horizon.eq(g.horizon.iloc[0])].sort_values(keys).reset_index(drop=True)
        g=g.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(ref[keys],g[keys])
        np.testing.assert_allclose(ref.actual,g.actual)
    comparison['absolute_error']=(comparison.actual-comparison.predicted).abs()
    comparison.to_parquet(out/'foundation_seasonal_predictions.parquet',index=False)
    summarize(comparison,out)
    pd.DataFrame(runtimes).to_csv(out/'foundation_seasonal_runtime.csv',index=False)
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'config.json',ROOT/'data/consumption.parquet',out/'asof_cohort_protocol.json',out/'asof_foundation_predictions.parquet',out/'asof_cohort_scored.parquet',out/'asof_regional_predictions.parquet',ROOT/'docs/protocols/FOUNDATION_SEASONAL_EXPERIMENT.md']}
    protocol=dict(profile_factors=profile.tolist(),profile_ids=eligible_ids.tolist(),eligible_ids=len(eligible_ids),sample_ids=ids.tolist(),chronos_bolt_revision=config['chronos_revision'],chronos_2_revision=config['chronos2_revision'],torch_threads=2,batch_size=64,local_files_only=True,verification=checks,input_sha256=provenance,raw_runtime='Not recorded historically; raw forecasts reused, no cost comparison claimed',limits='Repeatedly studied 2024; no independent test. Zero reporting lag assumed. One date each at horizons 6/12. Pretrained model historical contamination cannot be excluded.')
    protocol['cached_artifact_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'foundation_seasonal_predictions.parquet',out/'foundation_seasonal_runtime.csv']}
    (out/'foundation_seasonal_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(pd.read_csv(out/'foundation_seasonal_summary.csv').to_string(index=False),flush=True)


def refresh(root=None):
    """Rebuild deterministic metric tables from hash-protected cached inference.

    Protocol and cache files are preserved. Their integrity is additionally
    protected by the project reproducibility manifest when bundled.
    """
    import hashlib
    import json
    from sberindex.paths import ROOT
    root = ROOT if root is None else root
    out = root / 'results'
    protocol = json.loads((out/'foundation_seasonal_protocol.json').read_text())
    required = {'results/foundation_seasonal_predictions.parquet',
                'results/foundation_seasonal_runtime.csv'}
    if set(protocol.get('cached_artifact_sha256', {})) != required:
        raise ValueError('Foundation cache hashes are missing or unexpected')
    for path, digest in {**protocol['input_sha256'], **protocol['cached_artifact_sha256']}.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Foundation cached refresh input hash mismatch: {path}')
    frame = pd.read_parquet(out/'foundation_seasonal_predictions.parquet')
    summarize(frame, out)
    return {'refreshed': ['foundation_seasonal_summary.csv',
                          'foundation_seasonal_monthly.csv',
                          'foundation_seasonal_paired_monthly.csv'],
            'inference': False}


def verify_foundation_seasonal_artifacts():
    """Recalculate saved metrics and pair/provenance checks without inference."""
    import hashlib
    import json
    from sberindex.paths import ROOT
    out = ROOT / 'results'
    checks = verify_foundation_seasonal_invariants()
    protocol = json.loads((out/'foundation_seasonal_protocol.json').read_text())
    for path, digest in {**protocol['input_sha256'], **protocol['cached_artifact_sha256']}.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    source = pd.read_parquet(ROOT/'data/consumption.parquet')
    panel = source[source.category.eq('Все категории')].pivot(index='territory_id',columns='date',values='value').sort_index()
    profile, ids = known_2023_profile(panel)
    np.testing.assert_array_equal(ids, protocol['profile_ids'])
    np.testing.assert_allclose(profile, protocol['profile_factors'])
    changed = panel.copy()
    changed.iloc[:, 18:] = np.nan
    profile_changed, ids_changed = known_2023_profile(changed)
    np.testing.assert_array_equal(ids, ids_changed)
    np.testing.assert_allclose(profile, profile_changed)
    pd.testing.assert_frame_equal(normalized_history(panel,17,profile), normalized_history(changed,17,profile_changed))
    frame = pd.read_parquet(out/'foundation_seasonal_predictions.parquet')
    keys = ['territory_id','origin','target','horizon']
    assert not frame.duplicated(keys+['model']).any()
    assert np.isfinite(frame[['actual','predicted']].to_numpy()).all()
    assert frame.predicted.ge(0).all()
    raw = pd.read_parquet(out/'asof_foundation_predictions.parquet')
    summary = pd.read_csv(out/'foundation_seasonal_summary.csv')
    monthly = pd.read_csv(out/'foundation_seasonal_monthly.csv')
    for (h,m),group in frame.groupby(['horizon','model']):
        reference = frame[frame.horizon.eq(h) & frame.model.eq('seasonal_pooled')].sort_values(keys).reset_index(drop=True)
        group = group.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference[keys],group[keys])
        np.testing.assert_allclose(reference.actual,group.actual)
        if m in ['chronos_bolt_tiny','chronos_2']:
            saved = raw[raw.horizon.eq(h)&raw.model.eq(m)&(raw.horizon.eq(12)|raw.origin.ge('2024-06'))].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(saved[keys],group[keys])
            np.testing.assert_array_equal(saved.predicted,group.predicted)
        expected = score_metrics(group)
        row = summary[summary.horizon.eq(h)&summary.model.eq(m)].iloc[0]
        for key,value in expected.items():
            np.testing.assert_allclose(row[key],value,rtol=1e-12)
        for target,part in group.groupby('target'):
            row = monthly[monthly.horizon.eq(h)&monthly.model.eq(m)&monthly.target.eq(target)].iloc[0]
            for key,value in score_metrics(part).items():
                np.testing.assert_allclose(row[key],value,rtol=1e-12)
    # Protect per-date claims as well as aggregate metrics.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as temporary:
        summarize(frame, Path(temporary))
        expected_pairs = pd.read_csv(Path(temporary)/'foundation_seasonal_paired_monthly.csv')
    saved_pairs = pd.read_csv(out/'foundation_seasonal_paired_monthly.csv')
    pd.testing.assert_frame_equal(saved_pairs, expected_pairs, check_exact=False, rtol=1e-12)
    checks.update(saved_pairs=True,raw_reuse_exact=True,saved_metrics=True,paired_monthly_metrics=True,input_provenance=True,real_data_future_invariance=True)
    return checks


if __name__=='__main__':
    import sys
    if '--refresh' in sys.argv:
        print(refresh())
    elif '--verify' in sys.argv:
        print(verify_foundation_seasonal_invariants())
    else:
        main()
