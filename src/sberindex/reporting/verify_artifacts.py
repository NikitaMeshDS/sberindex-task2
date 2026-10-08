"""Checks of provenance, join cardinality and temporal model-selection boundaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
def main():
    checks = []
    assert (ROOT/'results/asof_regional_protocol.json').exists(), 'Regional experiment on the 2023-known cohort must be regenerated'
    assert (ROOT/'results/asof_regional_delay_protocol.json').exists(), 'Regional delay transfer must be regenerated'
    manifest = json.loads((ROOT / "data_sources.json").read_text())
    for source in manifest["sources"]:
        content = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == source["sha256"], source["path"]
    checks.append("source snapshot SHA256 matches manifest")
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    joined = pd.read_parquet(ROOT / "results/consumption_with_geography_cpi.parquet")
    key = ["territory_id", "date", "category"]
    assert len(raw) == len(joined) == 303126
    assert not joined.duplicated(key).any()
    assert (joined.year_from <= joined.year).all() and (joined.year < joined.year_to).all()
    assert joined.region_name.notna().all() and joined.cpi_all_mom.notna().all()
    a = raw.set_index(key).value.sort_index()
    b = joined.set_index(key).value.sort_index()
    pd.testing.assert_series_equal(a, b)
    checks.append("version-aware geographic/CPI join preserves every target once")
    crosswalk = pd.read_csv(ROOT / "results/region_crosswalk.csv")
    assert crosswalk.region_code.is_unique and crosswalk.rosstat_region_code.is_unique
    assert crosswalk.set_index("region_code").loc[29, "rosstat_region_code"] == 11001000
    assert crosswalk.set_index("region_code").loc[72, "rosstat_region_code"] == 71001000
    checks.append("parent oblast CPI excludes separately modelled autonomous districts")
    cpi = pd.read_csv(ROOT / "results/rosstat_cpi_monthly.csv")
    national = cpi[cpi.rosstat_region_code == 643].set_index("date")
    assert abs(100 * national.loc["2023-12", "factor_all_yoy"] - 107.42) < .08
    assert abs(100 * national.loc["2024-12", "factor_all_yoy"] - 109.52) < .08
    checks.append("chained CPI reconciles with official December annual inflation within rounding")
    p = pd.read_parquet(ROOT / "results/primary_predictions.parquet")
    assert not p.duplicated(["territory_id", "origin", "horizon", "model"]).any()
    assert p[p.model == "blend_75"].origin.ge("2024-06").all()
    for h, dates in [(1, 6), (3, 4), (6, 1), (12, 1)]:
        part = p[p.horizon == h]
        keys = None
        for model, group in part.groupby("model"):
            current = set(zip(group.territory_id, group.origin))
            if keys is None: keys = current
            assert keys == current, (h, model)
            assert group.origin.nunique() == dates
        for model, group in part.groupby("model"):
            metric = pd.read_csv(ROOT / "results/forecast_comparison.csv")
            expected = metric[(metric.horizon == h) & (metric.model == model)].MAE.item()
            assert np.isclose(np.abs(group.actual - group.predicted).mean(), expected)
    checks.append("all primary model comparisons use identical observations after tuning cutoff")
    events = pd.read_csv(ROOT / "data/external/news_events.csv", parse_dates=["available_from", "published_date"])
    assert (events.available_from > events.published_date).all()
    news = pd.read_csv(ROOT / "results/news_asof_features.csv")
    assert (pd.to_datetime(news.latest_news_date) <= pd.to_datetime(news.origin) + pd.offsets.MonthEnd(0)).all()
    checks.append("news features precede their forecast origin")
    from sberindex.detection.hierarchical_changes import signals, fit, evaluate
    # A noiseless known example must retain a common jump and leave local residuals quiet.
    toy = np.ones((20, 24)) * 100
    regions = np.repeat([1, 2], 10)
    original = signals(toy, regions, np.array([1, 2]))
    fitted = fit(original)
    changed = toy.copy(); changed[:, 18:] *= 1.20
    altered = signals(changed, regions, np.array([1, 2]))
    for method in ("spike", "rolling_3m", "ewma", "cusum"):
        for level in ("national", "regional", "local"):
            center, threshold = fitted[level, method]
            assert not evaluate(original[level], center, threshold, method).any()
            detected = evaluate(altered[level], center, threshold, method).any(axis=1)
            assert (not detected.any()) if level == "local" else detected.all()
    audit = json.loads((ROOT / "results/hierarchical_audit.json").read_text())
    assert audit["future_perturbation_max_error"] < 1e-12
    assert audit["common_shift_preservation_max_error"] < 1e-12
    checks.append("hierarchical channels preserve a known common jump without broadcasting local alarms; fit and July scores exclude future data")
    from sberindex.detection.forecast_residual_changes import predict
    for model in ("adaptive_1m", "frozen_6m"):
        original_error = np.log(toy[:, 12:] / predict(toy, model))
        altered_error = np.log(changed[:, 12:] / predict(changed, model))
        expected = np.zeros(12)
        expected[6:] = np.log(1.2)
        if model == "adaptive_1m": expected[7:] = 0
        np.testing.assert_allclose(altered_error-original_error, np.tile(expected, (20, 1)), atol=1e-12)
    audit = json.loads((ROOT / "results/forecast_residual_audit.json").read_text())
    assert audit["future_prediction_max_error"] < 1e-12
    assert audit["calibration_error_after_injection"] < 1e-12
    checks.append("forecast residuals preserve frozen level shift and show adaptive absorption without future-dependent predictions")
    from sberindex.detection.shift_confirmation import monitor
    permanent = np.array([[0, .2, .2, .2, .2, .2, .2]])
    pulse = np.array([[0, .2, 0, 0, 0, 0, 0]])
    late = np.array([[0, 0, 0, 0, 0, .2, .2]])
    for sign in (-1, 1):
        out = monitor(sign*permanent, .1)
        assert out[0][0, 0] and out[1][0, 2] and out[5][0, 2] == sign
        assert not out[1][0, :2].any()
        assert not monitor(sign*pulse, .1)[1].any()
        out = monitor(sign*late, .1)
        assert not out[1].any() and out[3][0]
    audit = json.loads((ROOT / "results/confirmation_audit.json").read_text())
    assert audit["future_prefix_mismatches"] == 0
    checks.append("sequential confirmation distinguishes noiseless step/pulse, preserves direction and leaves late triggers unresolved")
    overview = pd.read_csv(ROOT / "results/forecast_audit_overview.csv")
    reference = pd.read_csv(ROOT / "results/forecast_comparison.csv")
    joined_metrics = overview.merge(reference, on=["horizon", "model"], validate="one_to_one", suffixes=("_audit", "_main"))
    assert len(joined_metrics) == len(reference)
    np.testing.assert_allclose(joined_metrics.MAE_audit, joined_metrics.MAE_main)
    coverage = pd.read_csv(ROOT / "results/forecast_sample_coverage.csv")
    assert coverage.sample_municipalities.sum() == 256
    assert coverage.complete_municipalities.sum() == 2016
    assert overview[overview.horizon.isin([6, 12])].R2_within_municipality.isna().all()
    checks.append("forecast subgroup audit reconciles with main MAE and complete/sample coverage; within-series R2 undefined for one date")
    extension = pd.read_parquet(ROOT / "results/geographic_extension_predictions.parquet")
    assert not set(extension.territory_id) & set(p.territory_id)
    assert extension.region_code.nunique() == 8 and extension.territory_id.nunique() == 30
    assert extension.origin.ge("2024-06").all()
    assert not extension.duplicated(["territory_id", "origin", "horizon", "model"]).any()
    ew = extension.pivot(index=["territory_id", "origin", "horizon"], columns="model", values="predicted")
    np.testing.assert_allclose(ew.blend_75, .75*ew.seasonal_pooled+.25*ew.prophet)
    expected = raw[raw.category == "Все категории"].copy()
    expected["target"] = expected.date.astype(str).str[:7]
    merged = extension.merge(expected[["territory_id", "target", "value"]], on=["territory_id", "target"], validate="many_to_one")
    assert len(merged) == len(extension)
    np.testing.assert_allclose(merged.actual, merged.value)
    checks.append("geographic extension uses disjoint IDs, unchanged blend weight, valid origins and matching source targets")
    delayed = pd.read_parquet(ROOT / "results/reporting_delay_predictions.parquet")
    for delay, group in delayed.groupby("reporting_delay_months"):
        assert (group.horizon == 1+delay).all()
        expected_decision = (pd.PeriodIndex(group.origin, freq="M")+int(delay)).astype(str)
        assert np.array_equal(expected_decision, group.decision_month.to_numpy())
        assert group.target.nunique() == 4 and group.territory_id.nunique() == 256
    online_cal = pd.read_csv(ROOT / "results/online_interval_calibration.csv")
    assert (online_cal.calibration_last <= online_cal.origin).all()
    assert (online_cal.calibration_last < online_cal.target).all()
    intervals = pd.read_parquet(ROOT / "results/online_intervals.parquet")
    assert intervals.lower.ge(0).all() and intervals.upper.ge(intervals.lower).all()
    offline = pd.read_parquet(ROOT / "results/forecast_intervals.parquet")
    offline = offline[offline.interval_method == "absolute"]
    fixed = intervals[intervals.strategy == "fixed_jul_aug"]
    keys = ["territory_id", "target", "model", "nominal_coverage"]
    match = fixed.merge(offline, on=keys, validate="one_to_one", suffixes=("_online", "_fixed"))
    assert len(match) == len(fixed) == len(offline)
    np.testing.assert_allclose(match.lower_online, match.lower_fixed)
    np.testing.assert_allclose(match.upper_online, match.upper_fixed)
    from sberindex.forecasting.operational_audit import finite_quantile
    predictions = pd.concat([pd.read_parquet(ROOT / "results/primary_predictions.parquet"), pd.read_parquet(ROOT / "results/adaptive_forecast_predictions.parquet")], ignore_index=True)
    for row in online_cal.itertuples():
        c = predictions[(predictions.model == row.model) & (predictions.horizon == 1) &
                        predictions.target.between(row.calibration_first, row.calibration_last)]
        expected_q = finite_quantile((c.actual-c.predicted).abs(), row.nominal_coverage)
        assert len(c) == row.calibration_rows and np.isclose(expected_q, row.q)
    checks.append("delay horizons use matching target dates; online interval quantiles recompute from past-only residuals and fixed baseline agrees")
    from sberindex.forecasting.regional_seasonality import ratios
    values = np.arange(20*24, dtype=float).reshape(20,24)+100
    regions = np.repeat([1, 2], 10)
    past_ratio = ratios(values, regions, 17, 20, "shrink_100")
    changed = values.copy(); changed[:, 18:] *= 9
    np.testing.assert_allclose(past_ratio, ratios(changed, regions, 17, 20, "shrink_100"), rtol=0, atol=0)
    regional = pd.read_parquet(ROOT / "results/regional_seasonality_predictions.parquet")
    pooled = regional[(regional.category == "Все категории") & (regional.candidate == "pooled")]
    baseline = p[(p.model == "seasonal_pooled") & p.horizon.isin([1, 3, 6])]
    keys = ["territory_id", "origin", "target", "horizon"]
    m = pooled.merge(baseline, on=keys, validate="one_to_one", suffixes=("_regional", "_baseline"))
    assert len(m) == len(baseline)
    np.testing.assert_allclose(m.predicted_regional, m.predicted_baseline)
    val = pd.read_csv(ROOT / "results/regional_blend_validation.csv")
    selected = json.loads((ROOT / "results/regional_blend_protocol.json").read_text())
    best = val.loc[val.MAE.idxmin()]
    assert best.candidate == selected["candidate"] and best.seasonal_weight == selected["seasonal_weight"]
    assert np.isclose(best.MAE, selected["validation_MAE"])
    checks.append("regional ratios ignore future observations, pooled baseline reproduces existing predictions, blend selection matches early-validation minimum")
    from sberindex.forecasting.adaptive_forecast import choose
    selections = pd.read_csv(ROOT / "results/adaptive_forecast_selections.csv")
    scores = pd.read_csv(ROOT / "results/adaptive_candidate_monthly_errors.csv")
    for row in selections.itertuples():
        c, w, loss, first, last = choose(scores, row.origin, row.policy)
        assert c == row.candidate and np.isclose(w, row.weight) and np.isclose(loss, row.past_MAE)
        assert first == row.calibration_first and last == row.calibration_last and last <= row.origin
        altered = scores.copy(); altered.loc[altered.target > row.origin, "ae"] = -1e9
        assert choose(altered, row.origin, row.policy) == (c, w, loss, first, last)
    adaptive = pd.read_parquet(ROOT / "results/adaptive_forecast_predictions.parquet")
    keys = ["territory_id", "origin", "target", "horizon"]
    for model, group in adaptive.groupby("model"):
        base = p[(p.model == "prophet") & p.horizon.isin([1,3,6])]
        assert set(map(tuple, group[keys].to_numpy())) == set(map(tuple, base[keys].to_numpy()))
    checks.append("sequential selection recomputes from admissible past errors, ignores adversarial future scores and keeps comparison keys identical")
    from sberindex.forecasting.robust_anchor import predict as anchor_predict, profiles as anchor_profiles
    toy_values = np.ones((20,24))*20
    toy_values[:,15:18] = [10,20,30]
    flat = np.ones((20,12))
    for name,expected in [("last",30),("mean_3m",20),("median_3m",20),("weighted_3m",23)]:
        np.testing.assert_allclose(anchor_predict(toy_values,flat,17,1,name),expected)
    region_ids = np.repeat([1,2],10)
    original = anchor_profiles(toy_values,region_ids,"regional_shrink100")
    perturbed = toy_values.copy(); perturbed[:,18:] *= 9
    np.testing.assert_allclose(original,anchor_profiles(perturbed,region_ids,"regional_shrink100"))
    rv = pd.read_csv(ROOT / "results/robust_anchor_validation.csv")
    rp = json.loads((ROOT / "results/robust_anchor_protocol.json").read_text())
    assert rp["future_perturbation_max_error"] == 0
    for (category,cohort), group in rv.groupby(["category","cohort"]):
        selected = rp["selection"][category+"|"+cohort]
        best = group.loc[group.MAE.idxmin()]
        assert best.profile == selected["profile"] and best.anchor == selected["anchor"]
        assert np.isclose(best.MAE, selected["validation_MAE"])
    checks.append("seasonally normalized anchors match analytic examples, ignore future seasonal data and select early-validation minima per cohort")
    from sberindex.external.news_corpus_audit import load_events
    from sberindex.external.news_ablation import known_news
    events = load_events()
    assert len(events) == 17 and (events.delta_bps == 0).sum() == 9
    assert events.published_date.is_unique and events.published_date.is_monotonic_increasing
    assert ((events.available_from-events.published_date).dt.days == 1).all()
    np.testing.assert_allclose(events.rate_before_pct.iloc[1:], events.rate_after_pct.iloc[:-1])
    np.testing.assert_allclose(events.delta_bps, 100*(events.rate_after_pct-events.rate_before_pct))
    old = pd.read_csv(ROOT/'data/external/news_events.csv', parse_dates=['published_date','available_from'])
    pd.testing.assert_frame_equal(events[events.delta_bps!=0][old.columns].reset_index(drop=True), old, check_dtype=False)
    for origin in pd.date_range('2023-03-01', '2024-11-01', freq='MS'):
        end = origin+pd.offsets.MonthEnd(0)
        np.testing.assert_array_equal(known_news(origin,events),known_news(origin,events[events.available_from<=end]))
        np.testing.assert_array_equal(known_news(origin,events)[:2], known_news(origin,old)[:2])
    assert known_news(pd.Timestamp('2024-06-01'),events)[2] == 2
    news = pd.read_parquet(ROOT/'results/news_corpus_predictions.parquet')
    news_protocol = json.loads((ROOT/'results/news_corpus_protocol.json').read_text())
    news_raw = pd.read_parquet(ROOT/'data/consumption.parquet')
    news_panel = news_raw[news_raw.category == 'Все категории'].pivot(
        index='territory_id', columns='date', values='value').sort_index()
    news_known_ids = news_panel.index[news_panel.iloc[:, :12].notna().all(axis=1)]
    assert news_protocol['training_rows'] == len(news_known_ids) * 9, 'News training cohort must depend only on 2023 completeness'
    assert news.groupby('model').size().nunique() == 1
    assert news.groupby(['territory_id','origin']).actual.nunique().eq(1).all()
    assert news.target.min() == '2024-02' and news.target.max() == '2024-12'
    checks.append('17 policy releases preserve original eight hikes and rate chain; next-day availability; future news excluded; identical rates/deltas across corpora and matched forecast observations')
    from sberindex.external.news_corpus_audit import select_training_cohort
    assert news_protocol['training_ids'] == news_known_ids.astype(int).tolist()
    future_removed = news_panel.copy()
    future_removed.iloc[:, 12:] = np.nan
    assert select_training_cohort(future_removed).index.equals(news_known_ids)
    coverage = pd.read_csv(ROOT/'results/news_corpus_coverage.csv')
    cohort_values = news_panel.loc[news_known_ids].to_numpy(float)
    for row in coverage.itertuples():
        i = list(news_panel.columns).index(row.origin)
        history_ok = np.isfinite(cohort_values[:, i-2:i+1]).all(axis=1)
        target_ok = np.isfinite(cohort_values[:, i+1])
        ids = news_known_ids[history_ok & target_ok]
        assert row.eligible_ids == len(news_known_ids)
        assert row.missing_history == int((~history_ok).sum())
        assert row.missing_target == int((~target_ok).sum())
        assert row.scored_ids == len(ids) and row.unscored_ids == len(news_known_ids)-len(ids)
        for _, group in news[news.target == row.target].groupby('model'):
            assert sorted(group.territory_id) == ids.tolist()
            np.testing.assert_allclose(group.sort_values('territory_id').actual, cohort_values[history_ok & target_ok, i+1])
    assert np.isfinite(news[['actual','predicted']].to_numpy()).all()
    news_summary = pd.read_csv(ROOT/'results/news_corpus_summary.csv')
    for row in news_summary.itertuples():
        group = news[(news.period == row.period) & (news.model == row.model)]
        assert np.isclose(row.MAE, abs(group.actual-group.predicted).mean())
        assert row.municipalities == group.territory_id.nunique()
    checks.append('News HGB training cohort ignores all 2024 data; observed three-month inputs and targets define identical scored pairs; coverage and MAE recomputed')
    import re
    from sberindex.external.weather_news_audit import region_pattern
    for region,title,expected in [
        ('Москва','Прогноз по Московской области',False),
        ('Республика Коми','Заседание комиссии и комитета',False),
        ('Республика Коми','Погода в Республике Коми',True),
        ('Московская область','Прогноз по Москве и Московской области',True),
        ('Республика Алтай','В Алтайском крае',False),
        ('Алтайский край','В Республике Алтай',False),
        ('Ненецкий автономный округ','Ямало-Ненецкого автономного округа',False),
        ('Ненецкий автономный округ','В Ненецком автономном округе',True)]:
        assert bool(re.search(region_pattern(region),title.lower())) == expected
    articles = pd.read_csv(ROOT/'results/weather_news_articles.csv')
    pages = pd.read_csv(ROOT/'results/weather_news_pages.csv')
    mentions = pd.read_csv(ROOT/'results/weather_news_region_mentions.csv')
    features = pd.read_csv(ROOT/'results/weather_news_asof_features.csv')
    type_columns=[c for c in features if c.endswith('_title_90d')]
    assert len(type_columns)==4
    np.testing.assert_array_equal(features[type_columns].sum(axis=1),features.explicit_title_mentions_90d)
    assert articles.source_url.is_unique
    for year,group in pages.groupby('year'):
        assert group.parsed.sum() == group.reported.iloc[0]
        assert len(articles[articles.published_date.str.startswith(str(year))]) == group.parsed.sum()
    assert set(mentions.source_url) <= set(articles.source_url)
    for origin,group in features.groupby('origin'):
        end=pd.Timestamp(origin)+pd.offsets.MonthEnd(0)
        dates=pd.to_datetime(mentions.available_from)
        known=mentions[(dates<=end)&(dates>end-pd.Timedelta(days=90))]
        counts=known.groupby('region_code').source_url.nunique()
        np.testing.assert_array_equal(group.explicit_title_mentions_90d,[counts.get(c,0) for c in group.region_code])
    checks.append('weather archive counts reconcile per year; regional aliases separate Moscow/oblast, two Altai and two Nenets regions; rolling title counts respect availability')
    from sberindex.external.weather_body_audit import parse_body,coordinated_regions
    toy='<h1>Обзор за 1 января</h1><strong>3 января 2024 г</strong><p>Погода за прошедший день: мороз.</p><script>FAKE REGION</script><!-- begin: BRED CRUMBS -->MENU'
    title,date,body=parse_body(toy)
    assert date=='2024-01-03' and 'FAKE' not in body and 'MENU' not in body
    regions=pd.read_csv(ROOT/'results/region_crosswalk.csv')
    assert set(coordinated_regions('В Курской, Белгородской и Воронежской областях ожидается снег.',regions))=={46,31,36}
    assert coordinated_regions('Комитет обсуждает комиссию.',regions)==[]
    review=pd.read_csv(ROOT/'results/weather_body_review.csv')
    assert len(review)==34 and review.source_url.is_unique
    annotations=pd.read_csv(ROOT/'data/external/weather_body_sample/annotations.csv')
    warnings=pd.read_csv(ROOT/'data/external/weather_body_sample/warning_events.csv')
    forbidden=set(annotations.loc[annotations.temporal_status!='explicit_period','source_url'])
    assert not set(warnings.source_url)&forbidden
    assert annotations[annotations.source_url.str.contains('/38439/')].temporal_status.iloc[0]=='conflicting_dates'
    horizons=pd.read_csv(ROOT/'results/weather_warning_horizons.csv')
    for r in horizons.itertuples():
        e=warnings[warnings.source_url==r.source_url].iloc[0]
        end=pd.Timestamp(r.origin)+pd.offsets.MonthEnd(0)
        assert pd.Timestamp(e.available_from)<=end
        target=pd.Timestamp(r.origin)+pd.DateOffset(months=r.horizon)
        expected=pd.Timestamp(e.event_start)<=target+pd.offsets.MonthEnd(0) and pd.Timestamp(e.event_end)>=target
        assert expected==r.forecast_period_overlaps_target
    overlap=horizons[horizons.forecast_period_overlaps_target]
    assert len(overlap)==1 and overlap.target.iloc[0]=='2024-01' and overlap.horizon.iloc[0]==1
    checks.append('body parser uses publication date and excludes navigation/scripts; coordinated region lists resolve; conflicting/relative dates excluded from warning events; target-overlap chronology recomputed including year boundary')
    from sberindex.external.seasonal_news_alignment import available_claims
    claims=pd.read_csv(ROOT/'data/external/seasonal_weather/claims.csv',parse_dates=['published_date','available_from'])
    assert set(claims.temperature_direction)=={-1,1}
    assert ((claims.available_from-claims.published_date).dt.days==1).all()
    assert available_claims(claims,'2024-09-01',3).empty
    november=available_claims(claims,'2024-10-01',1)
    normal=november[november.comparison_baseline=='climatology_1991_2020']
    yoy=november[november.comparison_baseline=='same_month_previous_year']
    assert set(normal.region_code)=={3,75} and set(yoy.region_code)=={87}
    altered=claims.copy();altered.loc[altered.available_from>'2023-12-31','temperature_direction'] *= -1
    pd.testing.assert_frame_equal(available_claims(claims,'2023-11-01',1),available_claims(altered,'2023-11-01',1))
    seasonal=pd.read_csv(ROOT/'results/seasonal_news_available_features.csv')
    for r in seasonal.itertuples():
        assert pd.Timestamp(r.available_from)<=pd.Timestamp(r.origin)+pd.offsets.MonthEnd(0)
        assert (pd.Timestamp(r.origin)+pd.DateOffset(months=r.horizon)).strftime('%Y-%m')==r.target
    checks.append('seasonal claims keep climate-normal and prior-year comparisons distinct; unknown stays absent; September30 release excluded at September origin; future bulletin perturbations cannot affect old features')
    news_pairs=pd.read_csv(ROOT/'results/news_pair_eligibility.csv')
    news_coverage=pd.read_csv(ROOT/'results/news_pair_coverage.csv')
    news_keys=['territory_id','origin','target','horizon']
    assert not news_pairs.duplicated(news_keys).any()
    expected_pairs=pd.read_parquet(ROOT/'results/asof_cohort_scored.parquet',columns=news_keys).drop_duplicates()
    assert len(news_pairs)==len(expected_pairs)==3013
    assert len(news_pairs.merge(expected_pairs,on=news_keys,validate='one_to_one'))==len(expected_pairs)
    assert news_pairs.groupby('horizon').size().to_dict()=={1:1506,3:1004,6:251,12:252}
    assert news_pairs.groupby('horizon').seasonal_claims_eligible.sum().to_dict()=={1:4,3:0,6:0,12:0}
    assert news_pairs.weather_warnings_eligible.sum()==0
    assert news_pairs.warm_temperature_claims_eligible.sum()==0
    assert news_pairs.groupby('horizon').warm_precipitation_claims_eligible.sum().to_dict()=={1:15,3:9,6:0,12:0}
    warm=news_pairs[news_pairs.warm_precipitation_claims_eligible>0]
    assert set(warm.target)=={'2024-08','2024-09'}
    assert set(warm.origin)=={'2024-06','2024-07','2024-08'}
    assert (warm.origin_end.str[:10]>='2024-03-29').all()
    four=news_pairs[news_pairs.seasonal_claims_eligible>0]
    assert set(four.origin)=={'2024-10'} and set(four.target)=={'2024-11'} and set(four.region_code)=={75}
    for r in news_coverage.itertuples():
        source={'CBR national decisions':'announcements_90d',
                'Roshydromet seasonal regional claims':'seasonal_claims_eligible',
                'Roshydromet dated regional warnings':'weather_warnings_eligible',
                'Roshydromet warm-season regional temperature':'warm_temperature_claims_eligible',
                'Roshydromet warm-season regional precipitation':'warm_precipitation_claims_eligible'}[r.source]
        q=news_pairs[news_pairs.horizon==r.horizon]
        covered=q[q[source]>0]
        assert len(q)==r.evaluated_pairs and len(covered)==r.covered_pairs
        assert covered.target.nunique()==r.covered_target_months
        assert covered.region_code.nunique()==r.covered_regions
        assert int(q[source].sum())==r.eligible_items_total_over_pairs
    checks.append('news coverage uses exact matched as-of forecast pairs, month-end availability, region, forecast period, separate national context and hashed warm-season PDF claims')
    registry=pd.read_csv(ROOT/'results/news_claim_registry.csv',keep_default_na=False)
    assert len(registry)==58 and registry.claim_id.is_unique
    assert registry.event_kind.value_counts().to_dict()=={'warm_forecast':21,'announced_policy_decision':17,'heating_forecast':14,'dated_weather_warning':5,'reported_real_event':1}
    assert (pd.to_datetime(registry.available_from)>pd.to_datetime(registry.published_date)).all()
    assert registry.period_semantics.ne('').all() and registry.scope_limits.ne('').all()
    assert set(registry.source_file)<=set(x['path'] for x in manifest['sources'])
    assert registry[registry.event_kind=='reported_real_event'].allowed_role.eq('retrospective_context_only').all()
    assert registry[registry.event_kind=='announced_policy_decision'].geographic_scope.eq('national').all()
    for r in registry.itertuples():
        codes=json.loads(r.region_codes)
        assert len(codes)==len(json.loads(r.region_names))
        assert pd.Timestamp(r.event_start)<=pd.Timestamp(r.event_end)
    checks.append('unified news registry preserves source roles, geography, publication availability and distinct announcement/forecast/onset period semantics')
    from sberindex.forecasting.online_bias_correction import weighted_median,factors,base_predictions
    assert weighted_median([1,2,10],[6,3,1])==1
    history=base_predictions()
    bias_protocol=json.loads((ROOT/'results/bias_correction_protocol.json').read_text())
    val=pd.read_csv(ROOT/'results/bias_correction_validation.csv')
    assert val.loc[val.MAE.idxmin(),'candidate']==bias_protocol['selected']
    for origin,h in [('2024-06',1),('2024-09',3),('2024-06',6)]:
        ids=np.sort(history.territory_id.unique())
        altered=history.copy();altered.loc[altered.target>origin,'actual']*=100
        for candidate in bias_protocol['candidates']:
            a,meta=factors(history,origin,h,ids,candidate);b,other=factors(altered,origin,h,ids,candidate)
            np.testing.assert_array_equal(a,b);assert meta==other
            assert np.all((a>=.85)&(a<=1.15))
            if h==6:np.testing.assert_array_equal(a,np.ones(len(ids)))
    chosen=pd.read_parquet(ROOT/'results/bias_correction_predictions.parquet')
    original=p[p.model=='blend_75']
    keys=['territory_id','origin','target','horizon']
    merged=chosen.merge(original,on=keys,validate='one_to_one',suffixes=('_new','_base'))
    assert len(merged)==len(original)
    if bias_protocol['selected']=='none':np.testing.assert_allclose(merged.predicted_new,merged.predicted_base)
    checks.append('MAE multiplier weighted median analytic example; bias correction ignores future errors and requires two completed dates per horizon; early selection and original baseline reconcile')
    from sberindex.forecasting.deseasonal_hgb import forecast as dh_forecast
    class ZeroChange:
        def predict(self,X):return np.zeros(len(X))
    toy=np.arange(1,73,dtype=float).reshape(3,24)+100
    profile=np.arange(1,13,dtype=float)
    expected=toy[:,17]*profile[20%12]/profile[17%12]
    np.testing.assert_allclose(dh_forecast(ZeroChange(),profile,toy,17,3),expected)
    modified=toy.copy();modified[:,18:]*=100
    np.testing.assert_array_equal(dh_forecast(ZeroChange(),profile,toy,17,3),dh_forecast(ZeroChange(),profile,modified,17,3))
    dh=json.loads((ROOT/'results/deseasonal_hgb_protocol.json').read_text())
    dv=pd.read_csv(ROOT/'results/deseasonal_hgb_validation.csv')
    assert dv.loc[dv.MAE.idxmin(),'hgb_weight']==dh['selected_hgb_weight']
    dc=pd.read_parquet(ROOT/'results/deseasonal_hgb_candidates.parquet')
    m=dc[dc.hgb_weight==0].merge(original,on=keys,validate='one_to_one',suffixes=('_new','_base'))
    assert len(m)==len(original);np.testing.assert_allclose(m.predicted_new,m.predicted_base)
    checks.append('deseasonal recursion reproduces analytic seasonal ratio and ignores future actuals; early blend choice and zero-weight baseline reconcile')
    survival=json.loads((ROOT/'results/survivorship_audit.json').read_text())
    assert (survival['complete_2023_ids'],survival['complete_2023_and_2024_ids'],
            survival['excluded_by_future_completeness'])==(2075,2016,59)
    assert survival['selected_asof_weight']==.75
    sensitivity=pd.read_csv(ROOT/'results/survivorship_forecast_sensitivity.csv')
    reference=pd.read_csv(ROOT/'results/forecast_comparison.csv')
    for horizon in [1,3,6]:
        for audit_model,benchmark_model in [('seasonal_original','seasonal_pooled'),
                                            ('blend_original','blend_75')]:
            observed=sensitivity[(sensitivity.horizon==horizon)&
                                 (sensitivity.model==audit_model)].MAE.item()
            expected=reference[(reference.horizon==horizon)&
                               (reference.model==benchmark_model)].MAE.item()
            assert np.isclose(observed,expected)
    partial=pd.read_csv(ROOT/'results/survivorship_partial_ids.csv')
    assert partial.observed_pairs.tolist()==[42,24,3]
    checks.append('future-completeness cohort audit reconciles original scores, reselects weight on early targets, and identifies sparse observed pairs outside the balanced panel')
    asof_protocol=json.loads((ROOT/'results/asof_cohort_protocol.json').read_text())
    original_data=pd.read_parquet(ROOT/'data/consumption.parquet')
    asof_panel=(original_data[original_data.category=='Все категории']
                .pivot(index='territory_id',columns='date',values='value').sort_index())
    eligible=asof_panel.index[asof_panel.iloc[:,:12].notna().all(axis=1)].to_numpy()
    expected_ids=np.sort(np.random.default_rng(20260930).choice(eligible,256,replace=False))
    np.testing.assert_array_equal(expected_ids,asof_protocol['sample_ids'])
    ap=pd.read_parquet(ROOT/'results/asof_cohort_predictions.parquet')
    assert set(ap.territory_id)<=set(expected_ids)
    assert not ap.duplicated(['territory_id','origin','horizon','model']).any()
    for r in ap.itertuples():
        assert (pd.Period(r.origin,freq='M')+r.horizon).strftime('%Y-%m')==r.target
        assert np.isfinite(asof_panel.loc[r.territory_id,:r.origin]).all()
        assert asof_panel.loc[r.territory_id,r.target]==r.actual
    av=pd.read_csv(ROOT/'results/asof_cohort_validation.csv')
    asummary=json.loads((ROOT/'results/asof_cohort_summary.json').read_text())
    assert av.loc[av.MAE.idxmin(),'seasonal_weight']==asummary['selected_weight_from_early_2024']
    scored=pd.read_parquet(ROOT/'results/asof_cohort_scored.parquet')
    comp=pd.read_csv(ROOT/'results/asof_cohort_comparison.csv')
    for r in comp.itertuples():
        subset=scored[(scored.horizon==r.horizon)&(scored.model==r.model)]
        assert len(subset)==r.observations and np.isclose(subset.absolute_error.mean(),r.MAE)
    checks.append('as-of cohort IDs selected from 2023 only; each forecast uses complete observed history, scores only observed targets, and selects weight on early dates')
    from sberindex.forecasting.asof_regional_forecast import seasonal_predictions
    from sberindex.forecasting.regional_seasonality import CANDIDATES
    regional_panel = asof_panel.loc[eligible].sort_index()
    regional_geo = pd.read_csv(ROOT/'results/municipal_lookup.csv')
    regional_geo = regional_geo[regional_geo.year == 2023].set_index('territory_id')
    regional_regions = regional_geo.loc[regional_panel.index, 'region_code'].to_numpy()
    regional_base = ap[(ap.model == 'prophet') & ap.horizon.isin([1,3,6])].copy()
    regional_early = regional_base[(regional_base.horizon == 1) & regional_base.target.le('2024-06')]
    regional_val = pd.read_csv(ROOT/'results/asof_regional_validation.csv')
    regional_protocol = json.loads((ROOT/'results/asof_regional_protocol.json').read_text())
    assert len(regional_val) == regional_protocol['candidate_count'] == 25
    assert regional_protocol['eligible_ids'] == len(eligible) == 2075
    assert regional_protocol['geography_year'] == regional_protocol['profile_year'] == 2023
    for candidate in CANDIDATES:
        season = seasonal_predictions(regional_panel, regional_regions, regional_early, candidate)
        for row in regional_val[regional_val.candidate == candidate].itertuples():
            prediction = row.seasonal_weight*season + (1-row.seasonal_weight)*regional_early.predicted
            error = abs(regional_early.actual-prediction)
            assert np.isclose(row.MAE_date_balanced, error.groupby(regional_early.target).mean().mean())
    chosen = regional_val.loc[regional_val.MAE_date_balanced.idxmin()]
    assert chosen.candidate == regional_protocol['selected_candidate']
    assert chosen.seasonal_weight == regional_protocol['selected_seasonal_weight']
    regional_pred = pd.read_parquet(ROOT/'results/asof_regional_predictions.parquet')
    regional_keys = ['territory_id','origin','target','horizon']
    regional_late = regional_base[regional_base.origin.ge('2024-06')]
    assert set(map(tuple,regional_pred[regional_keys].to_numpy())) == set(map(tuple,regional_late[regional_keys].to_numpy()))
    matched = regional_pred.merge(regional_late, on=regional_keys, validate='one_to_one', suffixes=('_regional','_prophet'))
    season = seasonal_predictions(regional_panel,regional_regions,matched,chosen.candidate)
    np.testing.assert_allclose(matched.predicted_regional, chosen.seasonal_weight*season+(1-chosen.seasonal_weight)*matched.predicted_prophet)
    np.testing.assert_allclose(matched.actual_regional,matched.actual_prophet)
    changed = regional_panel.copy(); changed.loc[:, '2024-07':] *= 100
    june = regional_late[regional_late.origin == '2024-06']
    np.testing.assert_array_equal(seasonal_predictions(regional_panel,regional_regions,june,chosen.candidate),seasonal_predictions(changed,regional_regions,june,chosen.candidate))
    pooled = seasonal_predictions(regional_panel,regional_regions,regional_late,'pooled')
    saved_pooled = ap[(ap.model == 'seasonal_pooled') & ap.horizon.isin([1,3,6]) & ap.origin.ge('2024-06')]
    check_pooled = regional_late[regional_keys].assign(recomputed=pooled).merge(saved_pooled,on=regional_keys,validate='one_to_one')
    np.testing.assert_allclose(check_pooled.recomputed,check_pooled.predicted)
    comparison = pd.read_csv(ROOT/'results/asof_regional_comparison.csv')
    for row in comparison[comparison.model == 'regional_asof_selected'].itertuples():
        q = regional_pred[regional_pred.horizon == row.horizon]
        assert np.isclose(row.MAE_date_balanced,abs(q.actual-q.predicted).groupby(q.target).mean().mean())
        assert row.observations == len(q)
    checks.append('Regional mixtures use only the 2023 profile/cohort/geography; 25 early candidate errors and selection recompute; future changes leave June forecasts unchanged; scored pairs and pooled baseline match')
    monthly=pd.read_csv(ROOT/'results/asof_monthly_mae.csv')
    robustness=pd.read_csv(ROOT/'results/asof_month_robustness.csv')
    for r in monthly.itertuples():
        q=scored[(scored.horizon==r.horizon)&(scored.target==r.target)&(scored.model==r.model)]
        assert len(q)==r.municipalities and np.isclose(q.absolute_error.mean(),r.MAE)
    for r in robustness.itertuples():
        q=monthly[monthly.horizon==r.horizon].pivot(index='target',columns='model',values='MAE')
        delta=q[r.challenger]-q[r.benchmark]
        assert len(delta)==r.dates and np.isclose(delta.mean(),r.MAE_difference_rub)
        assert (delta<0).sum()==r.months_challenger_better
        assert (delta>0).sum()==r.months_challenger_worse
        jackknife=[delta.drop(target).mean() for target in delta.index]
        assert np.isclose(min(jackknife),r.leave_one_date_out_min_rub)
        assert np.isclose(max(jackknife),r.leave_one_date_out_max_rub)
    checks.append('as-of monthly MAE and leave-one-date-out sensitivity reconcile with matched individual forecasts')
    lag=pd.read_parquet(ROOT/'results/asof_reporting_delay_predictions.parquet')
    lag_summary=pd.read_csv(ROOT/'results/asof_reporting_delay_summary.csv')
    lag_protocol=json.loads((ROOT/'results/asof_reporting_delay_protocol.json').read_text())
    assert lag_protocol['delays_months']==[0,1,2]
    assert set(lag.territory_id)<=set(expected_ids)
    assert not lag.duplicated(['territory_id','target','model','reporting_delay_months']).any()
    assert lag.groupby(['territory_id','target','model']).reporting_delay_months.nunique().eq(3).all()
    assert lag.groupby(['territory_id','target']).actual.nunique().eq(1).all()
    assert set(lag.target)=={'2024-09','2024-10','2024-11','2024-12'}
    for r in lag.itertuples():
        assert r.horizon==1+r.reporting_delay_months
        assert pd.Period(r.origin,freq='M')+r.horizon==pd.Period(r.target,freq='M')
        assert pd.Period(r.decision_month,freq='M')+1==pd.Period(r.target,freq='M')
        assert pd.Period(r.origin,freq='M')+r.reporting_delay_months==pd.Period(r.decision_month,freq='M')
        assert np.isfinite(asof_panel.loc[r.territory_id,:r.origin]).all()
        assert asof_panel.loc[r.territory_id,r.target]==r.actual
        assert np.isclose(abs(r.actual-r.predicted),r.absolute_error)
    old=lag[lag.reporting_delay_months.isin([0,2]) & lag.model.isin(['prophet','seasonal_pooled'])]
    joined=old.merge(ap,on=['territory_id','origin','target','horizon','model'],validate='one_to_one',suffixes=('_lag','_asof'))
    assert len(joined)==len(old)
    np.testing.assert_allclose(joined.predicted_lag,joined.predicted_asof)
    eligible_profile=asof_panel.loc[eligible].iloc[:,:12].sum().to_numpy(float)
    mid=lag[(lag.reporting_delay_months==1)&(lag.model=='seasonal_pooled')]
    origin_values=np.array([asof_panel.loc[r.territory_id,r.origin] for r in mid.itertuples()],float)
    origin_month=pd.PeriodIndex(mid.origin,freq='M').month.to_numpy()-1
    target_month=pd.PeriodIndex(mid.target,freq='M').month.to_numpy()-1
    expected_season=origin_values*eligible_profile[target_month]/eligible_profile[origin_month]
    np.testing.assert_allclose(mid.predicted,expected_season)
    blend=lag[lag.model.isin(['prophet','seasonal_pooled','blend_75'])].pivot(
        index=['territory_id','target','reporting_delay_months'],columns='model',values='predicted')
    np.testing.assert_allclose(blend.blend_75,.75*blend.seasonal_pooled+.25*blend.prophet)
    for r in lag_summary.itertuples():
        q=lag[(lag.model==r.model)&(lag.reporting_delay_months==r.reporting_delay_months)]
        assert len(q)==r.observations and q.target.nunique()==r.dates
        assert np.isclose(q.groupby('target').absolute_error.mean().mean(),r.MAE_date_balanced)
        assert np.isclose(q.absolute_error.mean(),r.MAE_pooled)
    checks.append('as-of reporting lag 0/1/2 uses matched observed targets, correct effective horizons, historical inputs, and reuses frozen lag 0/2 forecasts')
    from sberindex.forecasting.asof_regional_delay import KEYS as delay_keys
    transferred = pd.read_parquet(ROOT/'results/asof_regional_delay_predictions.parquet')
    transfer_protocol = json.loads((ROOT/'results/asof_regional_delay_protocol.json').read_text())
    assert transfer_protocol['candidate'] == chosen.candidate
    assert transfer_protocol['seasonal_weight'] == chosen.seasonal_weight
    for model, group in transferred.groupby('model'):
        assert not group.duplicated(delay_keys).any()
        assert set(map(tuple,group[delay_keys].to_numpy())) == set(map(tuple,lag[lag.model == 'prophet'][delay_keys].to_numpy()))
        if model != 'regional_asof_transferred':
            original = lag[lag.model == model]
            merged = group.merge(original,on=delay_keys,validate='one_to_one',suffixes=('_new','_old'))
            np.testing.assert_array_equal(merged.predicted_new,merged.predicted_old)
            np.testing.assert_array_equal(merged.actual_new,merged.actual_old)
    new = transferred[transferred.model == 'regional_asof_transferred']
    base = lag[lag.model == 'prophet']
    merged = new.merge(base,on=delay_keys,validate='one_to_one',suffixes=('_regional','_prophet'))
    seasonal = seasonal_predictions(regional_panel,regional_regions,merged,chosen.candidate)
    np.testing.assert_allclose(merged.predicted_regional,chosen.seasonal_weight*seasonal+(1-chosen.seasonal_weight)*merged.predicted_prophet)
    september = merged[merged.target == '2024-09']
    changed = regional_panel.copy();changed.loc[:,'2024-09':] *= 100
    np.testing.assert_array_equal(seasonal_predictions(regional_panel,regional_regions,september,chosen.candidate),seasonal_predictions(changed,regional_regions,september,chosen.candidate))
    transfer_summary = pd.read_csv(ROOT/'results/asof_regional_delay_summary.csv')
    for row in transfer_summary.itertuples():
        group = transferred[(transferred.model == row.model)&(transferred.reporting_delay_months == row.reporting_delay_months)]
        assert np.isclose(row.MAE_date_balanced,abs(group.actual-group.predicted).groupby(group.target).mean().mean())
        assert row.observations == len(group) == 1004
    transfer_monthly = pd.read_csv(ROOT/'results/asof_regional_delay_monthly.csv')
    for row in pd.read_csv(ROOT/'results/asof_regional_delay_sensitivity.csv').itertuples():
        errors = transfer_monthly[transfer_monthly.reporting_delay_months == row.reporting_delay_months].pivot(index='target',columns='model',values='MAE')
        delta = errors.regional_asof_transferred-errors[row.benchmark]
        assert np.isclose(row.MAE_difference_rub,delta.mean())
        assert row.months_regional_better == int((delta<0).sum())
        leave_one_out = [delta.drop(month).mean() for month in delta.index]
        assert np.isclose(row.leave_one_date_out_min_rub,min(leave_one_out))
        assert np.isclose(row.leave_one_date_out_max_rub,max(leave_one_out))
    checks.append('Regional reporting-delay transfer preserves frozen parameters and all saved baseline pairs/predictions; lagged anchors ignore future facts; MAE and date sensitivity recompute')
    foundation=pd.read_parquet(ROOT/'results/asof_foundation_predictions.parquet')
    foundation_protocol=json.loads((ROOT/'results/asof_foundation_protocol.json').read_text())
    cfg=json.loads((ROOT/'config.json').read_text())
    assert foundation_protocol['chronos_bolt_revision']==cfg['chronos_revision']
    assert foundation_protocol['chronos_2_revision']==cfg['chronos2_revision']
    keys=['territory_id','origin','target','horizon']
    prophet_asof=ap[ap.model=='prophet'][keys+['actual']]
    for model in ['chronos_2','chronos_bolt_tiny']:
        part=foundation[foundation.model==model]
        joined=prophet_asof.merge(part,on=keys,validate='one_to_one',suffixes=('_prophet','_foundation'))
        assert len(joined)==len(prophet_asof)==len(part)
        np.testing.assert_allclose(joined.actual_prophet,joined.actual_foundation)
        np.testing.assert_array_equal((pd.PeriodIndex(joined.origin,freq='M')+
                                       joined.horizon.to_numpy()).astype(str),joined.target.to_numpy())
    checks.append('pinned Chronos models use the same 2023-known IDs, origin-target pairs and observed facts as the as-of Prophet comparison')
    from sberindex.detection.event_attribution_audit import paired_metrics, factors
    toy_original=np.zeros((3,6),dtype=bool)
    toy_changed=toy_original.copy();toy_changed[0,0]=True;toy_changed[1,1]=True;toy_changed[2,2]=True
    toy_stats=paired_metrics(toy_original,toy_changed,np.array([True,True,False]),0)
    assert toy_stats['new_at_onset']==.5 and toy_stats['new_by_second_active_month']==1
    assert toy_stats['new_later_without_onset']==.5 and toy_stats['new_any_control']==1
    assert factors('ramp',.2)[0]==1 and factors('pulse',.2)[1]==1
    event_runs=pd.read_csv(ROOT/'results/event_attribution_runs.csv')
    assert len(event_runs)==1440
    assert json.loads((ROOT/'results/event_attribution_audit.json').read_text())['max_prechange_signal_difference']==0
    step=event_runs[(event_runs['shape']=='step')&(event_runs.shift_pct==20)]
    scope=pd.read_csv(ROOT/'results/change_scope_repetitions.csv')
    scope=scope[(scope.coverage==.25)&(scope.shift_pct==20)]
    paired=step.merge(scope,on=['category','seed','method'],validate='one_to_one')
    assert len(paired)==len(step)==len(scope)
    for new,old in [('changed_any_treated','treated_alarm_rate'),
                    ('original_any_treated','treated_baseline_alarm_rate'),
                    ('changed_any_control','control_alarm_rate'),
                    ('original_any_control','control_baseline_alarm_rate')]:
        np.testing.assert_allclose(paired[new],paired[old],atol=1e-12)
    checks.append('paired alarm timing matches analytic onset example and reconciles with existing frozen-threshold scope audit')
    event=pd.read_csv(ROOT/'data/external/mchs_orsk_event.csv').iloc[0]
    case=pd.read_csv(ROOT/'results/real_event_orsk.csv')
    assert len(case)==24 and not case.duplicated(['category','month']).any()
    assert event.event_start=='2024-04-05' and event.first_report_date=='2024-04-06'
    assert event.annotation_role=='illustrative_context_not_spending_shift_ground_truth'
    assert case.loc[case.event_month,'month'].eq('2024-04').all()
    assert case.loc[~case.event_month,'month'].ne('2024-04').all()
    city=case[(case.category=='Все категории')&(case.month=='2024-04')].iloc[0]
    actual=raw[(raw.territory_id==1673)&(raw.category=='Все категории')].set_index('date').value
    assert city.spending_rub==actual['2024-04']==30327
    np.testing.assert_allclose(city.yoy_pct,100*(actual['2024-04']/actual['2023-04']-1))
    assert city.seasonal_forecast_rub==pd.read_parquet(ROOT/'results/rolling_predictions.parquet').query(
        "territory_id == 1673 and origin == '2024-03' and horizon == 1 and model == 'seasonal_pooled'").predicted.iloc[0]
    checks.append('externally dated Orsk illustration reconciles event month, municipal spend and pre-event seasonal forecast without treating the event as a spending-shift label')
    from sberindex.detection.asof_detector_audit import verify_artifacts as verify_asof_detectors
    assert verify_asof_detectors()
    checks.append('Detector cohort depends only on June-available history; missing windows remain unobserved and sequential states reset; ID-stable assignments and summary coverage reconcile')
    checks.append('Regularized detector noise scales use calibration only and identical quantile burden; frozen synthetic tradeoffs disclosed without winner selection')
    from sberindex.forecasting.calendar_audit import verify as verify_calendar
    assert verify_calendar()
    checks.append('Calendar HGB ablation uses identical observed pairs and fixed 2023 training; leap dates, future invariance, source hashes and existing 12m HGB reconcile')
    from sberindex.forecasting.foundation_seasonal_audit import verify_foundation_seasonal_artifacts
    assert all(verify_foundation_seasonal_artifacts().values())
    checks.append('Foundation seasonal normalization restores known seasonal factors, ignores future data, preserves fixed revisions/raw predictions and matches scored pairs and metrics')
    from sberindex.external.regional_news_experiment import verify as verify_regional_news
    assert verify_regional_news()['fit_rejected']
    checks.append('Regional news separates unknown/negative directions and publication/composition/observation dates; sparse-release fit rejected before error inspection; official source hashes match')
    from sberindex.forecasting.asof_distribution_audit import verify as verify_distribution
    assert verify_distribution()
    checks.append('Frozen as-of distribution diagnostics reconcile identical forecast keys, actual targets, 2023-only strata, paired wins, concentration and leave-date/region sensitivity without refit or selection')
    from sberindex.forecasting.foundation_delay_audit import verify as verify_foundation_delay
    assert verify_foundation_delay()
    checks.append('Foundation/HGB delay experiment preserves decision-origin-target arithmetic, past-only inference cohort, matched IDs across all models/lags, exact reused forecasts, fixed source/code hashes and recomputed paired metrics')
    from sberindex.forecasting.operational_early_audit import verify as verify_operational_early
    assert verify_operational_early()
    checks.append('Early operational h2 uses frozen 2023 HGB, past-only contexts without actual bridge substitution, transferred blend weight, strict paired facts, exact reused late forecasts and complete source/cache hashes; early/late sensitivity recomputes')
    from sberindex.detection.detector_delay_audit import verify as verify_detector_delay
    assert verify_detector_delay()
    checks.append('Detector calendar-delay experiment keeps fixed calibration and denominators, shifts alarms without backdating, distinguishes pending from missing sources, censors after December and exactly reconciles zero-lag third-month rates with the as-of experiment')
    from sberindex.detection.bocpd_audit import verify as verify_bocpd
    assert verify_bocpd()
    checks.append('Full BOCPD posterior with boundary before current observation; matched Feb-Jun thresholds, fixed prior/hazard, paired source denominators, complete hashes and synthetic calendar tables rebuild')
    from sberindex.external.real_event_registry import verify as verify_real_registry
    assert verify_real_registry()
    checks.append('Source-frozen official events reconcile registry/snapshot hashes, explicit 2024 ID/name/region matches and all seven municipal links including absent expense series; event dates are context, not shift truth')
    output = {"passed": len(checks), "checks": checks,
              "limits": "These checks validate implementation and chronology, not an unseen holdout or causal interpretation."}
    (ROOT / "results/verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
