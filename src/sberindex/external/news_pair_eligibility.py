"""Point-in-time news coverage on the exact as-of forecast evaluation pairs.

This is a coverage audit, not a claim that regional forecasts describe each
municipality or that eligible news improves predictive accuracy.
"""

import json
import hashlib

import pandas as pd

from sberindex.paths import ROOT

OUT = ROOT / "results"
PAIR_KEYS = ["territory_id", "origin", "target", "horizon"]


def main():
    scored = pd.read_parquet(OUT / "asof_cohort_scored.parquet")
    pairs = scored[PAIR_KEYS].drop_duplicates().copy()
    assert not pairs.duplicated(PAIR_KEYS).any()
    pairs["origin_year"] = pairs.origin.str[:4].astype(int)
    lookup = pd.read_csv(OUT / "municipal_lookup.csv")
    assert not lookup.duplicated(["territory_id", "year"]).any()
    pairs = pairs.merge(
        lookup[["territory_id", "year", "region_code"]],
        left_on=["territory_id", "origin_year"],
        right_on=["territory_id", "year"], validate="many_to_one",
    ).drop(columns=["year", "origin_year"])
    assert pairs.region_code.notna().all()
    pairs["origin_end"] = pd.PeriodIndex(pairs.origin, freq="M").to_timestamp(how="end").normalize()
    pairs["target_start"] = pd.PeriodIndex(pairs.target, freq="M").to_timestamp()
    pairs["target_end"] = pd.PeriodIndex(pairs.target, freq="M").to_timestamp(how="end").normalize()

    claims = pd.read_csv(ROOT / "data/external/seasonal_weather/claims.csv",
                         parse_dates=["available_from"])
    claims = claims[claims.comparison_baseline == "climatology_1991_2020"]
    matched_claims = pairs.merge(
        claims[["target", "region_code", "bulletin", "available_from"]],
        on=["target", "region_code"], how="left", validate="many_to_many",
    )
    matched_claims = matched_claims[
        matched_claims.available_from.notna()
        & (matched_claims.available_from <= matched_claims.origin_end)
    ]
    claim_counts = matched_claims.groupby(PAIR_KEYS).size().rename("seasonal_claims_eligible")

    warm_protocol = json.loads((ROOT / "data/external/warm_season/protocol.json").read_text())
    for source in warm_protocol["sources"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    warm = pd.read_csv(ROOT / "data/external/warm_season/claims.csv",
                       parse_dates=["published_date", "available_from"])
    assert set(warm.variable) == {"temperature", "precipitation"}
    assert set(warm.direction) == {-1, 1}
    assert set(warm.comparison_baseline) == {"source_climate_normal_period_unspecified"}
    assert set(warm.scope) == {"whole_named_region"}
    assert not warm.duplicated(["bulletin", "target", "region_code", "variable"]).any()
    assert ((warm.available_from - warm.published_date).dt.days == 1).all()
    for source in warm_protocol["sources"]:
        group = warm[warm.bulletin == source["bulletin"]]
        assert len(group) and set(group.published_date.dt.strftime("%Y-%m-%d")) == {source["published_date"]}
        assert set(group.available_from.dt.strftime("%Y-%m-%d")) == {source["available_from"]}
        assert group.page.between(3, 8).all()
    assert set(warm.region_code).issubset(set(lookup.region_code))
    matched_warm = pairs.merge(
        warm[["target", "region_code", "variable", "available_from"]],
        on=["target", "region_code"], how="left", validate="many_to_many",
    )
    matched_warm = matched_warm[
        matched_warm.available_from.notna()
        & (matched_warm.available_from <= matched_warm.origin_end)
    ]
    warm_counts = {
        variable: matched_warm[matched_warm.variable == variable].groupby(PAIR_KEYS).size().rename(
            f"warm_{variable}_claims_eligible")
        for variable in ("temperature", "precipitation")
    }

    warnings = pd.read_csv(ROOT / "data/external/weather_body_sample/warning_events.csv",
                           parse_dates=["available_from", "event_start", "event_end"])
    warning_regions = warnings.assign(region_code=warnings.region_codes.map(json.loads)).explode("region_code")
    warning_regions.region_code = warning_regions.region_code.astype(int)
    matched_warnings = pairs.merge(
        warning_regions[["region_code", "source_url", "available_from", "event_start", "event_end"]],
        on="region_code", how="left", validate="many_to_many",
    )
    matched_warnings = matched_warnings[
        matched_warnings.available_from.notna()
        & (matched_warnings.available_from <= matched_warnings.origin_end)
        & (matched_warnings.event_start <= matched_warnings.target_end)
        & (matched_warnings.event_end >= matched_warnings.target_start)
    ]
    warning_counts = matched_warnings.groupby(PAIR_KEYS).source_url.nunique().rename("weather_warnings_eligible")

    news = pd.read_csv(OUT / "news_corpus_features.csv")
    news = news[news.corpus == "all_decisions"][
        ["origin", "announced_rate_pct", "announcements_90d"]
    ]
    assert news.origin.is_unique
    pairs = pairs.merge(news, on="origin", validate="many_to_one")
    pairs = pairs.join(claim_counts, on=PAIR_KEYS).join(warning_counts, on=PAIR_KEYS)
    for counts in warm_counts.values():
        pairs = pairs.join(counts, on=PAIR_KEYS)
    for column in ("seasonal_claims_eligible", "weather_warnings_eligible",
                   "warm_temperature_claims_eligible", "warm_precipitation_claims_eligible"):
        pairs[column] = pairs[column].fillna(0).astype(int)
    assert (pairs.seasonal_claims_eligible >= 0).all()
    assert (pairs.weather_warnings_eligible >= 0).all()
    assert (pairs.announcements_90d >= 0).all()
    pairs.sort_values(PAIR_KEYS).to_csv(OUT / "news_pair_eligibility.csv", index=False)

    rows = []
    for horizon, group in pairs.groupby("horizon"):
        for source, column in (
            ("CBR national decisions", "announcements_90d"),
            ("Roshydromet seasonal regional claims", "seasonal_claims_eligible"),
            ("Roshydromet dated regional warnings", "weather_warnings_eligible"),
            ("Roshydromet warm-season regional temperature", "warm_temperature_claims_eligible"),
            ("Roshydromet warm-season regional precipitation", "warm_precipitation_claims_eligible"),
        ):
            covered = group[group[column] > 0]
            rows.append({
                "horizon": int(horizon), "source": source,
                "evaluated_pairs": len(group), "covered_pairs": len(covered),
                "covered_target_months": covered.target.nunique(),
                "covered_regions": covered.region_code.nunique(),
                "eligible_items_total_over_pairs": int(group[column].sum()),
            })
    pd.DataFrame(rows).to_csv(OUT / "news_pair_coverage.csv", index=False)
    assert len(pairs) == len(scored[PAIR_KEYS].drop_duplicates())


if __name__ == "__main__":
    main()
