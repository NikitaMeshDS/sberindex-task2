"""Build a reviewable registry of structured source claims, preserving roles."""

import json

import pandas as pd

from sberindex.paths import ROOT


def main():
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    regions = lookup[["region_code", "region_name"]].drop_duplicates().set_index("region_code").region_name.to_dict()
    rows = []

    def add(claim_id, source_kind, event_kind, published_date, available_from,
            event_start, event_end, region_codes, variable, value, unit,
            source_url, source_file, allowed_role, **extra):
        rows.append(dict(claim_id=claim_id, source_kind=source_kind, event_kind=event_kind,
            published_date=published_date, available_from=available_from,
            availability_basis="assumed_next_day_date_only", event_start=event_start,
            event_end=event_end, region_codes=json.dumps(region_codes),
            region_names=json.dumps([regions[c] for c in region_codes], ensure_ascii=False),
            variable=variable, value=value, unit=unit, source_url=source_url,
            source_file=source_file, allowed_role=allowed_role, release_url="",
            source_page="", territory_id="", municipality="", comparison_baseline="",
            review_status="single_assistant_review_not_independent",
            period_semantics="", geographic_scope="", scope_limits="", **extra))

    cbr_path = "data/external/cbr_rate_decisions_2023_2024.csv"
    for r in pd.read_csv(ROOT / cbr_path).itertuples():
        add(f"cbr_{r.published_date}", "CBR", "announced_policy_decision",
            r.published_date, r.available_from, r.published_date, r.published_date, [],
            "announced_key_rate", r.rate_after_pct, "percent", r.source_url, cbr_path,
            "candidate_asof_covariate")
        rows[-1].update(period_semantics="announcement_day_not_effective_date",
                        geographic_scope="national", scope_limits="One shared national signal; not municipality-specific exposure.")

    for folder, kind in [("seasonal_weather", "heating_forecast"), ("warm_season", "warm_forecast")]:
        protocol = json.loads((ROOT / f"data/external/{folder}/protocol.json").read_text())
        sources = {s["bulletin"]: s for s in protocol["sources"]}
        path = f"data/external/{folder}/claims.csv"
        claims = pd.read_csv(ROOT / path)
        for r in claims.itertuples():
            variable = getattr(r, "variable", "temperature")
            direction = getattr(r, "direction", getattr(r, "temperature_direction", None))
            target = pd.Period(r.target, freq="M")
            source = sources[r.bulletin]
            add(f"{r.bulletin}_{r.target}_{r.region_code}_{variable}_{r.comparison_baseline}",
                "Roshydromet", kind, r.published_date, r.available_from,
                target.start_time.strftime("%Y-%m-%d"), target.end_time.strftime("%Y-%m-%d"),
                [int(r.region_code)], variable, direction, "qualitative_direction", source["source_url"],
                path, "candidate_asof_covariate")
            rows[-1].update(release_url=source["release_url"], source_page=int(r.page),
                            comparison_baseline=r.comparison_baseline,
                            period_semantics="forecast_target_month",
                            geographic_scope="whole_named_region_qualitative_context",
                            scope_limits="Regional direction is not measured exposure of every municipality; historical attachment vintage unverified.")

    path = "data/external/weather_body_sample/warning_events.csv"
    for r in pd.read_csv(ROOT / path).itertuples():
        add("warning_" + r.source_url.rstrip("/").split("/")[-1], "Roshydromet",
            "dated_weather_warning", r.published_date, r.available_from,
            r.event_start, r.event_end, json.loads(r.region_codes), r.hazard,
            "", "hazard_presence", r.source_url, path, "candidate_asof_covariate")
        rows[-1].update(period_semantics="stated_warning_interval",
                        geographic_scope="union_of_named_regions",
                        scope_limits=r.scope)

    path = "data/external/mchs_orsk_event.csv"
    for r in pd.read_csv(ROOT / path).itertuples():
        region = lookup[(lookup.territory_id == r.territory_id) & (lookup.year == 2024)].region_code.item()
        available = (pd.Timestamp(r.first_report_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        add(r.event_id, "MChS", "reported_real_event", r.first_report_date, available,
            r.event_start, r.event_start, [int(region)], "flood_onset", "", "reported_event",
            r.first_report_url, path, "retrospective_context_only")
        rows[-1].update(territory_id=int(r.territory_id), municipality=r.municipality,
                        release_url=r.event_date_source_url,
                        period_semantics="onset_day_not_event_duration",
                        geographic_scope="named_municipality",
                        scope_limits="Retrospectively selected illustration; event duration and spending-shift label unknown.")

    frame = pd.DataFrame(rows).sort_values(["published_date", "claim_id"])
    assert len(frame) == 58 and frame.claim_id.is_unique
    assert (pd.to_datetime(frame.available_from) > pd.to_datetime(frame.published_date)).all()
    assert (pd.to_datetime(frame.event_end) >= pd.to_datetime(frame.event_start)).all()
    assert set(frame[frame.event_kind == "reported_real_event"].allowed_role) == {"retrospective_context_only"}
    frame.to_csv(ROOT / "results/news_claim_registry.csv", index=False)
    frame.groupby(["source_kind", "event_kind", "allowed_role"], as_index=False).size().rename(
        columns={"size": "claims"}).to_csv(ROOT / "results/news_registry_summary.csv", index=False)


if __name__ == "__main__":
    main()
