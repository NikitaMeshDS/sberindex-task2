"""Separate retrospective regional-news audit; never rewrite prior results.

Publication + 1 day is an explicit sensitivity assumption, not a historical
vintage assertion. Absence from an event-selected corpus is unknown coverage.
"""

import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sberindex.paths import ROOT
from sberindex.external.news_corpus_audit import select_training_cohort

DATA = ROOT / "data/external/regional_news_review"
OUT = ROOT / "reports/regional_news_review"

# source_number, publication, region, explicit dictionary names, episode,
# event_start (only independently stated onset), observed/forecast day, role,
# evidence: original line references in preserved web-tool extraction.
ANNOTATIONS = [
    (
        1,
        "2024-04-06",
        56,
        ["Орск"],
        "orsk_apr2024",
        "2024-04-05",
        "2024-04-06",
        "observed",
        "L207,L216-L219; onset independently source25 L238",
    ),
    (
        2,
        "2024-08-07",
        46,
        [],
        "kursk_emergency_aug2024",
        "2024-08-07",
        "2024-08-07",
        "administrative",
        "L214,L223: emergency regime; not military onset",
    ),
    (
        3,
        "2024-12-17",
        23,
        ["Анапа"],
        "anapa_spill_dec2024",
        "",
        "2024-12-17",
        "observed",
        "L62,L64-L68: cleanup now; first beach arrival not proved by this page",
    ),
    (
        5,
        "2024-08-11",
        46,
        [],
        "kursk_evacuation_aug2024",
        "",
        "2024-08-10",
        "observed",
        "L240,L253-L256: departures during preceding day; districts unnamed",
    ),
    (
        6,
        "2024-12-26",
        23,
        ["Анапа", "Темрюкский"],
        "anapa_spill_dec2024",
        "",
        "2024-12-26",
        "observed",
        "L233,L238-L244: cleanup in Anapa/Veselovka; municipality subset, no spill onset",
    ),
    (
        7,
        "2023-08-30",
        25,
        ["Хасанский", "Дальнереченский"],
        "primorye_flood_aug2023",
        "2023-08-29",
        "2023-08-30",
        "observed",
        "L233,L241: Aug29-30, explicitly Дальнереченский urban district",
    ),
    (
        8,
        "2023-08-31",
        25,
        ["Хасанский", "Уссурийский"],
        "primorye_flood_aug2023",
        "",
        "2023-08-31",
        "observed",
        "L233,L242-L250: named affected places; ambiguous Дальнереченский omitted from exposure",
    ),
    (
        9,
        "2023-04-26",
        66,
        ["Сосьвинский"],
        "sosva_fire_apr2023",
        "2023-04-25",
        "2023-04-26",
        "observed",
        "L233,L247; Apr25 onset independently source23",
    ),
    (
        11,
        "2023-05-16",
        66,
        ["Сосьвинский"],
        "sosva_fire_apr2023",
        "2023-04-25",
        "2023-05-16",
        "recovery",
        "L233,L245-L248: reconstruction; Taezhny geography not mapped",
    ),
    (
        12,
        "2023-06-15",
        23,
        ["Лабинский", "Отрадненский", "Кавказский", "Сочи"],
        "krasnodar_flood_jun2023",
        "2023-06-13",
        "2023-06-15",
        "observed",
        "L224,L235-L278: June13 districts, June14 Sochi; start is episode earliest report day",
    ),
    (
        13,
        "2024-04-11",
        56,
        ["Орск", "Оренбург"],
        "orenburg_flood_apr2024",
        "",
        "2024-04-11",
        "observed",
        "L233,L243-L247: observed levels and preventive work; no citywide impact assumed",
    ),
    (
        15,
        "2024-08-22",
        46,
        [],
        "kursk_evacuation_aug2024",
        "",
        "2024-08-22",
        "observed",
        "L233,L242-L250: 9 unnamed municipalities; no mapping to all municipalities",
    ),
    (
        15,
        "2024-08-22",
        31,
        [],
        "belgorod_assistance_aug2024",
        "",
        "2024-08-22",
        "observed",
        "L250: payments in Belgorod region, municipality recipients unnamed",
    ),
    (
        16,
        "2023-05-02",
        54,
        ["Новосибирск"],
        "novosibirsk_fires_apr2023",
        "2023-04-24",
        "2023-04-30",
        "observed",
        "L221,L229-L233: weekly aggregate Apr24-30; city incidents explicitly named",
    ),
    (
        17,
        "2023-05-08",
        45,
        [],
        "kurgan_fires_may2023",
        "",
        "2023-05-08",
        "observed",
        "L233,L242-L249: active fire counts region only; onset not stated",
    ),
    (
        18,
        "2024-04-10",
        56,
        ["Орск", "Оренбург"],
        "orenburg_flood_apr2024",
        "",
        "2024-04-10",
        "observed",
        "L233 and body: Orsk utility interruption and Orenburg protective works",
    ),
    (
        20,
        "2024-04-24",
        56,
        ["Орск", "Оренбург", "Илекский"],
        "orenburg_flood_apr2024",
        "",
        "2024-04-24",
        "observed",
        "L233,L243-L244: named cities/settlement; Ilek settlement manually assigned to Iлекский district",
    ),
    (
        20,
        "2024-04-24",
        45,
        ["Курган"],
        "kurgan_flood_apr2024",
        "",
        "2024-04-24",
        "observed",
        "L245 onwards: actual Kurgan levels/impact; no physical onset",
    ),
    (
        20,
        "2024-04-24",
        72,
        ["Абатский"],
        "tyumen_flood_apr2024",
        "",
        "2024-04-24",
        "warning",
        "body explicitly approaching Abatskoye; warning, not inundation label",
    ),
    (
        21,
        "2024-05-12",
        31,
        ["Белгород"],
        "belgorod_collapse_may2024",
        "2024-05-12",
        "2024-05-12",
        "observed",
        "L23 dated publication; body current rescue after collapse on Shchorsa street",
    ),
    (
        23,
        "2023-04-27",
        66,
        ["Сосьвинский"],
        "sosva_fire_apr2023",
        "2023-04-25",
        "2023-04-27",
        "observed",
        "L23 publication, body explicitly Apr25 start and Apr27 extinguished",
    ),
    (
        24,
        "2024-08-15",
        31,
        [],
        "belgorod_emergency_aug2024",
        "2024-08-15",
        "2024-08-15",
        "administrative",
        "L233 and body: federal emergency decision, not military onset",
    ),
    (
        25,
        "2024-04-15",
        56,
        ["Орск"],
        "orsk_apr2024",
        "2024-04-05",
        "2024-04-12",
        "recovery",
        "L222 publication; L229 visit Apr12; L238 onset Apr5 explicitly",
    ),
    (
        26,
        "2023-08-28",
        25,
        ["Хасанский", "Уссурийский"],
        "primorye_flood_aug2023",
        "",
        "2023-08-29",
        "warning",
        "L218 publication; L228-L235 warning for Aug29-31, not proven realized event",
    ),
]

# Same-name district vs urban district: retain all candidates and reason.
CHOICES = {(25, "Дальнереченский"): (780, "source7 explicitly says городской округ")}


def source_text(number):
    return json.loads((DATA / f"source_{number}.web.json").read_text())


def build_registry():
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    rows, mapping_rows = [], []
    for (
        number,
        published,
        region,
        names,
        episode,
        start,
        observation,
        role,
        evidence,
    ) in ANNOTATIONS:
        text = source_text(number)
        if "Failed to fetch" in text or "not accessible via this tool" in text:
            raise ValueError(f"Failed source cannot be evidence: {number}")
        url = re.search(r'Source: open\(\{"ref_id":"([^\"]+)', text).group(1)
        title = re.search(r"L\d+: # (.*)", text).group(1)
        year = pd.Timestamp(published).year
        ids = []
        for name in names:
            candidates = lookup[
                (lookup.year == year)
                & (lookup.region_code == region)
                & (lookup.municipal_district_name_short == name)
            ]
            candidate_ids = candidates.territory_id.astype(int).tolist()
            if len(candidate_ids) == 1:
                chosen, reason = (
                    candidate_ids[0],
                    "unique exact region/year dictionary alias",
                )
            elif (region, name) in CHOICES and CHOICES[region, name][
                0
            ] in candidate_ids:
                chosen, reason = CHOICES[region, name]
            else:
                chosen, reason = None, "unresolved; exclude municipal exposure"
            if chosen is not None:
                ids.append(chosen)
            mapping_rows.append(
                dict(
                    source_number=number,
                    region_code=region,
                    year=year,
                    alias=name,
                    candidates=json.dumps(candidate_ids),
                    territory_id=chosen,
                    status="accepted" if chosen is not None else "unresolved",
                    reason=reason,
                )
            )
        rows.append(
            dict(
                claim_id=f"rnr_{number}_{region}",
                episode_id=episode,
                title=title,
                published_date=published,
                event_start=start,
                observation_date=observation,
                event_end="",
                event_time_precision="day" if start else "onset_unknown",
                available_from=(
                    pd.Timestamp(published) + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d"),
                availability_status="assumed_publication_plus_one_day",
                historical_vintage_verified=False,
                region_code=region,
                municipality_aliases=json.dumps(names, ensure_ascii=False),
                territory_ids=json.dumps(ids),
                exposure_scope="named_local_subset" if ids else "region_context_only",
                event_role=role,
                source_url=url,
                snapshot_path=f"data/external/regional_news_review/source_{number}.web.json",
                evidence_note=evidence,
                allowed_role="external_context_not_spending_shift_ground_truth",
                source_coverage="unknown_event_selected",
                captured_at_utc="2026-10-06; clock time not recorded by web extraction",
            )
        )
    # Reuse six already inspected official sources; retain separate provenance.
    old = pd.read_csv(ROOT / "data/external/real_event_registry/registry.csv").fillna(
        ""
    )
    aliases = {
        "Обь": "Обь",
        "Новороссийск": "Новороссийск",
        "Карабашский городской округ": "Карабашский",
        "Сегежский муниципальный округ": "Сегежский",
        "Новочеркасск": "Новочеркасск",
        "Октябрьский район": "Октябрьский",
        "Чебоксары": "Чебоксары",
    }
    for event in old.itertuples():
        region_candidates = lookup[
            (lookup.year == 2024) & (lookup.region_name == event.region)
        ]
        if region_candidates.empty and event.region == "Чувашская Республика":
            region_candidates = lookup[
                (lookup.year == 2024) & (lookup.region_code == 21)
            ]
        region = int(region_candidates.region_code.iloc[0])
        ids = []
        for mention in event.municipality_names.split(";"):
            name = aliases[mention]
            candidates = region_candidates[
                region_candidates.municipal_district_name_short == name
            ]
            candidate_ids = candidates.territory_id.astype(int).tolist()
            chosen = candidate_ids[0] if len(candidate_ids) == 1 else None
            if chosen is not None:
                ids.append(chosen)
            mapping_rows.append(
                dict(
                    source_number=event.event_id,
                    region_code=region,
                    year=2024,
                    alias=name,
                    candidates=json.dumps(candidate_ids),
                    territory_id=chosen,
                    status="accepted" if chosen is not None else "unresolved",
                    reason="reused explicit existing official annotation",
                )
            )
        rows.append(
            dict(
                claim_id="rnr_" + event.event_id,
                episode_id=event.event_id,
                title=event.title,
                published_date=event.published_at,
                event_start=event.event_start,
                observation_date=event.event_start,
                event_end=event.event_end,
                event_time_precision=event.event_time_precision,
                available_from=(
                    pd.Timestamp(event.published_at) + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d"),
                availability_status="assumed_publication_plus_one_day",
                historical_vintage_verified=False,
                region_code=region,
                municipality_aliases=event.municipality_names,
                territory_ids=json.dumps(ids),
                exposure_scope="named_local_subset",
                event_role="observed",
                source_url=event.source_url,
                snapshot_path=event.snapshot_path,
                evidence_note=event.evidence_note + " " + event.limitations,
                allowed_role="external_context_not_spending_shift_ground_truth",
                source_coverage="unknown_event_selected",
                captured_at_utc="see existing capture_manifest.json",
            )
        )
    registry = pd.DataFrame(rows).sort_values(["published_date", "claim_id"])
    registry.to_csv(DATA / "registry.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(OUT / "geography_audit.csv", index=False)
    # Explicitly preserve ambiguity that is deliberately absent from annotations.
    pd.DataFrame(
        [
            dict(
                source_number=8,
                mention="Дальнереченском муниципальных образованиях",
                candidates="[780,788]",
                decision="exclude_ambiguous_place",
                reason="urban and municipal district share name",
            ),
            dict(
                source_number=11,
                mention="Таежный",
                candidates="",
                decision="exclude_not_resolved",
                reason="no parent municipality stated in selected evidence",
            ),
            dict(
                source_number=5,
                mention="приграничные районы",
                candidates="",
                decision="region_context_only",
                reason="no named municipal exposure",
            ),
            dict(
                source_number=15,
                mention="9 муниципальных образований",
                candidates="",
                decision="region_context_only",
                reason="nine districts not named; cannot spread evacuation to every municipality",
            ),
        ]
    ).to_csv(OUT / "unresolved_mentions.csv", index=False)
    manifest = []
    for path in sorted(DATA.glob("source_*.web.json")):
        body = path.read_bytes()
        text = json.loads(body)
        manifest.append(
            dict(
                path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(body).hexdigest(),
                bytes=len(body),
                capture_type="web_tool_text_extraction_not_raw_html",
                source_url=(
                    re.search(r'Source: open\(\{"ref_id":"([^\"]+)', text).group(1)
                ),
                capture_status="failed_excluded"
                if any(
                    s in text
                    for s in [
                        "Failed to fetch",
                        "not accessible via this tool",
                        "Internal Error",
                    ]
                )
                else "readable",
                retrieval_session_date="2026-10-06",
                historical_vintage=False,
            )
        )
    for path in sorted(set(registry.snapshot_path) - set(m["path"] for m in manifest)):
        p = ROOT / path
        manifest.append(
            dict(
                path=path,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                bytes=p.stat().st_size,
                capture_type="reused_official_web_extraction",
                capture_status="readable",
                retrieval_session_date="see prior manifest",
                historical_vintage=False,
            )
        )
    (DATA / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    exposures = []
    for row in registry.itertuples():
        for tid in json.loads(row.territory_ids):
            exposures.append(
                dict(
                    claim_id=row.claim_id,
                    region_code=row.region_code,
                    territory_id=tid,
                    exposure_scope="named_local_subset_not_whole_municipality",
                    observation_date=row.observation_date,
                    local_onset_date="2023-06-14"
                    if row.claim_id == "rnr_12_23" and tid == 2812
                    else "",
                    local_onset_semantics="dispatch_report_day_only"
                    if row.claim_id == "rnr_12_23" and tid == 2812
                    else "unknown_unless_specific_evidence_note",
                    event_start_semantics="episode_earliest_stated_day_not_all_local_onsets",
                    event_role=row.event_role,
                    source_url=row.source_url,
                    spending_shift_ground_truth=False,
                )
            )
    pd.DataFrame(exposures).to_csv(OUT / "claim_exposures.csv", index=False)
    return registry


def asof_features(registry, region, territory_id, origin, strict=False):
    """Only past available records; no records means unknown counts, never 0.

    Regional context and explicit municipality exposure are different channels.
    Source coverage is always unknown: an event-selected search did not enumerate
    all publications. Zero evidence flags mean no selected evidence, not no shock.
    """
    end = pd.Timestamp(origin) + pd.offsets.MonthEnd(0)
    available = pd.to_datetime(registry.available_from)
    use = registry[
        (available <= end)
        & (available > end - pd.Timedelta(days=90))
        & (registry.region_code == region)
    ]
    if strict:
        use = use[use.historical_vintage_verified.astype(str).str.lower().eq("true")]
    local = use.loc[
        use.territory_ids.map(lambda x: territory_id in json.loads(x)).astype(bool)
    ]
    r_count, l_count = use.episode_id.nunique(), local.episode_id.nunique()
    return np.array(
        [
            r_count if r_count else np.nan,
            float(bool(r_count)),
            l_count if l_count else np.nan,
            float(bool(l_count)),
        ]
    )


def run_ablation(registry):
    protocol = json.loads((DATA / "protocol.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = select_training_cohort(
        raw[raw.category == "Все категории"].pivot(
            index="territory_id", columns="date", values="value"
        )
    ).reindex(columns=[f"{y}-{m:02d}" for y in [2023, 2024] for m in range(1, 13)])
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    values = panel.to_numpy(float)
    logs = np.log(values)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    blocks, coverage, feature_rows = {}, [], []
    for i in [*range(2, 11), *range(12, 23)]:
        region_map = (
            lookup[lookup.year == months[i].year].set_index("territory_id").region_code
        )
        regions = region_map.reindex(panel.index).fillna(-1).astype(int)
        base = np.column_stack(
            [
                logs[:, i],
                logs[:, i] - logs[:, i - 1],
                logs[:, i - 1] - logs[:, i - 2],
                np.full(len(panel), np.sin(2 * np.pi * ((i + 1) % 12) / 12)),
                np.full(len(panel), np.cos(2 * np.pi * ((i + 1) % 12) / 12)),
            ]
        )
        # Batch join the tiny registry once per origin. Never inspect expenses.
        end = months[i] + pd.offsets.MonthEnd(0)
        available = pd.to_datetime(registry.available_from)
        known = registry[(available <= end) & (available > end - pd.Timedelta(days=90))]
        region_episodes, local_episodes = {}, {}
        for row in known.itertuples():
            region_episodes.setdefault(row.region_code, set()).add(row.episode_id)
            for tid in json.loads(row.territory_ids):
                local_episodes.setdefault((row.region_code, tid), set()).add(
                    row.episode_id
                )
        news = np.array(
            [
                [
                    len(region_episodes[region])
                    if region in region_episodes
                    else np.nan,
                    float(region in region_episodes),
                    len(local_episodes[region, tid])
                    if (region, tid) in local_episodes
                    else np.nan,
                    float((region, tid) in local_episodes),
                ]
                for tid, region in zip(panel.index, regions)
            ]
        )
        blocks[i] = (base, np.column_stack([base, news]))
        observed = np.isfinite(logs[:, i - 2 : i + 2]).all(axis=1)
        coverage.append(
            dict(
                stage="training" if i < 12 else "evaluation",
                origin=str(months[i].to_period("M")),
                target=str(months[i + 1].to_period("M")),
                eligible_ids=len(panel),
                scored_pairs=int(observed.sum()),
                selected_region_evidence_pairs=int(
                    (observed & (news[:, 1] == 1)).sum()
                ),
                selected_local_evidence_pairs=int((observed & (news[:, 3] == 1)).sum()),
                unknown_source_coverage_pairs=int(observed.sum()),
                missing_history_or_target=int((~observed).sum()),
            )
        )
        feature_rows.extend(
            dict(
                territory_id=int(tid),
                region_code=int(region),
                origin=str(months[i].to_period("M")),
                regional_episodes=vec[0],
                selected_region_evidence=int(vec[1]),
                local_episodes=vec[2],
                selected_local_evidence=int(vec[3]),
                source_coverage_known=False,
            )
            for tid, region, vec in zip(panel.index, regions, news)
        )
    training = np.concatenate([logs[:, i + 1] - logs[:, i] for i in range(2, 11)])
    rows = []
    for arm, column in [("without_news", 0), ("with_selected_regional_news", 1)]:
        model = HistGradientBoostingRegressor(**protocol["model"])
        model.fit(np.concatenate([blocks[i][column] for i in range(2, 11)]), training)
        for i in range(12, 23):
            use = np.isfinite(logs[:, i - 2 : i + 2]).all(axis=1)
            prediction = np.exp(
                logs[use, i] + np.clip(model.predict(blocks[i][column][use]), -0.5, 0.5)
            )
            for tid, actual, predicted in zip(
                panel.index[use], values[use, i + 1], prediction
            ):
                rows.append(
                    dict(
                        territory_id=int(tid),
                        origin=str(months[i].to_period("M")),
                        target=str(months[i + 1].to_period("M")),
                        model=arm,
                        actual=float(actual),
                        predicted=float(predicted),
                    )
                )
    predictions = pd.DataFrame(rows)
    predictions["absolute_error"] = abs(predictions.actual - predictions.predicted)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    predictions["period"] = np.where(
        predictions.target < "2024-07", "early_feb_jun", "late_jul_dec"
    )
    summary = []
    for period in ["all_feb_dec", "early_feb_jun", "late_jul_dec"]:
        group = (
            predictions
            if period == "all_feb_dec"
            else predictions[predictions.period == period]
        )
        for arm, part in group.groupby("model"):
            summary.append(
                dict(
                    period=period,
                    model=arm,
                    MAE=float(part.absolute_error.mean()),
                    WAPE_pct=float(100 * part.absolute_error.sum() / part.actual.sum()),
                    pairs=len(part),
                    dates=part.target.nunique(),
                    municipalities=part.territory_id.nunique(),
                )
            )
    pd.DataFrame(summary).to_csv(OUT / "ablation_summary.csv", index=False)
    predictions.groupby(
        ["target", "model"], as_index=False
    ).absolute_error.mean().rename(columns={"absolute_error": "MAE"}).to_csv(
        OUT / "ablation_monthly.csv", index=False
    )
    pd.DataFrame(coverage).to_csv(OUT / "pair_coverage.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUT / "asof_features.csv", index=False)
    audit = dict(
        cohort_ids=len(panel),
        training_rows=len(training),
        training_origins=9,
        evaluation_origins=11,
        protocol_sha256=hashlib.sha256(
            (DATA / "protocol.json").read_bytes()
        ).hexdigest(),
        registry_sha256=hashlib.sha256(
            (DATA / "registry.csv").read_bytes()
        ).hexdigest(),
        independent_holdout=False,
        source_selection_before_this_fit=True,
        strict_vintage_mode="no verified historical sources; incremental strict feature unavailable",
        availability_mode="assumed publication plus one day; sensitivity only",
        expenditure_release_lead_time="unknown: actual expenditure publication vintages not supplied",
        primary_model_changed=False,
        causal_inference=False,
    )
    (OUT / "ablation_audit.json").write_text(json.dumps(audit, indent=2))
    return pd.DataFrame(summary)


def write_provenance():
    import platform
    import sklearn

    artifacts = [
        DATA / "registry.csv",
        DATA / "protocol.json",
        DATA / "capture_manifest.json",
        ROOT / "results/municipal_lookup.csv",
        ROOT / "data/consumption.parquet",
        ROOT / "src/sberindex/external/regional_news_review.py",
        ROOT / "tests/test_regional_news_review.py",
        OUT / "ablation_audit.json",
        OUT / "REPORT.md",
        *sorted(OUT.glob("*.csv")),
        *sorted(OUT.glob("*.parquet")),
    ]
    registry = pd.read_csv(DATA / "registry.csv")
    captured = json.loads((DATA / "capture_manifest.json").read_text())
    record = dict(
        command="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m sberindex.external.regional_news_review",
        verification="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m unittest discover -s tests -p test_regional_news_review.py -v",
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        sklearn=sklearn.__version__,
        deterministic_seed=20260930,
        fit_threads=2,
        files=[
            dict(
                path=str(p.relative_to(ROOT)),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                bytes=p.stat().st_size,
            )
            for p in artifacts
        ],
        acquisition="web-tool text extraction preserved; no raw HTML claim; existing six official snapshots reused",
        discovery="event-selected manual official/reliable-source search; no source-universe enumeration",
        chronology_only_source="source_27.web.json: tanker distress Dec15; not a municipality exposure or forecasting feature",
        corpus=dict(
            claim_region_rows=len(registry),
            source_documents=int(registry.source_url.nunique()),
            regions=int(registry.region_code.nunique()),
            episodes=int(registry.episode_id.nunique()),
            explicit_municipal_ids=len(
                set(
                    tid
                    for encoded in registry.territory_ids
                    for tid in json.loads(encoded)
                )
            ),
            readable_captures=sum(x["capture_status"] == "readable" for x in captured),
            failed_excluded_captures=sum(
                x["capture_status"] == "failed_excluded" for x in captured
            ),
        ),
        frozen_dataset="existing 2023-2024 expense snapshot; no outcome relabeling",
        historical_vintage_verified=False,
        independent_holdout=False,
    )
    (OUT / "provenance.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2)
    )


def lead_time_table(registry):
    out = registry[
        [
            "claim_id",
            "episode_id",
            "published_date",
            "available_from",
            "event_start",
            "observation_date",
            "event_role",
            "exposure_scope",
            "source_url",
        ]
    ].copy()
    # Positive means news was available BEFORE independently stated external onset.
    out["lead_days_to_external_onset_assumed"] = (
        pd.to_datetime(out.event_start, errors="coerce")
        - pd.to_datetime(out.available_from)
    ).dt.days
    out["lead_days_to_spending_publication"] = np.nan
    out["spending_shift_ground_truth"] = False
    out["spending_release_timing_status"] = "unknown_no_historical_release_vintage"
    out.to_csv(OUT / "event_alignment.csv", index=False)
    return out


def main():
    registry = build_registry()
    lead_time_table(registry)
    summary = run_ablation(registry)
    write_provenance()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
