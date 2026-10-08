"""Retrospective field-level abstention replay; strict pilot outputs stay intact."""

import hashlib
import json
import re

import numpy as np
import pandas as pd

from sberindex.detection.event_conditioned_review import event_available
from sberindex.detection.peer_residual_review import alert_episodes
from sberindex.external.news_event_extraction_review import TYPES, normal, resolve_geo
from sberindex.paths import ROOT

DATA = ROOT / "data/external/news_event_extraction_review"
OUT = ROOT / "reports/news_event_extraction_review/field_validation_variant"


def quoted_geography(quote, region, year, lookup):
    """Dictionary NER only inside a grounded quote, with narrow Russian inflections."""
    geo = lookup[(lookup.region_code == region) & (lookup.year == year)]
    candidates = []
    for row in geo.itertuples():
        name = normal(row.municipal_district_name_short)
        if name.endswith("ский"):
            pattern = re.escape(name[:-2]) + r"(?:ий|ого|ому|им|ом)"
        elif name.endswith("ск"):
            pattern = re.escape(name) + r"(?:е|а|у|ом)?"
        else:
            pattern = re.escape(name)
        match = re.search(r"(?<!\w)" + pattern + r"(?!\w)", normal(quote))
        if match:
            candidates.append(
                (
                    int(row.territory_id),
                    row.municipal_district_name_short,
                    match.group(),
                )
            )
    if len(candidates) != 1:
        return None, None, None, "ambiguous_quote" if candidates else "no_quoted_place"
    tid, name, token = candidates[0]
    # A second same short name would have created ambiguity above; no reforms bridged.
    return tid, name, token, "quoted_dictionary_unique"


def validate_fields(value, body, publication, region, lookup):
    if not isinstance(value, dict):
        raise TypeError("missing_schema")
    if value.get("publication_date") != publication or str(value.get("region")) != str(
        region
    ):
        raise ValueError("metadata_mismatch")
    quote = value.get("evidence_quote")
    if not isinstance(quote, str) or not quote or quote not in body:
        raise ValueError("unsupported_quote")
    typ = value.get("event_type")
    if typ not in TYPES or not re.search(TYPES[typ], quote, re.IGNORECASE):
        raise ValueError("unsupported_type")
    errors = {}
    row = dict(value)
    available = str((pd.Timestamp(publication) + pd.Timedelta(days=1)).date())
    if row.get("available_from") != available:
        errors["available_from"] = (
            "corrected_to_authoritative_publication_plus_one_day_scenario"
        )
    row["available_from"] = available
    date = row.get("event_date")
    if date is not None:
        try:
            supported = date in quote and pd.Timestamp(date) <= pd.Timestamp(
                publication
            )
        except (ValueError, TypeError):
            supported = False
        if not supported:
            row["event_date"] = None
            errors["event_date"] = "unsupported_inferred_or_future_date_abstained"
    confidence = row.get("confidence")
    if confidence in (None, "unknown"):
        row["confidence"] = None
    else:
        try:
            confidence = float(confidence)
            if not 0 <= confidence <= 1:
                raise ValueError
            row["confidence"] = confidence
        except (ValueError, TypeError):
            row["confidence"] = None
            errors["confidence"] = "unsupported_confidence_abstained"
    if row.get("severity") != "unknown":
        errors["severity"] = "unverified_severity_abstained"
    row["severity"] = "unknown"
    # Native model IDs never authorize geographic linkage.
    if row.get("municipality_id") is not None:
        errors["municipality_id"] = "untrusted_model_id_discarded"
    name = row.get("municipality_name")
    tid = None
    status = "missing"
    token = None
    if name and normal(name) in normal(quote):
        tid, status = resolve_geo(name, region, int(publication[:4]), lookup)
    elif name:
        errors["municipality_name"] = "unsupported_geography_abstained"
    if tid is None:
        tid, name, token, status = quoted_geography(
            quote, region, int(publication[:4]), lookup
        )
    row.update(
        municipality_id=tid,
        municipality_name=name,
        geography_status=status,
        geography_evidence_token=token,
        geography_method="official_dictionary_NER_inside_LLM_quote",
        field_errors=errors,
    )
    return row


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((DATA / "llm_outputs.json").read_text())
    inputs = json.loads((DATA / "extraction_inputs.json").read_text())
    source = {d["source_url"]: d for d in inputs}
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    rows = []
    events = []
    for record in records:
        d = source[record["source_url"]]
        row = {
            "source_url": record["source_url"],
            "strict_accepted": record["accepted"],
            "accepted": False,
            "rejection": "",
        }
        try:
            v = validate_fields(
                record.get("parsed"),
                d["body"],
                d["publication_date"],
                d["region"],
                lookup,
            )
            row.update(v)
            row["accepted"] = True
            if v["municipality_id"] is not None:
                events.append(
                    dict(
                        v,
                        source_url=record["source_url"],
                        territory_id=v["municipality_id"],
                    )
                )
        except (ValueError, TypeError, KeyError) as e:
            row["rejection"] = str(e)
        row["field_errors"] = json.dumps(
            row.get("field_errors", {}), ensure_ascii=False
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "structured_events.csv", index=False)
    pd.DataFrame(events).to_csv(OUT / "llm_quote_exposure_registry.csv", index=False)
    base = pd.read_parquet(ROOT / "reports/peer_residual_review/detector_panel.parquet")
    base = base[base.method.eq("ewma") & base.signal.eq("own")].copy()
    details = []
    burdens = []
    for lag in [0, 1, 2]:
        available = np.array(
            [
                event_available(events, r.territory_id, r.target, lag, 2)
                for r in base.itertuples()
            ]
        )
        for policy in ["baseline", "llm_quote_event_0.75", "global_0.75"]:
            frame = base.copy()
            factor = np.ones(len(frame))
            if policy == "llm_quote_event_0.75":
                factor[available] = 0.75
            if policy == "global_0.75":
                factor[:] = 0.75
            frame["active_alert"] = frame.observed & (
                frame.score > frame.threshold * factor
            )
            frame["alert_episode"] = False
            for _, group in frame.groupby("territory_id", sort=False):
                g = group.sort_values("target")
                frame.loc[g.index, "alert_episode"] = alert_episodes(
                    g.active_alert.to_numpy()[None, :]
                )[0]
            frame["event_available"] = available
            frame["threshold_factor"] = factor
            frame["policy"] = policy
            frame["release_lag_months"] = lag
            details.append(frame)
            n = int(frame.observed.sum())
            burdens.append(
                {
                    "policy": policy,
                    "release_lag_months": lag,
                    "observed_months": n,
                    "event_available_months": int((frame.observed & available).sum()),
                    "active_alert_months": int(frame.active_alert.sum()),
                    "episodes": int(frame.alert_episode.sum()),
                    "active_per100": 100 * frame.active_alert.sum() / n,
                    "unlabeled_active_months": int(
                        (frame.active_alert & ~available).sum()
                    ),
                    "real_false_alarm_rate": None,
                }
            )
    pd.DataFrame(burdens).to_csv(OUT / "monitoring_burden.csv", index=False)
    pd.concat(details, ignore_index=True).to_parquet(
        OUT / "detector_panel.parquet", index=False
    )
    paths = [
        "src/sberindex/external/news_event_field_review.py",
        "data/external/news_event_extraction_review/llm_outputs.json",
        "data/external/news_event_extraction_review/extraction_inputs.json",
        "reports/peer_residual_review/detector_panel.parquet",
        "results/municipal_lookup.csv",
    ]
    audit = {
        "status": "complete",
        "new_inference": False,
        "retrospective_extractor_development": True,
        "strict_original_accepted": sum(r["accepted"] for r in records),
        "field_variant_accepted": sum(r["accepted"] for r in rows),
        "municipal_exposures": len(events),
        "municipality_ids": sorted({e["territory_id"] for e in events}),
        "event_date_inference_allowed": False,
        "threshold_factor": 0.75,
        "threshold_tuned_to_cases": False,
        "manual_registry_used": False,
        "municipal_method": "dictionary NER inside exact native LLM quote; native municipality_name was null",
        "absence_label": "unlabeled",
        "real_false_alarm_rate": None,
        "causality": False,
        "input_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
        },
    }
    exposure_ids = {e["territory_id"] for e in events}
    panel_ids = set(base.territory_id)
    audit["municipal_ids_in_detector_panel"] = sorted(exposure_ids & panel_ids)
    audit["municipal_ids_absent_detector_panel"] = sorted(exposure_ids - panel_ids)
    pd.concat(details, ignore_index=True).loc[
        lambda p: p.territory_id.isin(exposure_ids)
    ].to_csv(OUT / "municipal_case_months.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (OUT / "REPORT.md").write_text(
        "# Полевая валидация тех же native ответов Qwen\n\n```json\n"
        + json.dumps(audit, ensure_ascii=False, indent=2)
        + "\n```\n\n```csv\n"
        + pd.DataFrame(burdens).to_csv(index=False)
        + "\n```\n\nСохранены исходные строгие результаты 2/12. Вариант повторно разбирает ровно те же 12 сохранённых native ответов: неподтверждённая дата становится null с field_errors, а поддержанные тип и точная цитата сохраняются. Неподдержанные цитаты/типы остаются отклонёнными. ID разрешается только уникальным словарным NER внутри буквальной модельной цитаты, с узкими русскими окончаниями; это гибридная постобработка, не native геокодирование Qwen.\n\nРазвитие экстрактора ретроспективное, после наблюдения прежних отказов; независимая точность не заявляется. Порог EWMA не менялся: factor 0.75, публикация + 1 день, окно 2 месяца, отдельная LLM-quote экспозиция против baseline и global 0.75. Ручной реестр не используется в этом варианте. Неизвестный onset не восстанавливается из даты публикации. Событийная экспозиция не является границей расходов или причинным эффектом; отсутствие источника не является отрицательной меткой/FAR.\n"
    )
    with (OUT / "REPORT.md").open("a") as handle:
        handle.write(
            "\nМО вне исходной detector-панели: "
            + str(audit["municipal_ids_absent_detector_panel"])
            + "; в нагрузку входят только наблюдаемые месяцы с доступной экспозицией. Неизвестные дата события и confidence остаются null.\n"
        )
    return audit


if __name__ == "__main__":
    print(run())
