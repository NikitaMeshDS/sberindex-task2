"""Additional saved official page-body batch, preserving the original twelve outputs."""

import hashlib
import html
import importlib.metadata
import json
import re

import numpy as np
import pandas as pd

from sberindex.detection.event_conditioned_review import event_available
from sberindex.detection.peer_residual_review import alert_episodes
from sberindex.external.news_event_extraction_review import (
    MODEL,
    PROMPT,
    REVISION,
    body_text,
    validate_output,
)
from sberindex.external.news_event_field_review import validate_fields
from sberindex.paths import ROOT

DATA = ROOT / "data/external/news_event_extraction_review/expanded_batch"
OUT = ROOT / "reports/news_event_extraction_review/expanded_batch"
MONTHS = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


def reliable_page(raw, body, publication):
    """Require captured open page and its own visible date; search results are excluded."""
    if "Source: open" not in raw or "Content type: text/html" not in raw:
        return False, "not_saved_open_html_page"
    date = pd.Timestamp(publication)
    if not re.search(
        rf"(?<!\d){date.day}\s+{MONTHS[date.month - 1]}\s+{date.year}\b",
        body,
        re.IGNORECASE,
    ):
        return False, "publication_not_literal_page_date"
    if len(body) < 300:
        return False, "insufficient_non_navigation_body"
    return True, "official_saved_page_body_with_visible_publication_date"


def build_manifest():
    registry = pd.read_csv(
        ROOT / "data/external/regional_news_review/registry.csv"
    ).fillna("")
    original = json.loads(
        (
            ROOT / "data/external/news_event_extraction_review/extraction_inputs.json"
        ).read_text()
    )
    already = {x["source_url"] for x in original}
    coverage = []
    inputs = []
    for row in (
        registry.drop_duplicates("snapshot_path")
        .sort_values("snapshot_path")
        .itertuples()
    ):
        entry = {
            "source_url": row.source_url,
            "snapshot_path": row.snapshot_path,
            "publication_date": row.published_date,
            "region": int(row.region_code),
        }
        if not re.search(r"https?://(?:[^/]+\.)?mchs\.gov\.ru/", row.source_url):
            entry.update(eligible=False, reason="not_official_mchs_host")
            coverage.append(entry)
            continue
        raw = (ROOT / row.snapshot_path).read_text()
        body = body_text(ROOT / row.snapshot_path)
        eligible, reason = reliable_page(raw, body, row.published_date)
        if row.source_url in already:
            eligible = False
            reason = "already_original12_preserved"
        entry.update(
            eligible=eligible,
            reason=reason,
            body_characters=len(body),
            snapshot_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        )
        # Some long saved extracts are truncated at the SAME 5500-character model cap as original pilot.
        entry["model_input_capped_5500"] = len(body) == 5500
        coverage.append(entry)
        if eligible:
            inputs.append(dict(entry, body=body))
    # Raw official MChS HTML pages have explicit articleBody/datePublished metadata.
    checks = json.loads(
        (ROOT / "data/external/regional_news_expansion/source_checks.json").read_text()
    )
    for check in checks:
        if (
            not check["path"].endswith(".html")
            or "mchs.gov.ru" not in check["source_url"]
        ):
            continue
        path = ROOT / check["path"]
        raw = path.read_text()
        date_match = re.search(
            r'itemprop="datePublished"[^>]*datetime="(\d{4}-\d{2}-\d{2})', raw
        )
        article = re.search(
            r'<article[^>]*itemprop="articleBody"[^>]*>(.*?)</article>', raw, re.DOTALL
        )
        region_match = re.match(r"https://(\d+)\.mchs\.gov\.ru/", check["source_url"])
        entry = {
            "source_url": check["source_url"],
            "snapshot_path": check["path"],
            "eligible": bool(date_match and article and region_match),
            "reason": "official_HTML_articleBody_and_datePublished"
            if date_match and article and region_match
            else "missing_article_or_date_or_region",
            "snapshot_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        }
        if entry["eligible"]:
            publication = date_match[1]
            region = int(region_match[1])
            text = html.unescape(re.sub(r"<[^>]+>", " ", article[1]))
            text = re.sub(r"[ \t]+", " ", text)
            body = text.strip()[:5500]
            entry.update(
                publication_date=publication,
                region=region,
                body_characters=len(body),
                model_input_capped_5500=len(body) == 5500,
            )
            inputs.append(dict(entry, body=body))
        coverage.append(entry)
    warning = json.loads(
        (
            ROOT / "data/external/mchs_archive_review/verified_warning_pages.json"
        ).read_text()
    )["response"]
    warnings = json.loads(
        (
            ROOT / "data/external/mchs_archive_review/verified_warning_annotations.json"
        ).read_text()
    )["warnings"]
    for row in warnings:
        part = next(
            (
                p
                for p in warning.split(
                    "--------------------------------------------------------------------------------"
                )
                if row["source_url"] in p
            ),
            None,
        )
        if part is None:
            coverage.append(
                {
                    "source_url": row["source_url"],
                    "eligible": False,
                    "reason": "missing_saved_warning_page",
                }
            )
            continue
        derived = DATA / ("warning_" + row["source_url"].split("/")[-1] + ".json")
        derived.write_text(json.dumps(part, ensure_ascii=False))
        body = body_text(derived)
        eligible, reason = reliable_page(part, body, row["published_date"])
        entry = {
            "source_url": row["source_url"],
            "snapshot_path": str(derived.relative_to(ROOT)),
            "publication_date": row["published_date"],
            "region": int(row["region_code"]),
            "eligible": eligible,
            "reason": reason,
            "body_characters": len(body),
            "model_input_capped_5500": len(body) == 5500,
            "snapshot_sha256": hashlib.sha256(derived.read_bytes()).hexdigest(),
            "parent_capture": "data/external/mchs_archive_review/verified_warning_pages.json",
            "document_role": "warning_not_observed_onset",
        }
        coverage.append(entry)
        if eligible:
            inputs.append(dict(entry, body=body))
    # Saved official RosHydromet bodies lack a single reliable municipal region binding.
    weather = json.loads(
        (ROOT / "data/external/weather_body_sample/manifest.json").read_text()
    )
    for page in weather["pages"]:
        coverage.append(
            {
                "source_url": page["source_url"],
                "snapshot_path": page["path"],
                "eligible": False,
                "reason": "official_weather_body_but_no_reliable_single_region_binding",
                "snapshot_sha256": page.get("sha256"),
            }
        )
    # The 81 historical archive items are search-index extracts unless verified above.
    archive = pd.read_csv(
        ROOT / "data/external/mchs_archive_review/historical_items.csv"
    ).fillna("")
    seen = {r["source_url"] for r in coverage}
    for row in archive.itertuples():
        if row.source_url not in seen:
            coverage.append(
                {
                    "source_url": row.source_url,
                    "snapshot_path": row.snapshot_path,
                    "eligible": False,
                    "reason": "historical_search_index_not_saved_article_body",
                    "publication_date": row.published_date,
                    "region": int(row.region_code),
                }
            )
    return inputs, coverage


def infer(inputs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    (DATA / "runtime_versions.json").write_text(
        json.dumps(
            {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "huggingface_hub"]
            },
            indent=2,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, local_files_only=True
    ).to("mps" if torch.backends.mps.is_available() else "cpu")
    outputs = (
        json.loads((DATA / "native_outputs.json").read_text())
        if (DATA / "native_outputs.json").exists()
        else []
    )
    completed = {r["source_url"] for r in outputs}
    remaining = [d for d in inputs if d["source_url"] not in completed]
    for n, d in enumerate(remaining):
        message = f"publication_date={d['publication_date']} region={d['region']}\nBODY:\n{d['body']}"
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": message},
        ]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(ids, max_new_tokens=350, do_sample=False)
        raw = tokenizer.decode(generated[0, ids.shape[-1] :], skip_special_tokens=True)
        record = {"source_url": d["source_url"], "raw_output": raw}
        try:
            record["parsed"] = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        except (ValueError, TypeError) as e:
            record["parse_error"] = str(e)
        outputs.append(record)
        (DATA / "native_outputs.json").write_text(
            json.dumps(outputs, ensure_ascii=False, indent=2)
        )
        print(f"expanded checkpoint {len(outputs)}/{len(inputs)}", flush=True)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return outputs


def run(run_inference=False):
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, coverage = build_manifest()
    (DATA / "batch_manifest.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "max_new_tokens": 350,
                "do_sample": False,
                "torch_threads": 1,
                "input_documents": inputs,
                "coverage": coverage,
                "regional_binding": "same saved manual source context as original; multi-region page uses first saved binding, not inferred geography",
                "new_batch_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    pd.DataFrame(coverage).to_csv(OUT / "coverage_exclusions.csv", index=False)
    records = (
        infer(inputs)
        if run_inference
        else json.loads((DATA / "native_outputs.json").read_text())
    )
    original = json.loads(
        (
            ROOT / "data/external/news_event_extraction_review/llm_outputs.json"
        ).read_text()
    )
    original_inputs = json.loads(
        (
            ROOT / "data/external/news_event_extraction_review/extraction_inputs.json"
        ).read_text()
    )
    source = {d["source_url"]: d for d in [*original_inputs, *inputs]}
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    rows = []
    events = []
    for record in [*original, *records]:
        d = source[record["source_url"]]
        row = {
            "source_url": record["source_url"],
            "batch": "original12" if record in original else "expanded",
            "accepted": False,
            "rejection": "",
        }
        try:
            validate_output(
                record.get("parsed"),
                d["body"],
                d["publication_date"],
                d["region"],
                lookup,
            )
            row["strict_accepted"] = True
        except (ValueError, TypeError, KeyError):
            row["strict_accepted"] = False
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
    contract = pd.read_csv(
        ROOT
        / "reports/news_event_extraction_review/field_validation_variant/structured_events.csv",
        nrows=0,
    ).columns
    pd.DataFrame(rows).reindex(columns=contract).to_csv(
        OUT / "structured_events.csv", index=False
    )
    pd.DataFrame(rows)[["source_url", "batch", "strict_accepted", "accepted"]].to_csv(
        OUT / "batch_labels.csv", index=False
    )
    pd.DataFrame(events).to_csv(OUT / "llm_quote_exposure_registry.csv", index=False)
    panel = pd.read_parquet(
        ROOT / "reports/peer_residual_review/detector_panel.parquet"
    )
    panel = panel[panel.method.eq("ewma") & panel.signal.eq("own")].copy()
    details = []
    burdens = []
    for lag in [0, 1, 2]:
        available = np.array(
            [
                event_available(events, r.territory_id, r.target, lag, 2)
                for r in panel.itertuples()
            ]
        )
        for policy in ["baseline", "expanded_llm_quote_0.75", "global_0.75"]:
            frame = panel.copy()
            factor = np.ones(len(frame))
            if policy == "expanded_llm_quote_0.75":
                factor[available] = 0.75
            if policy == "global_0.75":
                factor[:] = 0.75
            frame["event_available"] = available
            frame["threshold_factor"] = factor
            frame["active_alert"] = frame.observed & (
                frame.score > frame.threshold * factor
            )
            frame["alert_episode"] = False
            for _, g in frame.groupby("territory_id", sort=False):
                g = g.sort_values("target")
                frame.loc[g.index, "alert_episode"] = alert_episodes(
                    g.active_alert.to_numpy()[None, :]
                )[0]
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
    exposure_ids = {e["territory_id"] for e in events}
    paths = [
        "src/sberindex/external/news_event_expanded_review.py",
        "src/sberindex/external/news_event_field_review.py",
        "data/external/news_event_extraction_review/expanded_batch/batch_manifest.json",
        "data/external/news_event_extraction_review/expanded_batch/native_outputs.json",
        "data/external/news_event_extraction_review/llm_outputs.json",
        "data/external/news_event_extraction_review/extraction_inputs.json",
        "data/external/regional_news_review/registry.csv",
        "results/municipal_lookup.csv",
        "docs/protocols/NEWS_EVENT_EXPANDED_REVIEW.md",
        "reports/peer_residual_review/detector_panel.parquet",
    ]
    audit = {
        "status": "complete",
        "new_inference_documents": len(records),
        "original12_preserved": True,
        "combined_documents": len(rows),
        "combined_accepted": sum(r["accepted"] for r in rows),
        "expanded_accepted": sum(
            r["accepted"] for r in rows if r["batch"] == "expanded"
        ),
        "municipal_exposures": len(events),
        "municipality_ids": sorted(exposure_ids),
        "ids_absent_detector_panel": sorted(exposure_ids - set(panel.territory_id)),
        "threshold_factor": 0.75,
        "threshold_tuned": False,
        "retrospective_extractor_development": True,
        "publication_vintage_verified": False,
        "full_corpus": False,
        "absence_label": "unlabeled",
        "event_onset_inferred": False,
        "real_false_alarm_rate": None,
        "model": MODEL,
        "revision": REVISION,
        "input_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
        },
    }
    audit["coverage_reason_counts"] = (
        pd.Series([r["reason"] for r in coverage]).value_counts().to_dict()
    )
    audit["runtime_versions"] = (
        json.loads((DATA / "runtime_versions.json").read_text())
        if (DATA / "runtime_versions.json").exists()
        else {"status": "saved inference runtime manifest not available"}
    )
    audit["saved_capture_completeness"] = (
        "raw HTML articleBody for 3 documents; other official saved page-body excerpts; input capped 5500 characters"
    )
    capture_paths = {d["snapshot_path"] for d in [*original_inputs, *inputs]} | {
        "data/external/mchs_archive_review/verified_warning_pages.json",
        "data/external/regional_news_expansion/source_checks.json",
    }
    captures = [
        {
            "path": path,
            "bytes": (ROOT / path).stat().st_size,
            "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
        for path in sorted(capture_paths)
    ]
    (OUT / "capture_manifest.json").write_text(
        json.dumps(captures, ensure_ascii=False, indent=2)
    )
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (OUT / "REPORT.md").write_text(
        "# Дополнительные сохранённые официальные страницы\n\n```json\n"
        + json.dumps(audit, ensure_ascii=False, indent=2)
        + "\n```\n\n```csv\n"
        + pd.DataFrame(burdens).to_csv(index=False)
        + "\n```\n\nВ дополнительную партию включены все оставшиеся сохранённые официальные страницы с подтверждённой датой публикации и содержательным телом: 3 исходных HTML articleBody и 14 сохранённых выдержек из открытых страниц. Полнота выдержек не заявляется. Поисковые сниппеты и навигация исключены. Совпадают фиксированная ревизия Qwen, промпт, детерминированная генерация максимум 350 токенов и ограничение входа 5500 символов исходного пилота. Ограничение входа отмечено явно. Регион — прежняя сохранённая ручная привязка источника; для страницы с несколькими регионами взята первая сохранённая привязка, поэтому это не полный реестр всех региональных экспозиций страницы.\n\nИсходные 12 ответов модели и строгие результаты сохранены. Объединённый вариант с проверкой отдельных полей заменяет неподтверждённые даты на null; муниципальный ID разрешается только уникальным словарём внутри точной модельной цитаты. Это ретроспективное развитие экстрактора, не независимая оценка. Отдельное сравнение EWMA всей панели с фиксированным множителем 0.75 не использует ручной реестр и не настраивается по случаям. Экспозиция не является разрывом расходов или причинным эффектом; отсутствие новости остаётся неразмеченным.\n"
    )
    return audit


if __name__ == "__main__":
    import sys

    print(run("--infer" in sys.argv))
