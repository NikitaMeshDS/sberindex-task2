"""Local structured extraction with literal evidence, dated availability, and abstention."""

import hashlib
import json
import re
from difflib import SequenceMatcher

import pandas as pd

from sberindex.paths import ROOT

OUT = ROOT / "reports/news_event_extraction_review"
DATA = ROOT / "data/external/news_event_extraction_review"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TYPES = {
    "flood": r"павод|подтоп|наводнен|размыв.*дамб",
    "fire": r"пожар|возгоран",
    "evacuation": r"эваку",
    "emergency": r"режим.*чрезвычайн|режим.*ЧС",
    "pollution": r"мазут|нефтепродукт",
}
PROMPT = """Extract one event from this Russian official news body. Return ONLY JSON with keys publication_date,event_date,available_from,region,municipality_id,municipality_name,event_type,severity,confidence,evidence_quote. Dates YYYY-MM-DD or null. publication_date and region must equal supplied metadata. available_from is publication_date plus one day (scenario). municipality_id must be null: a separate official dictionary resolves it. event_date null unless explicit calendar date in evidence_quote; never infer onset from publication. event_type one of flood,fire,evacuation,emergency,pollution,unknown. severity unknown unless explicitly supported. confidence between 0 and 1. evidence_quote must be a literal substring of the supplied body supporting event_type and municipality_name; do not paraphrase. If no support, null municipality_name, unknown event_type and empty evidence_quote. No future observed event dates."""


def normal(text):
    return re.sub(r"[^а-яa-z0-9]+", " ", str(text).lower().replace("ё", "е")).strip()


def resolve_geo(name, region, year, lookup):
    """Unique region/year alias; close fuzzy ties abstain; never bridge reforms."""
    if not name:
        return None, "missing"
    geo = lookup[(lookup.region_code == region) & (lookup.year == year)]
    n = normal(name)
    exact = geo[geo.municipal_district_name_short.map(normal) == n]
    if len(exact) == 1:
        return int(exact.iloc[0].territory_id), "exact"
    if len(exact) > 1:
        return None, "ambiguous_exact"
    scores = [
        (
            SequenceMatcher(None, n, normal(r.municipal_district_name_short)).ratio(),
            int(r.territory_id),
        )
        for r in geo.itertuples()
    ]
    scores.sort(reverse=True)
    if (
        scores
        and scores[0][0] >= 0.92
        and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.08)
    ):
        return scores[0][1], "fuzzy_unique"
    return None, "abstain"


def validate_output(value, body, publication, region, lookup):
    """Untrusted model output cannot set geography IDs, availability, or unsupported facts."""
    required = {
        "publication_date",
        "event_date",
        "available_from",
        "region",
        "municipality_id",
        "event_type",
        "severity",
        "confidence",
        "evidence_quote",
    }
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("missing_schema")
    if value["publication_date"] != publication or str(value["region"]) != str(region):
        raise ValueError("metadata_mismatch")
    available = str((pd.Timestamp(publication) + pd.Timedelta(days=1)).date())
    if value["available_from"] != available:
        raise ValueError("availability_mismatch")
    quote = value["evidence_quote"]
    if not isinstance(quote, str) or not quote or quote not in body:
        raise ValueError("unsupported_quote")
    typ = value["event_type"]
    if typ not in TYPES or not re.search(TYPES[typ], quote, re.IGNORECASE):
        raise ValueError("unsupported_type")
    if value["municipality_id"] is not None:
        raise ValueError("model_id_forbidden")
    name = value.get("municipality_name")
    if name and normal(name) not in normal(quote):
        raise ValueError("unsupported_geography")
    date = value["event_date"]
    # ISO support is strict; unparsed Russian dates abstain rather than invent.
    if date is not None and (
        date not in quote or pd.Timestamp(date) > pd.Timestamp(publication)
    ):
        raise ValueError("unsupported_or_future_event_date")
    confidence = (
        None if value["confidence"] in (None, "unknown") else float(value["confidence"])
    )
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError("invalid_confidence")
    if value["severity"] != "unknown":
        raise ValueError("severity_unverified")
    tid, status = resolve_geo(name, region, int(publication[:4]), lookup)
    return dict(
        value,
        confidence=confidence,
        municipality_id=tid,
        geography_status=status,
        available_from=available,
    )


def body_text(path):
    raw = path.read_text()
    try:
        content = json.loads(raw)
    except json.JSONDecodeError:
        content = raw
    content = (
        content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    )
    # MChS common navigation contains every region, so discard before page date/body.
    lines = content.splitlines()
    dated = [
        i
        for i, l in enumerate(lines)
        if re.search(r"L\d+:.*\b(?:2023|2024)\b", l)
        and re.search(
            r"январ|феврал|март|апрел|мая|июн|июл|август|сентябр|октябр|ноябр|декабр", l
        )
    ]
    if dated:
        lines = lines[dated[0] :]
    text = "\n".join(lines)
    text = re.sub(r"[^]*", "", text)
    text = re.sub(r"L\d+:\s*", "", text)
    return text[:5500]


def regex_extract(body):
    for kind, pattern in TYPES.items():
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            start = max(0, body.rfind("\n", 0, match.start()) + 1)
            end = body.find("\n", match.end())
            end = len(body) if end < 0 else end
            return kind, body[start:end]
    return "unknown", ""


def run(llm=False):
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    registry = pd.read_csv(
        ROOT / "data/external/regional_news_review/registry.csv"
    ).fillna("")
    official = (
        registry[registry.source_url.str.contains(r"mchs\.gov\.ru", regex=True)]
        .drop_duplicates("snapshot_path")
        .sort_values("snapshot_path")
    )
    docs = []
    for r in official.itertuples():
        body = body_text(ROOT / r.snapshot_path)
        kind, quote = regex_extract(body)
        docs.append(
            {
                "source_url": r.source_url,
                "snapshot_path": r.snapshot_path,
                "publication_date": r.published_date,
                "region": int(r.region_code),
                "body": body,
                "regex_type": kind,
                "regex_quote": quote,
                "manual_role": r.event_role,
                "manual_ids": r.territory_ids,
            }
        )
    # Freeze subset by source filename BEFORE model outputs, no detector-case inspection.
    selected = docs[:12]
    (DATA / "extraction_inputs.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2)
    )
    (DATA / "prompt.txt").write_text(PROMPT)
    records = []
    if llm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float16
        ).to("mps" if torch.backends.mps.is_available() else "cpu")
        for n, d in enumerate(selected):
            message = f"publication_date={d['publication_date']} region={d['region']}\nBODY:\n{d['body']}"
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": message},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids, max_new_tokens=350, do_sample=False
                )
            raw = tokenizer.decode(
                generated[0, input_ids.shape[-1] :], skip_special_tokens=True
            )
            record = {
                "source_url": d["source_url"],
                "raw_output": raw,
                "accepted": False,
            }
            try:
                value = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
                record["parsed"] = value
                record["validated"] = validate_output(
                    value, d["body"], d["publication_date"], d["region"], geo
                )
                record["accepted"] = True
            except (ValueError, KeyError, TypeError) as e:
                record["rejection"] = str(e)
            records.append(record)
            (DATA / "llm_outputs.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )
            print(
                f"extracted {n + 1}/{len(selected)} accepted={record['accepted']}",
                flush=True,
            )
    elif (DATA / "llm_outputs.json").exists():
        records = json.loads((DATA / "llm_outputs.json").read_text())
        source_map = {d["source_url"]: d for d in selected}
        for record in records:
            d = source_map[record["source_url"]]
            try:
                record["validated"] = validate_output(
                    record.get("parsed"),
                    d["body"],
                    d["publication_date"],
                    d["region"],
                    geo,
                )
                record["accepted"] = True
                record.pop("rejection", None)
            except (ValueError, TypeError, KeyError) as e:
                record["accepted"] = False
                record["rejection"] = str(e)
        (DATA / "llm_outputs.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2)
        )
    exported = []
    for record in records:
        row = dict(
            record.get("parsed", {}),
            source_url=record["source_url"],
            extraction_method="local_qwen_native",
            accepted=record["accepted"],
            rejection=record.get("rejection", ""),
        )
        if record.get("validated"):
            row.update(record["validated"])
        exported.append(row)
    columns = [
        "publication_date",
        "event_date",
        "available_from",
        "region",
        "municipality_id",
        "event_type",
        "severity",
        "confidence",
        "evidence_quote",
        "source_url",
        "extraction_method",
        "accepted",
        "rejection",
    ]
    pd.DataFrame(exported).reindex(columns=columns).to_csv(
        OUT / "structured_events.csv", index=False
    )
    pd.DataFrame([{k: v for k, v in d.items() if k != "body"} for d in docs]).to_csv(
        OUT / "regex_ablation.csv", index=False
    )
    archive = pd.read_csv(
        ROOT / "data/external/mchs_archive_review/historical_items.csv"
    )
    audit = {
        "status": "complete",
        "model": MODEL,
        "revision": REVISION,
        "local_only": True,
        "training": False,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "official_documents": len(docs),
        "llm_documents": len(records),
        "accepted": sum(r["accepted"] for r in records),
        "archive_documents": len(archive),
        "archive_2023": int(archive.published_date.str.startswith("2023").sum()),
        "full_corpus": False,
        "historical_vintage_verified": False,
        "manual_qa": "retrospective existing annotations, not independent or blinded",
        "absence_is": "unlabeled",
        "event_label": "exposure/context, not spending changepoint",
        "event_date_policy": "strict ISO exact support; Russian date normalization abstains",
        "news_forecast_training_claim": False,
    }
    audit["accepted_with_municipality_id"] = sum(
        bool(r.get("validated", {}).get("municipality_id")) for r in records
    )
    audit["confidence_policy"] = "unknown remains nullable; no fabricated probability"
    audit["validation_note"] = (
        "nullable confidence normalization followed initial inference; native schema compliance not claimed"
    )
    paths = [
        "configs/news_event_extraction_review.json",
        "src/sberindex/external/news_event_extraction_review.py",
        "docs/protocols/NEWS_EVENT_EXTRACTION_REVIEW.md",
        "data/external/news_event_extraction_review/prompt.txt",
        "data/external/news_event_extraction_review/extraction_inputs.json",
    ]
    audit["input_sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
    }
    if (DATA / "translation_gkg_audit.json").exists():
        audit["gdelt_slice"] = json.loads(
            (DATA / "translation_gkg_audit.json").read_text()
        )
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (OUT / "REPORT.md").write_text(
        f"""# Извлечение событий из официальных новостей\n\nЛокальная {MODEL}, revision `{REVISION}`. Обработано {len(records)} документов, строгую проверку прошло {audit["accepted"]}. Сохранены исходные ответы, JSON, точные цитаты, промпт и причины отказа. Идентификаторы разрешает только официальный справочник региона/года; неоднозначность приводит к отказу. Переформирование территории не означает сохранение географии.\n\nRegex-абляция охватывает {len(docs)} официальных документов с сохранённым текстом. Это выборочная ретроспективная QA, её согласие с предыдущими ручными аннотациями не является независимой точностью. Навигация МЧС с перечислением всех регионов исключена из тела. Дата публикации + сутки — сценарий доступности, не доказанный исторический винтаж. Строгий валидатор отклоняет дату события без буквального ISO-подтверждения; это снижает полноту для русских дат.\n\nЧастичный архив: {len(archive)} документов, из 2023 года {audit["archive_2023"]}; полная лента не восстановлена. Обучение прогнозной модели на «отсутствии новостей» некорректно: такие наблюдения не размечены. Улучшение прогноза расходов или причинный эффект не доказаны.\n"""
    )
    return audit


if __name__ == "__main__":
    import sys

    print(json.dumps(run("--llm" in sys.argv), ensure_ascii=False))
