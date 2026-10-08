"""New dated official-page excerpt batch; preserves both previous Qwen batches."""

import argparse
import hashlib
import importlib.metadata
import json
import re

import numpy as np
import pandas as pd

from sberindex.detection.event_conditioned_review import event_available
from sberindex.detection.peer_residual_review import alert_episodes
from sberindex.external.news_event_extraction_review import MODEL, PROMPT, REVISION
from sberindex.external.news_event_field_review import validate_fields
from sberindex.paths import ROOT

DATA = ROOT / "data/external/news_body_recovery_20261007"
OUT = ROOT / "reports/news_body_recovery_20261007"
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


def article_excerpt(response, publication):
    if "Source: open" not in response or "Content type: text/html" not in response:
        raise ValueError("not_official_open_article_text")
    # Preserve visible link labels; remove tool-only citation syntax.
    plain = re.sub(
        r"cite[^†]+†([^]+)", lambda m: m.group(1).split("†")[0], response
    )
    plain = re.sub(r"[^]*", "", plain)
    plain = re.sub(r"L\d+:\s*", "\n", plain)
    date = pd.Timestamp(publication)
    start = re.search(
        rf"(?m)^\s*{date.day}\s+{MONTHS[date.month - 1]}\s+{date.year}(?:\s*,\s*\d\d:\d\d)?\s*$",
        plain,
    )
    if not start:
        raise ValueError("authoritative_page_publication_date_not_visible")
    body = plain[start.start() :]
    end = re.search(
        r"(?m)^\s*(?:Оцените материал|Поделиться|Подписка|Выберите теги|Мы используем файлы cookie)\b",
        body,
    )
    if end:
        body = body[: end.start()]
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def prepare_inputs():
    inputs, coverage = [], []
    for path in sorted(DATA.glob("page_*.json")):
        j = json.loads(path.read_text())
        row = j["row"]
        text = j["response"]
        entry = {
            "source_url": row["source_url"],
            "publication_date": row["publication_date"],
            "region": int(row["region"]),
            "snapshot_path": str(path.relative_to(ROOT)),
            "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "capture_kind": j["capture_kind"],
            "historical_vintage_verified": False,
            "accepted_for_extraction": False,
        }
        try:
            body = article_excerpt(text, row["publication_date"])
            if len(body) < 300:
                raise ValueError("insufficient_article_excerpt")
            inputs.append(
                dict(
                    entry,
                    body=body[:5500],
                    body_characters=len(body),
                    input_capped_5500=len(body) > 5500,
                )
            )
            entry.update(
                accepted_for_extraction=True,
                body_characters=len(body),
                input_capped_5500=len(body) > 5500,
            )
        except ValueError as e:
            entry["rejection"] = str(e)
        coverage.append(entry)
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "input_manifest.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "documents": inputs,
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    return inputs, coverage


def infer(inputs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, local_files_only=True
    ).to("mps" if torch.backends.mps.is_available() else "cpu")
    path = DATA / "native_outputs.json"
    outputs = json.loads(path.read_text()) if path.exists() else []
    done = {x["source_url"] for x in outputs}
    for d in inputs:
        if d["source_url"] in done:
            continue
        msg = f"publication_date={d['publication_date']} region={d['region']}\nBODY:\n{d['body']}"
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": PROMPT}, {"role": "user", "content": msg}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(model.device)
        with torch.inference_mode():
            gen = model.generate(ids, max_new_tokens=350, do_sample=False)
        raw = tokenizer.decode(gen[0, ids.shape[-1] :], skip_special_tokens=True)
        row = {"source_url": d["source_url"], "raw_output": raw}
        try:
            row["parsed"] = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        except (ValueError, TypeError) as e:
            row["parse_error"] = str(e)
        outputs.append(row)
        path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")
        print(f"recovered Qwen {len(outputs)}/{len(inputs)}", flush=True)
    (DATA / "runtime_versions.json").write_text(
        json.dumps(
            {
                p: importlib.metadata.version(p)
                for p in ["torch", "transformers", "huggingface_hub"]
            },
            indent=2,
        )
        + "\n"
    )
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return outputs


def require_complete_outputs(inputs, outputs):
    expected = [x["source_url"] for x in inputs]
    actual = [x["source_url"] for x in outputs]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Incomplete or duplicated extraction checkpoint")


def report(inputs, coverage):
    outputs = json.loads((DATA / "native_outputs.json").read_text())
    require_complete_outputs(inputs, outputs)
    sources = {x["source_url"]: x for x in inputs}
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    rows = []
    events = []
    for output in outputs:
        d = sources[output["source_url"]]
        row = {"source_url": d["source_url"], "accepted": False}
        try:
            v = validate_fields(
                output.get("parsed"),
                d["body"],
                d["publication_date"],
                d["region"],
                lookup,
            )
            row.update(v, accepted=True)
            if v["municipality_id"] is not None:
                events.append(
                    {
                        "territory_id": int(v["municipality_id"]),
                        "available_from": v["available_from"],
                        "source_url": d["source_url"],
                    }
                )
        except (ValueError, TypeError, KeyError) as e:
            row["rejection"] = str(e)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "structured_events.csv", index=False)
    # Include old two municipal links without rewriting old outputs.
    prior = pd.read_csv(
        ROOT
        / "reports/news_event_extraction_review/expanded_batch/llm_quote_exposure_registry.csv"
    )
    prior_events = prior.to_dict("records") if len(prior) else []
    all_events = events + prior_events
    pd.DataFrame(all_events).to_csv(OUT / "quote_exposure_registry.csv", index=False)
    panel = pd.read_parquet(
        ROOT / "reports/peer_residual_review/detector_panel.parquet"
    )
    panel = panel[(panel.method == "ewma") & (panel.signal == "own")].copy()
    burden = []
    detail = []
    for policy in ["baseline", "quoted_events_factor075"]:
        available = np.array(
            [
                event_available(all_events, r.territory_id, r.target, 0, 2)
                for r in panel.itertuples()
            ]
        )
        factor = (
            np.where(available, 0.75, 1.0)
            if policy != "baseline"
            else np.ones(len(panel))
        )
        flags = panel.observed & (panel.score > panel.threshold * factor)
        episodes = pd.Series(False, index=panel.index)
        for _, g in panel.groupby("territory_id", sort=False):
            g = g.sort_values("target")
            episodes.loc[g.index] = alert_episodes(
                flags.loc[g.index].to_numpy()[None, :]
            )[0]
        burden.append(
            {
                "policy": policy,
                "observed_months": int(panel.observed.sum()),
                "exposed_observed_months": int((panel.observed & available).sum()),
                "active_alert_months": int(flags.sum()),
                "episodes": int(episodes.sum()),
                "episodes_per100": 100 * episodes.sum() / panel.observed.sum(),
            }
        )
        detail.append(
            panel.assign(
                policy=policy,
                event_available=available,
                active_alert=flags,
                alert_episode=episodes,
            )
        )
    pd.DataFrame(burden).to_csv(OUT / "monitoring_burden.csv", index=False)
    pd.concat(detail).to_parquet(OUT / "detector_panel.parquet", index=False)
    panel_ids = set(panel.territory_id)
    ids = sorted({int(e["territory_id"]) for e in all_events})
    inputs_paths = [
        "src/sberindex/external/news_body_recovery.py",
        "src/sberindex/external/news_event_field_review.py",
        "src/sberindex/external/news_event_extraction_review.py",
        "src/sberindex/detection/event_conditioned_review.py",
        "src/sberindex/detection/peer_residual_review.py",
        "results/municipal_lookup.csv",
        "reports/peer_residual_review/detector_panel.parquet",
        "reports/news_event_extraction_review/expanded_batch/llm_quote_exposure_registry.csv",
        str((DATA / "input_manifest.json").relative_to(ROOT)),
        str((DATA / "native_outputs.json").relative_to(ROOT)),
    ]
    inputs_paths += [x["snapshot_path"] for x in coverage]
    inputs_paths += [
        "docs/protocols/NEWS_BODY_RECOVERY.md",
        "data/external/news_event_extraction_review/extraction_inputs.json",
        "data/external/news_event_extraction_review/expanded_batch/batch_manifest.json",
        "reports/news_event_extraction_review/expanded_batch/structured_events.csv",
    ]
    old_inputs = json.loads(
        (
            ROOT / "data/external/news_event_extraction_review/extraction_inputs.json"
        ).read_text()
    )
    expanded_inputs = json.loads(
        (
            ROOT
            / "data/external/news_event_extraction_review/expanded_batch/batch_manifest.json"
        ).read_text()
    )["input_documents"]
    old_urls = {x["source_url"] for x in old_inputs + expanded_inputs}
    new_urls = {x["source_url"] for x in inputs}
    overlap = old_urls & new_urls
    old_rows = pd.read_csv(ROOT / inputs_paths[-1])
    audit = {
        "status": "complete",
        "requested_pages": len(coverage),
        "prior_input_documents": len(old_urls),
        "overlapping_prior_urls": sorted(overlap),
        "combined_distinct_input_documents": len(old_urls | new_urls),
        "prior_accepted": int(old_rows.accepted.sum()),
        "combined_accepted_unique_urls": len(
            set(old_rows.loc[old_rows.accepted, "source_url"])
            | {r["source_url"] for r in rows if r["accepted"]}
        ),
        "accepted_new_regions": sorted(
            {int(r["region"]) for r in rows if r["accepted"]}
        ),
        "eligible_body_excerpts": len(inputs),
        "accepted_new": sum(x["accepted"] for x in rows),
        "new_municipal_links": len(events),
        "combined_unique_municipalities": len(ids),
        "municipalities_in_detector_panel": [i for i in ids if i in panel_ids],
        "input_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs_paths
        },
        "historical_vintage_verified": False,
        "full_article_completeness_verified": False,
        "full_corpus": False,
        "threshold_factor": 0.75,
        "threshold_tuned": False,
        "real_false_alarm_rate": None,
        "model": MODEL,
        "revision": REVISION,
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    (OUT / "REPORT.md").write_text(
        "# Восстановление текстов 80 официальных публикаций\n\n"
        + f"Запрошено {audit['requested_pages']} страниц; пригодны для извлечения {audit['eligible_body_excerpts']}. Принято {audit['accepted_new']} новых фрагментов. Всего с прежними партиями: {audit['combined_distinct_input_documents']} различных входов и {audit['combined_accepted_unique_urls']} подтверждённых фрагментов. Муниципальных ID в объединённом реестре: {audit['combined_unique_municipalities']}; в расходной панели: {len(audit['municipalities_in_detector_panel'])}.\n\n"
        + "| Политика | Наблюдаемые МО-месяцы | Доступная новость | Эпизоды | На 100 МО-месяцев |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {b['policy']} | {b['observed_months']} | {b['exposed_observed_months']} | {b['episodes']} | {b['episodes_per100']:.3f} |"
            for b in burden
        )
        + "\n\nОфициальные страницы повторно открыты 7 октября 2026 через инструмент веб-чтения после таймаута прямого TLS-запроса. Сохранены текстовые выдержки страницы, а не только поисковые сниппеты. Проверена видимая дата публикации; навигация до неё и типовые нижние блоки удалены. Полнота статьи и историческая неизменность не подтверждены. Вход Qwen ограничен 5500 символами, прежние два опыта не изменены. Используются тот же checkpoint, prompt, max_new_tokens=350 и проверка полей по дословной цитате. Проверка подтверждает цитату и упоминание типа явления, но не всегда направление риска: например, благоприятный прогноз паводка содержит то же ключевое слово. Есть предупреждения о будущих явлениях, а не сообщения о наступившем событии. Поддержанные фрагменты и география не объявляются разметкой реализованных бедствий или истинных структурных разрывов расходов.\n\nОтсутствие новости не является отрицательной меткой. Нагрузка тревог, а не реальная частота ложных срабатываний, сохранена в monitoring_burden.csv. Порог 0,75 и окно двух месяцев неизменны, не подбираются ради получения дополнительных тревог.\n\n## Манифест проверки\n\n```json\n"
        + json.dumps(audit, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    print(
        json.dumps(
            {
                k: audit[k]
                for k in [
                    "requested_pages",
                    "eligible_body_excerpts",
                    "accepted_new",
                    "new_municipal_links",
                    "combined_unique_municipalities",
                    "municipalities_in_detector_panel",
                ]
            },
            ensure_ascii=False,
        )
    )
    print(pd.DataFrame(burden).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    inputs, coverage = prepare_inputs()
    print("Eligible excerpts:", len(inputs), flush=True)
    if args.prepare_only:
        return
    if args.infer:
        infer(inputs)
    report(inputs, coverage)


if __name__ == "__main__":
    main()
