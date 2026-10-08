"""Build an offline, read-only municipal research explorer from saved evidence."""

import base64
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sberindex.paths import ROOT
from tools.presentation_comparison_cases import calibration_for_origin


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4) if math.isfinite(value) else None
    return value


def news_visible(item, region, territory, cutoff):
    """An event is context only when geography and availability both match."""
    ids = item.get("municipality_ids", [])
    geography = int(item["region_code"]) == int(region)
    if ids:
        geography = geography and int(territory) in ids
    return geography and item["available_from"] <= cutoff


def build(root=ROOT):
    root = Path(root)
    source_paths = [
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "reports/growth_bridge_review/predictions.parquet",
        "reports/peer_residual_review/detector_panel.parquet",
        "data/external/mchs_archive_review/historical_items.csv",
        "tools/build_research_dashboard.py",
    ]
    raw = pd.read_parquet(root / source_paths[0])
    raw["month"] = pd.to_datetime(raw.date).dt.strftime("%Y-%m")
    geography = pd.read_csv(root / source_paths[1])
    geography = geography[geography.year.eq(2024)].set_index("territory_id")
    if not geography.index.is_unique:
        raise ValueError("Ambiguous official municipality ID")
    predictions = pd.read_parquet(root / source_paths[2])
    predictions = predictions[predictions.model.eq("hierarchy05_growth_bridge")].copy()
    predictions["category"] = "Все категории"
    category_path = (
        root / "reports/category_growth_review/full_bridge_predictions.parquet"
    )
    if category_path.exists():
        category = pd.read_parquet(category_path)
        category = category[
            category.model.eq("hierarchy05_growth_bridge")
            & category.category.ne("Все категории")
        ]
        predictions = pd.concat([predictions, category], ignore_index=True)
        source_paths.append(str(category_path.relative_to(root)))
    if predictions.duplicated(["category", "territory_id", "target", "horizon"]).any():
        raise ValueError("Duplicate saved forecast")
    total = predictions[
        predictions.category.eq("Все категории") & predictions.horizon.eq(1)
    ]
    scale = (
        raw[raw.category.eq("Все категории") & raw.month.str.startswith("2023-")]
        .groupby("territory_id")
        .value.mean()
    )
    calibration = {
        o: calibration_for_origin(total, o, scale)
        for o in sorted(total.origin.unique())
    }
    detector = pd.read_parquet(root / source_paths[3])
    detector = detector[detector.method.eq("ewma")]
    alarms = {}
    for r in detector.itertuples():
        alarms.setdefault(int(r.territory_id), {}).setdefault(r.signal, {})[
            r.target
        ] = {
            "score": r.score,
            "threshold": r.threshold,
            "observed": bool(r.observed),
            "active": bool(r.active_alert),
            "episode": bool(r.alert_episode),
        }
    history = {}
    for (tid, cat), g in raw.groupby(["territory_id", "category"]):
        history.setdefault(int(tid), {})[cat] = dict(zip(g.month, g.value))
    cities = {}
    for tid, g in predictions.groupby("territory_id"):
        if tid not in geography.index:
            continue
        geo = geography.loc[tid]
        forecast = {}
        for r in g.itertuples():
            lo = hi = None
            if r.category == "Все категории" and r.horizon == 1:
                cal = calibration[r.origin]
                if cal is not None:
                    radius = cal["normalized_radius"] * scale.loc[tid]
                    lo, hi = max(0, r.predicted - radius), r.predicted + radius
            # Per target: point forecast, lower, upper, known last month.
            forecast.setdefault(r.category, {}).setdefault(int(r.horizon), {})[
                r.target
            ] = [r.predicted, lo, hi, r.origin]
        cities[int(tid)] = {
            "id": int(tid),
            "name": geo.municipal_district_name_short,
            "full_name": geo.municipal_district_name,
            "region": geo.region_name,
            "region_code": int(geo.region_code),
            "lat": pd.to_numeric(geo.municipal_district_center_lat, errors="coerce"),
            "lon": pd.to_numeric(geo.municipal_district_center_lon, errors="coerce"),
            "history": history[int(tid)],
            "forecasts": forecast,
            "alarms": alarms.get(int(tid), {}),
        }
    news = pd.read_csv(root / source_paths[4]).to_dict("records")
    news = [
        {
            "url": r["source_url"],
            "title": r["title"],
            "region_code": int(r["region_code"]),
            "published": r["published_date"],
            "available_from": r["available_from_scenario"],
            "municipality_ids": [],
            "kind": "Официальная региональная публикация",
        }
        for r in news
    ]
    extracted_path = (
        root
        / "reports/news_event_extraction_review/field_validation_variant/structured_events.csv"
    )
    expanded_path = (
        root
        / "reports/news_event_extraction_review/expanded_batch/structured_events.csv"
    )
    if expanded_path.exists():
        extracted_path = expanded_path
    if extracted_path.exists():
        extracted = pd.read_csv(extracted_path)
        linked = {item["url"]: item for item in news}
        for row in extracted[extracted.accepted.eq(True)].itertuples():
            ids = [] if pd.isna(row.municipality_id) else [int(row.municipality_id)]
            linked[row.source_url] = {
                "url": row.source_url,
                "title": row.evidence_quote,
                "region_code": int(row.region),
                "published": row.publication_date,
                "available_from": row.available_from,
                "municipality_ids": ids,
                "kind": "Точная цитата Qwen и проверка по справочнику",
            }
        news = list(linked.values())
        source_paths.append(str(extracted_path.relative_to(root)))
    outline_path = root / "data/external/dashboard_geography/russia_outline.json"
    outline = json.loads(outline_path.read_text())
    source_paths.append(str(outline_path.relative_to(root)))
    payload = clean(
        {
            "version": 2,
            "outline": outline,
            "period": "2023–2024",
            "municipalities": cities,
            "categories": sorted(predictions.category.unique()),
            "months": sorted(predictions.target.unique()),
            "horizons": [1, 3, 6, 12],
            "news": news,
            "limitations": {
                "independent_temporal_test": False,
                "true_shock_labels": False,
                "intervals": "80% illustration; previous three completed months, 2023 scale; lag 0",
                "news": "Partial corpus; publication +1 day assumed; regional context is not local impact",
                "geography": "Official 2024 municipality-centre snapshot, historic availability unverified",
            },
        }
    )
    destination = root / "dashboard"
    destination.mkdir(exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    packed = base64.b64encode(
        gzip.compress(serialized.encode("utf-8"), mtime=0)
    ).decode("ascii")
    # Inline gzip remains usable with file://: no fetch, CDN, or category loss.
    (destination / "data.js").write_text(
        'window.RESEARCH_DATA_READY=(async()=>{const b=Uint8Array.from(atob("'
        + packed
        + '"),c=>c.charCodeAt(0));const stream=new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"));'
        + "window.RESEARCH_DATA=JSON.parse(await new Response(stream).text());return window.RESEARCH_DATA;})();\n"
    )
    assert json.loads(gzip.decompress(base64.b64decode(packed))) == payload
    report = root / "reports/research_dashboard"
    report.mkdir(parents=True, exist_ok=True)
    files = (
        [root / n for n in source_paths]
        + list(destination.glob("*.js"))
        + list(destination.glob("*.html"))
        + list(destination.glob("*.css"))
    )
    audit = {
        "status": "passed",
        "municipalities": len(cities),
        "categories": payload["categories"],
        "mapped_municipalities": sum(
            c["lat"] is not None and c["lon"] is not None
            for c in payload["municipalities"].values()
        ),
        "news_items": len(news),
        "uncompressed_json_bytes": len(serialized.encode("utf-8")),
        "packed_script_bytes": (destination / "data.js").stat().st_size,
        "lossless_gzip_roundtrip": True,
        "independent_temporal_validation": False,
        "input_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
            if p != destination / "data.js"
        },
        "output_sha256": {
            "dashboard/data.js": hashlib.sha256(
                (destination / "data.js").read_bytes()
            ).hexdigest()
        },
    }
    (report / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    (report / "REPORT.md").write_text(
        "# Интерактивный исследовательский интерфейс\n\nОткройте `dashboard/index.html` в браузере. Данные и код находятся локально; сервер, API-ключи и интернет не требуются. Используется стандартный DecompressionStream в современном браузере; все шесть категорий сохранены без потерь. Для локального HTTP-просмотра: `python -m http.server 8765 --bind 127.0.0.1` из корня проекта, адрес `/dashboard/`.\n\nКарта показывает центры МО из официального справочника на грубом контуре Natural Earth 1:110m (public domain, версия и SHA256 в data/external/dashboard_geography/provenance.json); контур не является официальной муниципальной географией. Ось расходов ограничена диапазоном показанных фактов, прогнозов и обеих границ интервала; в Орске отмечен апрельский паводок как контекст, а не обнаруженная точка. Отсутствующие координаты не восстановлены догадкой. Цвет — относительная ошибка сохранённого прогноза в выбранном месяце. График показывает скользящие прогнозы выбранного горизонта. Иллюстративные интервалы 80% доступны только для h1 и общего показателя с апреля 2024; они рассчитаны по прошлым трём месяцам той же модели и масштабу 2023 года при предполагаемом нулевом лаге. Нет независимой гарантии покрытия.\n\nОчередь показывает начало эпизодов замороженного EWMA, отдельно по собственному и региональному остатку. Это очередь проверки, а не подтверждённые будущие шоки. Подтверждённые цитаты Qwen показаны отдельно по муниципальной привязке, если она разрешена справочником. Неподтверждённая дата начала события не восстанавливается. Региональные публикации ограничены датой доступности (публикация +1 день) на условную дату получения факта — первое число следующего месяца. Исторический корпус неполон: пустой список не доказывает отсутствие события; региональная новость не доказывает воздействие на каждый МО. Основной прогноз не использует эти новости как признаки.\n"
    )
    return audit


if __name__ == "__main__":
    audit = build()
    print(
        json.dumps(
            {k: v for k, v in audit.items() if not k.endswith("sha256")},
            ensure_ascii=False,
        )
    )
