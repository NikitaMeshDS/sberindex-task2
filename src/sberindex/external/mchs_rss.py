"""Public official RSS collector with explicit date coverage; never manufactures history."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import pandas as pd
from sberindex.paths import ROOT

OUT = ROOT / "data/external/mchs_rss"


def discover_feeds(html, page_url):
    result = []

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            href = attributes.get("href", "")
            if tag in ("a", "link") and (
                "rss" in href.lower()
                or attributes.get("type")
                in ("application/rss+xml", "application/atom+xml")
            ):
                url = urljoin(page_url, href)
                if (
                    urlparse(url).hostname == urlparse(page_url).hostname
                    and url not in result
                ):
                    result.append(url)

    parser = Links()
    parser.feed(html)
    return result


def parse_feed(data, url, start, end):
    root = ET.fromstring(data)
    records, seen = [], set()
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        fields = {x.tag.rsplit("}", 1)[-1]: x for x in item}
        title = fields.get("title")
        link = fields.get("link")
        publication = fields.get("pubDate")
        if publication is None:
            publication = fields.get("published")
        # updated dates are not original publication dates.
        if link is None or publication is None or not publication.text:
            continue
        source_url = link.attrib.get("href") or link.text
        if not source_url or urlparse(source_url).hostname != urlparse(url).hostname:
            continue
        value = publication.text.strip()
        try:
            stamp = (
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                if "T" in value
                else parsedate_to_datetime(value)
            )
        except (ValueError, TypeError):
            continue
        day = stamp.date().isoformat()
        if start <= day <= end and source_url not in seen:
            seen.add(source_url)
            records.append(
                dict(
                    title="".join(title.itertext()) if title is not None else "",
                    source_url=source_url,
                    published_date=day,
                    published_timestamp=value,
                    feed_url=url,
                    historical_vintage_verified=False,
                    territory_mapping_status="pending_manual_review",
                )
            )
    return records


def fetch(url, path, timeout):
    request = Request(
        url, headers={"User-Agent": "SberIndex reproducible public-source research"}
    )
    with urlopen(request, timeout=timeout) as response:
        data = response.read(5_000_000)
        path.write_bytes(data)
        return data


def collect_region(region, args):
    page = f"https://{region:02}.mchs.gov.ru/deyatelnost/press-centr/novosti"
    log, records = [], []
    try:
        raw_path = OUT / f"{region:02}_listing.html"
        raw = fetch(page, raw_path, args.timeout)
        feeds = discover_feeds(raw.decode("utf-8", errors="replace"), page)
        log.append(
            dict(
                region_code=region,
                url=page,
                status="captured_listing",
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    except Exception as error:
        feeds = (
            ["https://56.mchs.gov.ru/deyatelnost/press-centr/novosti/rss"]
            if region == 56
            else []
        )
        log.append(
            dict(
                region_code=region,
                url=page,
                status="fetch_failed",
                error=f"{type(error).__name__}: {error}",
            )
        )
    for index, feed in enumerate(feeds):
        try:
            path = OUT / f"{region:02}_feed_{index}.xml"
            data = fetch(feed, path, args.timeout)
            retained = parse_feed(data, feed, args.start, args.end)
            records.extend(dict(**r, region_code=region) for r in retained)
            log.append(
                dict(
                    region_code=region,
                    url=feed,
                    status="captured_feed",
                    historical_retained_items=len(retained),
                    sha256=hashlib.sha256(data).hexdigest(),
                    historical_completeness_verified=False,
                )
            )
        except Exception as error:
            log.append(
                dict(
                    region_code=region,
                    url=feed,
                    status="fetch_failed",
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return records, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--timeout", type=float, default=8)
    args = parser.parse_args()
    if args.start > args.end:
        raise ValueError("Invalid date interval")
    OUT.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(ROOT / "data/external/regional_news_review/registry.csv")
    regions = sorted(registry.region_code.unique().astype(int).tolist())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda region: collect_region(region, args), regions))
    records = [row for part, _ in results for row in part]
    requests = [row for _, part in results for row in part]
    payload = dict(
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        interval=[args.start, args.end],
        regions=regions,
        requests=requests,
        records=records,
        historical_complete=False,
        status="historical_items_collected_unverified_coverage"
        if records
        else "historical_collection_unavailable",
        known_feed_discovery_source="https://56.mchs.gov.ru/deyatelnost/press-centr/novosti",
        code_sha256={
            "src/sberindex/external/mchs_rss.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
    )
    (OUT / "collection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    pd.DataFrame(
        records,
        columns=[
            "title",
            "source_url",
            "published_date",
            "published_timestamp",
            "feed_url",
            "historical_vintage_verified",
            "territory_mapping_status",
            "region_code",
        ],
    ).to_csv(OUT / "historical_items.csv", index=False)
    report = ROOT / "reports/mchs_rss"
    report.mkdir(parents=True, exist_ok=True)
    (report / "REPORT.md").write_text(
        "# Исторические региональные RSS МЧС\n\n"
        f"Проверены {len(regions)} регионов; сохранено {len(records)} сообщений с явной датой публикации внутри 2023–2024. Состояние: `{payload['status']}`.\n\n"
        "Реальные запросы и ошибки сохранены в data/external/mchs_rss/collection.json. На сайте Оренбургского МЧС ссылка RSS ведёт на `/deyatelnost/press-centr/novosti/rss`. Остальные адреса обнаруживаются из официальных страниц. Текущая RSS лента не подтверждает полноту архива за прошлые годы; отсутствие исторических записей не означает отсутствие событий. Сбор не подменяет текущими публикациями новости 2023–2024.\n\n"
        "Команда для повторения сетевого сбора: `PYTHONPATH=src python -m sberindex.external.mchs_rss`. Она требует доступа к сайтам и изменяет capture timestamp. Численные опыты используют закреплённый региональный реестр, поэтому сетевой сбор не запускается автоматически при воспроизведении метрик. Полученные новые документы требуют проверки географии и реальной доступности перед включением в модель.\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "regions": len(regions),
                "records": len(records),
                "failed_requests": sum(r["status"] == "fetch_failed" for r in requests),
            }
        )
    )


if __name__ == "__main__":
    main()
