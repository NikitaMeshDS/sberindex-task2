"""Rebuild a partial official archive from captured search-index responses.

Search omission is unknown coverage, never a negative news label.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from sberindex.paths import ROOT

DATA = ROOT / "data/external/mchs_archive_review"
OUT = ROOT / "reports/mchs_archive_review"
REGIONS = [10, 21, 23, 25, 31, 45, 46, 54, 56, 61, 66, 72, 74]
MONTHS = dict(
    zip(
        "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(),
        range(1, 13),
    )
)
DATE = re.compile(r"(?m)^(\d{1,2}) (\w+) (20\d\d), (\d\d:\d\d)\s*$")


def forecast_date(title):
    if "прогноз" not in title.lower():
        return None
    russian = re.search(r"(?i)\bна\s+(\d{1,2})\s+(\w+)\s+(20\d\d)", title)
    numeric = re.search(r"(?i)\bна\s+(\d{1,2})\.(\d{1,2})\.(20\d\d)", title)
    match = russian or numeric
    if not match:
        monthly = re.search(
            r"(?i)\bна\s+(январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)\s+(?:месяц\s+)?(20\d\d)",
            title,
        )
        if monthly:
            month_names = "январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь".split()
            return (
                datetime(int(monthly[2]), month_names.index(monthly[1].lower()) + 1, 1)
                .date()
                .isoformat()
            )
        return None
    day, month, year = match.groups()
    try:
        month = MONTHS[month.lower()] if russian else int(month)
        return datetime(int(year), month, int(day)).date().isoformat()
    except (ValueError, KeyError):
        return None


def monthly_early(available, valid):
    if not valid:
        return None
    return pd.Timestamp(available) < pd.Timestamp(valid).replace(day=1)


def parse_capture(text, path):
    records, exclusions = [], []
    for block in text.split(
        "--------------------------------------------------------------------------------"
    ):
        block = block.strip()
        if not block:
            continue
        header = block.splitlines()[0]
        url_match = re.search(r"\((https://[^\s)]+)\)", header)
        if not url_match:
            exclusions.append(
                dict(snapshot_path=path, source_url="", reason="no_source_url")
            )
            continue
        url = url_match.group(1)
        host = urlparse(url).hostname or ""
        region = re.fullmatch(r"(\d{2})\.mchs\.gov\.ru", host)
        if not region:
            exclusions.append(
                dict(
                    snapshot_path=path,
                    source_url=url,
                    reason="not_official_regional_host",
                )
            )
            continue
        # Only publication metadata BEFORE the page heading, not body/navigation dates.
        prefix = re.split(r"(?m)^# ", block, maxsplit=1)[0]
        stamp = DATE.search(prefix)
        if not stamp:
            exclusions.append(
                dict(
                    snapshot_path=path,
                    source_url=url,
                    reason="no_explicit_header_publication",
                )
            )
            continue
        day, month, year, clock = stamp.groups()
        try:
            publication = datetime.strptime(
                f"{year}-{MONTHS[month]}-{day} {clock}", "%Y-%m-%d %H:%M"
            )
        except (ValueError, KeyError):
            exclusions.append(
                dict(
                    snapshot_path=path,
                    source_url=url,
                    reason="invalid_publication_date",
                )
            )
            continue
        if publication.year not in (2023, 2024):
            exclusions.append(
                dict(
                    snapshot_path=path,
                    source_url=url,
                    reason="publication_outside_history",
                )
            )
            continue
        title = header[: url_match.start()].strip()
        valid = forecast_date(title)
        available = (publication + timedelta(days=1)).date().isoformat()
        records.append(
            dict(
                source_url=url,
                title=title,
                region_code=int(region[1]),
                published_date=publication.date().isoformat(),
                published_clock=clock,
                publication_timezone="unknown",
                available_from_scenario=available,
                forecast_date=valid,
                before_forecast_day_scenario=(available < valid if valid else None),
                before_forecast_month_scenario=monthly_early(available, valid),
                document_role=("dated_forecast" if valid else "other_or_unparsed"),
                forecast_time_precision=(
                    "month"
                    if re.search(r"(?i)месяц", title) and valid
                    else "day"
                    if valid
                    else None
                ),
                hazard_warning_verified=False,
                historical_vintage_verified=False,
                source_transport="official_search_index_capture",
                coverage_status="partial_search_selected",
                snapshot_path=path,
                snapshot_sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                if (ROOT / path).is_file()
                else "",
            )
        )
    return records, exclusions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, rejected, inputs = [], [], []
    for path in sorted(DATA.glob("*.json")):
        response = json.loads(path.read_text())
        if not isinstance(response, str):
            continue
        records, excluded = parse_capture(response, str(path.relative_to(ROOT)))
        rows.extend(records)
        rejected.extend(excluded)
        inputs.append(
            dict(
                path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    frame = pd.DataFrame(rows)
    conflicts = frame.groupby("source_url").published_date.nunique()
    conflict_urls = set(conflicts[conflicts > 1].index)
    for url in sorted(conflict_urls):
        rejected.append(
            dict(
                source_url=url, snapshot_path="", reason="conflicting_publication_dates"
            )
        )
    frame = (
        frame[~frame.source_url.isin(conflict_urls)]
        .drop_duplicates("source_url")
        .sort_values(["published_date", "source_url"])
    )
    frame.to_csv(DATA / "historical_items.csv", index=False)
    pd.DataFrame(rejected).to_csv(OUT / "exclusions.csv", index=False)
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    lookup = lookup[lookup.year == 2024][
        ["territory_id", "region_code"]
    ].drop_duplicates()
    forecasts = pd.read_parquet(
        ROOT / "reports/growth_bridge_review/predictions.parquet"
    )
    forecasts = forecasts[
        (forecasts.model == "hierarchy05_growth_bridge") & (forecasts.horizon == 1)
    ]
    forecasts["target_month"] = (
        pd.to_datetime(forecasts.target).dt.to_period("M").astype(str)
    )
    joined = []
    for item in frame.to_dict("records"):
        if not item["forecast_date"] or pd.isna(item["forecast_date"]):
            continue
        valid = pd.Timestamp(item["forecast_date"])
        ids = lookup.loc[lookup.region_code == item["region_code"], "territory_id"]
        eligible = forecasts[
            (forecasts.territory_id.isin(ids))
            & (forecasts.target_month == str(valid.to_period("M")))
        ]
        joined.append(
            dict(
                source_url=item["source_url"],
                region_code=item["region_code"],
                available_from_scenario=item["available_from_scenario"],
                forecast_date=item["forecast_date"],
                before_target_month=monthly_early(
                    item["available_from_scenario"], item["forecast_date"]
                ),
                regional_lookup_ids=len(ids),
                expense_forecast_ids=eligible.territory_id.nunique(),
                target_month=str(valid.to_period("M")),
                hazard_warning_verified=False,
                use_for_shock_prediction=False,
                exposure_scope="regional_not_confirmed_municipal_impact",
            )
        )
    joined = pd.DataFrame(joined)
    joined.to_csv(OUT / "forecast_expense_alignment.csv", index=False)
    annotations = json.loads((DATA / "verified_warning_annotations.json").read_text())
    warning_rows = []
    for item in annotations["warnings"]:
        publication = pd.Timestamp(item["published_date"])
        available = publication + pd.Timedelta(days=1)
        start = pd.Timestamp(item["forecast_start"])
        warning_rows.append(
            dict(
                **item,
                available_from_scenario=available.date().isoformat(),
                publication_to_forecast_days=(start - publication).days,
                conservative_advance_days=(start - available).days,
                available_before_month_start=monthly_early(available, start),
                expenditure_shock_verified=False,
                independent_validation=False,
                snapshot_path=annotations["snapshot_path"],
            )
        )
    pd.DataFrame(warning_rows).to_csv(OUT / "warning_timing.csv", index=False)
    coverage = []
    for region in REGIONS:
        subset = frame[frame.region_code == region]
        coverage.append(
            dict(
                region_code=region,
                documents=len(subset),
                publication_days=subset.published_date.nunique(),
                first_publication=subset.published_date.min(),
                last_publication=subset.published_date.max(),
                full_history_verified=False,
                missing_days_mean="unknown_not_no_news",
            )
        )
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    months = pd.period_range("2023-01", "2024-12", freq="M").astype(str)
    grid = np.zeros((len(REGIONS), len(months)))
    for i, region in enumerate(REGIONS):
        counts = (
            frame.loc[frame.region_code == region, "published_date"]
            .str[:7]
            .value_counts()
        )
        for j, month in enumerate(months):
            grid[i, j] = counts.get(month, 0)
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = ListedColormap(["#eeeeee", "#b5d5e9", "#6fa4c8", "#27628c", "#173e5c"])
    ax.imshow(np.minimum(grid, 4), cmap=cmap, vmin=0, vmax=4, aspect="auto")
    for i, j in zip(*np.where(grid > 0)):
        ax.text(
            j,
            i,
            str(int(grid[i, j])),
            ha="center",
            va="center",
            color="white" if grid[i, j] >= 3 else "#163447",
            fontsize=9,
        )
    ax.set_xticks(range(len(months)), [m[2:] for m in months], rotation=45, ha="right")
    ax.set_yticks(range(len(REGIONS)), [f"Регион {r:02}" for r in REGIONS])
    ax.set_title(
        "Частичный исторический корпус МЧС: публикации по месяцам",
        loc="left",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel(
        "Серый: покрытие неизвестно, а не отсутствие новостей. Число: найденные публикации."
    )
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "coverage.png", dpi=180)
    fig.savefig(OUT / "coverage.svg")
    plt.close(fig)
    audit = dict(
        status="complete",
        completion_scope="partial_archive_processing_not_full_history_or_shock_validation",
        manually_verified_warning_documents=len(warning_rows),
        documents=len(frame),
        regions=int(frame.region_code.nunique()),
        dated_forecasts=int(frame.forecast_date.notna().sum()),
        forecasts_before_target_month=int(joined.before_target_month.sum()),
        early_forecasts_with_expense_targets=int(
            ((joined.before_target_month) & (joined.expense_forecast_ids > 0)).sum()
        ),
        all_documents_are_verified_hazard_warnings=False,
        full_rss_history_recovered=False,
        historical_vintage_verified=False,
        independent_real_shock_prediction_proven=False,
        publication_availability="next_calendar_day_scenario_timezone_unknown",
        inputs=inputs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    (OUT / "REPORT.md").write_text(
        f"""# Частичный исторический архив МЧС

Восстановлено **{len(frame)}** уникальных официальных публикаций 2023–2024 годов из **{audit["regions"]}** регионов. Сохранены исходные ответы поискового индекса, хеши, даты публикации, исключения и региональное покрытие. Повторный запуск не требует сети:

```bash
PYTHONPATH=src python -m sberindex.external.mchs_archive
```

Это резервный источник после сетевого отказа RSS. Полный исторический RSS-корпус **не восстановлен**. Выборка найдена тематическими запросами, отсутствующие дни имеют неизвестное покрытие. Текущие публикации с упоминанием 2024 года исключены по дате публикации. Заголовок прогноза не заменяет её.

## Проверка времени и согласования

Из заголовков выделены {audit["dated_forecasts"]} прогнозов с полной датой действия. При сценарии доступности на следующий день {audit["forecasts_before_target_month"]} опубликованы до начала месяца действия; {audit["early_forecasts_with_expense_targets"]} имеют сопоставимые месячные прогнозы расходов в региональной выборке. В `forecast_expense_alignment.csv` показаны все случаи, включая отсутствие целевых фактов. Региональная экспозиция не равна воздействию на каждый МО.

**Обычный ежедневный прогноз не является предупреждением о шоке.** Автоматически считать эти строки положительными предсказаниями нельзя. Срок действия предупреждения, начало реального события и структурный сдвиг расходов требуют раздельных меток. Публикации внутри месяца не предсказывают начало этого месяца.

## Ручная проверка предупреждений

Отдельно сохранены две официальные страницы Приморского края и ручная разметка срока ожидаемых сильных дождей: публикация 9 августа → прогноз 10–12 августа 2023; публикация 23 августа → прогноз 25 августа. В консервативном сценарии следующего дня запас составляет 0 и 1 календарный день соответственно. Это предупреждения о дальнейшем ухудшении при уже существовавшем паводке, а не доказательства предсказания начала наводнения. Оба опубликованы внутри августа и не предсказывают начало августовского месяца расходов. Фактическое начало отдельного погодного события и сдвиг расходов этими страницами не размечены. Подробности — `warning_timing.csv`.

## Что доказано и что остаётся

Доказана воспроизводимость восстановления частичных публикаций и проверки календарных ограничений. Не доказаны полнота новостного наблюдения, историческая неизменность страниц и предсказание независимых будущих сдвигов расходов. Recall и доля ложных предупреждений по этому поисковому корпусу не вычисляются: неполные отрицательные примеры сделали бы их недостоверными. Исходный журнал отказа RSS сохранён. Правила — `docs/protocols/MCHS_ARCHIVE_REVIEW.md`.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k != "inputs"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
