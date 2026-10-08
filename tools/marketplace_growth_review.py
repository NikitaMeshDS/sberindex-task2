"""One source-fixed marketplace rate; diagnostic only, never changes selection."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/marketplace_growth_review"


def main():
    source = json.loads(
        (ROOT / "data/external/marketplace_growth_review/source.json").read_text()
    )
    frame = pd.read_parquet(
        ROOT / "reports/category_growth_review/full_pooled_predictions.parquet"
    )
    frame = frame[frame.category == "Маркетплейсы"].copy()
    base = frame[frame.model == "hierarchy05"].copy()
    rate = source["growth_rate"]
    origin = pd.PeriodIndex(base.origin, freq="M")
    if (origin.end_time.normalize() < pd.Timestamp(source["available_from"])).any():
        raise ValueError("Source unavailable at an origin")
    k = (origin.month - 1 + base.horizon.to_numpy()) // 12
    base["predicted"] *= (1 + rate) ** k
    base["model"] = "hierarchy05_akit23_diagnostic"
    base["ae"] = abs(base.actual - base.predicted)
    allrows = pd.concat([frame, base], ignore_index=True)
    summary = (
        allrows.groupby(["model", "horizon", "target"])
        .ae.mean()
        .groupby(["model", "horizon"])
        .mean()
        .rename("MAE")
        .reset_index()
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    base.to_parquet(OUT / "predictions.parquet", index=False)
    rows = []
    for h in [1, 3, 6, 12]:
        t = summary[summary.horizon == h].set_index("model").MAE
        rows.append(
            f"| {h} | {t['hierarchy05']:.1f} | {t['hierarchy05_growth_bridge']:.1f} | {t['hierarchy05_akit23_diagnostic']:.1f} | {t['prophet_pooled_profile']:.1f} |"
        )
    (OUT / "REPORT.md").write_text(
        "# Категорийный рост маркетплейсов: отдельная диагностика\n\nАКИТ сообщила 7 ноября 2023 о росте всей интернет-торговли за девять месяцев на 23%; доступность принята с 8 ноября. Фиксируем единственную ставку 23%, без поиска по ошибкам. Это отраслевой первичный источник, не государственная статистика; интернет-торговля шире категории маркетплейсов СберИндекса. Опыт предложен после просмотра 2024, поэтому не заменяет выбранную модель и не является независимым подтверждением.\n\n| h | g=0 | g=17,2% | АКИТ g=23% | Prophet + профиль |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\nЕдиницы — руб./жителя; MAE сначала по МО внутри даты, затем по датам. Все ставки сравниваются на точных сохранённых парах; профили, вес 0,5 и контроль Prophet неизменны. Исходная опубликованная ставка не является оценкой роста муниципальных расходов. [Первичный источник](https://www.akit.ru/news/obyom-internet-torgovli-v-rossii-po-itogam-9-mesyatsev-2023-goda).\n"
    )
    files = [
        "tools/marketplace_growth_review.py",
        "data/external/marketplace_growth_review/source.json",
        "data/external/marketplace_growth_review/source.html",
        "reports/category_growth_review/full_pooled_predictions.parquet",
    ]
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "primary_model_changed": False,
                "independent_test": False,
                "pairs": len(base),
                "source_rate": rate,
                "input_sha256": {
                    f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                    for f in files
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
