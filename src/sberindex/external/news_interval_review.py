"""Event-conditioned interval diagnostic; no point-model change or tuning."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from sberindex.detection.event_conditioned_review import registry_events
from sberindex.external.news_forecast_diagnostic_review import known_at_origin
from sberindex.paths import ROOT

OUT = ROOT / "reports/news_interval_review"
INPUTS = [
    "configs/news_interval_review.json",
    "src/sberindex/external/news_interval_review.py",
    "src/sberindex/external/news_forecast_diagnostic_review.py",
    "src/sberindex/detection/event_conditioned_review.py",
    "reports/growth_bridge_review/predictions.parquet",
    "reports/presentation_comparison_cases/calibration.csv",
    "data/consumption.parquet",
    "data/external/regional_news_review/registry.csv",
    "reports/news_event_extraction_review/forecast_news_availability.csv",
]
KEYS = ["territory_id", "origin", "target"]


def event_intervals(predicted, radius, known, factor):
    predicted = np.asarray(predicted, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if factor < 1 or np.any(radius < 0):
        raise ValueError("Invalid widening factor or negative radius")
    multiplier = np.where(np.asarray(known, dtype=bool), factor, 1.0)
    return np.maximum(
        0.0, predicted - radius * multiplier
    ), predicted + radius * multiplier


def interval_score(actual, lower, upper, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    actual, lower, upper = (np.asarray(x, dtype=float) for x in (actual, lower, upper))
    if np.any(upper < lower):
        raise ValueError("Reversed interval")
    return (
        upper
        - lower
        + 2 / alpha * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
    )


def validate_calibrations(calibration, coverage):
    if (calibration.max_calibration_target > calibration.origin).any() or (
        calibration.calibration_end > calibration.origin
    ).any():
        raise ValueError("Calibration includes future target")
    if not np.allclose(calibration.nominal_coverage, coverage):
        raise ValueError("Nominal interval level mismatch")


def run():
    config = json.loads((ROOT / INPUTS[0]).read_text())
    if (
        config["model"] != "hierarchy05_growth_bridge"
        or config["horizon"] != 1
        or config["release_lag_months"] != 0
    ):
        raise ValueError(
            "Saved calibration only supports selected h1 model and lag0 scenario"
        )
    predictions = pd.read_parquet(ROOT / INPUTS[4])
    predictions = predictions[
        predictions.model.eq(config["model"])
        & predictions.horizon.eq(config["horizon"])
    ].copy()
    calibration = pd.read_csv(ROOT / INPUTS[5])
    validate_calibrations(calibration, config["nominal_coverage"])
    if calibration.origin.duplicated().any():
        raise ValueError("Duplicate calibration origins")
    events = registry_events()
    predictions["known_news"] = [
        known_at_origin(events, int(row.territory_id), row.origin)
        for row in predictions.itertuples()
    ]
    original = pd.read_csv(ROOT / INPUTS[-1])
    keys = predictions[KEYS + ["known_news"]].merge(
        original, on=KEYS, suffixes=("_new", "_old"), validate="one_to_one"
    )
    assert len(keys) == len(predictions)
    assert keys.known_news_new.equals(keys.known_news_old), (
        "Changed news forecast pairs"
    )
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    scale = (
        raw[raw.category.eq("Все категории") & raw.date.between("2023-01", "2023-12")]
        .groupby("territory_id")
        .value.mean()
    )
    predictions = predictions.merge(
        calibration, on="origin", how="left", validate="many_to_one"
    )
    predictions["radius"] = (
        predictions.normalized_radius * predictions.territory_id.map(scale)
    )
    valid = predictions.radius.notna() & predictions.radius.gt(0)
    panel = predictions.loc[valid].copy()
    assert len(panel[panel.known_news]) == int(original.known_news.sum()), (
        "Missing news interval pairs"
    )
    alpha = 1 - config["nominal_coverage"]
    summaries, details = [], []
    for policy, factor in [
        ("baseline", 1.0),
        (
            "news_radius_x"
            + format(config["event_radius_factor"], "g").replace(".", "_"),
            config["event_radius_factor"],
        ),
    ]:
        lower, upper = event_intervals(
            panel.predicted, panel.radius, panel.known_news, factor
        )
        frame = panel[
            KEYS + ["actual", "predicted", "known_news", "max_calibration_target"]
        ].copy()
        frame["policy"] = policy
        frame["lower"], frame["upper"] = lower, upper
        frame["covered"] = frame.actual.between(frame.lower, frame.upper)
        frame["width"] = upper - lower
        frame["interval_score"] = interval_score(frame.actual, lower, upper, alpha)
        details.append(frame)
        for scope, group in [
            ("entire_valid_panel", frame),
            ("available_news_subset", frame[frame.known_news]),
        ]:
            summaries.append(
                {
                    "policy": policy,
                    "scope": scope,
                    "pairs": len(group),
                    "dates": group.target.nunique(),
                    "municipalities": group.territory_id.nunique(),
                    "covered": int(group.covered.sum()),
                    "coverage_pct": 100 * group.covered.mean(),
                    "mean_width": float(group.width.mean()),
                    "mean_interval_score": float(group.interval_score.mean()),
                    "date_balanced_interval_score": float(
                        group.groupby("target").interval_score.mean().mean()
                    ),
                    "MAE_date_balanced": float(
                        (group.actual - group.predicted)
                        .abs()
                        .groupby(group.target)
                        .mean()
                        .mean()
                    ),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    detail = pd.concat(details, ignore_index=True)
    detail.to_parquet(OUT / "interval_panel.parquet", index=False)
    detail[detail.known_news].to_csv(OUT / "news_pairs.csv", index=False)
    audit = {
        "status": "passed",
        "paired_news_rows": int(original.known_news.sum()),
        "valid_pairs": int(valid.sum()),
        "missing_base_interval_pairs": int((~valid).sum()),
        "missing_news_interval_pairs": int((predictions.known_news & ~valid).sum()),
        "point_forecasts_unchanged": True,
        "factor": config["event_radius_factor"],
        "posthoc_diagnostic": True,
        "independent_coverage_guarantee": False,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in INPUTS
        },
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    report = "# Новости как сигнал неопределённости прогноза\n\n"
    factor_label = format(config["event_radius_factor"], "g").replace(".", ",")
    report += f"Точечный прогноз не меняется. Если официальное сообщение с явно названным МО известно на дату прогноза, радиус интервала умножается на {factor_label}. Это фиксированная гипотеза управления риском, предложенная после изучения архива 2024; множитель не оценён по данным и не выбран по метрикам.\n\n"
    report += "| Политика | Выборка | Пар | Попаданий | Покрытие,% | Ширина,руб. | IS,руб. |\n|---|---|---:|---:|---:|---:|---:|\n"
    for row in summaries:
        report += f"| {row['policy']} | {row['scope']} | {row['pairs']} | {row['covered']} | {row['coverage_pct']:.2f} | {row['mean_width']:.2f} | {row['mean_interval_score']:.2f} |\n"
    report += "\n"
    report += "Покрытие: доля фактов внутри границ. Ширина и интервальный штраф: рубли на жителя, меньше лучше. Штраф IS₀.₂ = (U−L) +10·max(L−y,0)+10·max(y−U,0) учитывает ширину и промахи; [определение](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618), §2.2. Покрытие и ширина взвешены по парам; штраф также сохранён с равными весами дат.\n\n"
    report += "Исходные иллюстративные 80% границы используют ошибки выбранной модели за три завершённых месяца до цели, масштаб расходов 2023 и сценарий выпуска лаг 0. Январь–март исключены из интервального сравнения; все 18 прежних новостных пар имеют границы. Удвоенная таблица news_pairs.csv позволяет проверить каждую пару. Нижнюю границу ограничиваем нулём.\n\n"
    report += "Расширение вложенного интервала механически не уменьшает покрытие и увеличивает ширину. Это само по себе не доказывает полезность новостей: для неё нужен выигрыш штрафа или оценка решений аналитика на независимом периоде. Корпус выборочный, отсутствие записи не является отсутствием события; историческая версия публикаций не подтверждена. Использован исходный официальный реестр, не расширенный Qwen-корпус из другого опыта.\n\n"
    report += "Запуск: `python -m sberindex.external.news_interval_review`. Конфигурация: configs/news_interval_review.json; входы и их SHA256 в audit.json. Старые фиксированные сдвиги ±3% сохранены как отрицательный опыт в reports/news_event_extraction_review/FORECAST_DIAGNOSTIC.md.\n"
    before, after = [r for r in summaries if r["scope"] == "available_news_subset"]
    report += f"\n**Результат:** покрытие {before['covered']}/{before['pairs']} → {after['covered']}/{after['pairs']}; ширина {before['mean_width']:.2f} → {after['mean_width']:.2f} руб.; интервальный штраф {before['mean_interval_score']:.2f} → {after['mean_interval_score']:.2f} руб. Множитель не включается в основной прогноз: независимая польза не установлена.\n"
    (OUT / "REPORT.md").write_text(report)
    print(json.dumps({k: v for k, v in audit.items() if k != "source_sha256"}))
    print(pd.DataFrame(summaries).to_string(index=False))
    return summaries


if __name__ == "__main__":
    run()
