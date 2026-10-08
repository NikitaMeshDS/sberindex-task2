"""Fixed calendar-year growth bridge; preserves all historical forecast artifacts."""

import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.forecasting.hierarchical_review import fit_profiles
from sberindex.forecasting.full_cohort_review import summarize

OUT = ROOT / "reports/growth_bridge_review"


def year_bridges(origin, horizon):
    if horizon < 1 or int(horizon) != horizon:
        raise ValueError("horizon must be a positive integer")
    return (pd.Period(origin, "M").month - 1 + int(horizon)) // 12


def growth_for_origin(origin, record):
    if pd.Period(origin, "M").end_time.normalize() < pd.Timestamp(
        record["available_from"]
    ):
        raise ValueError("Macro release is unavailable at origin")
    rate = float(record["growth_pct"]) / 100
    if not np.isfinite(rate) or rate <= -1:
        raise ValueError("Growth must be finite and greater than -100%")
    return rate


def bridge_prediction(origin_value, profile, origin_index, horizon, rate):
    if not np.isfinite(rate) or rate <= -1:
        raise ValueError("Invalid growth")
    month = int(origin_index) % 12
    k = (month + int(horizon)) // 12
    return float(
        origin_value
        * profile[(month + horizon) % 12]
        / profile[month]
        * (1 + rate) ** k
    )


def select_forecaster(result, cutoff="2024-06"):
    validation = result[(result.target <= cutoff) & result.horizon.isin([1, 3, 6])]
    mae = (
        validation.groupby(["horizon", "model", "target"])
        .ae.mean()
        .groupby(["horizon", "model"])
        .mean()
        .unstack()
    )
    ratios = mae.T.div(mae["seasonal_pooled"], axis=1)
    ranking = (
        ratios.mean(axis=1)
        .rename("score")
        .reset_index()
        .sort_values(["score", "model"])
    )
    return str(ranking.iloc[0].model)


def main():
    cfg = json.loads((ROOT / "configs/growth_bridge_review.json").read_text())
    source = json.loads((ROOT / cfg["source_record"]).read_text())
    previous = pd.read_parquet(ROOT / "reports/full_cohort_review/predictions.parquet")
    base = previous[previous.model == "seasonal_pooled"].reset_index(drop=True)
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (
        raw[raw.category == cfg["category"]]
        .pivot(index="territory_id", columns="date", values="value")
        .sort_index()
    )
    panel = panel.reindex(
        columns=pd.period_range("2023-01", "2024-12", freq="M").astype(str)
    )
    geo = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    regions = (
        geo[geo.year == cfg["fit_year"]]
        .set_index("territory_id")
        .region_code.reindex(panel.index)
    )
    gp, rp = fit_profiles(panel, regions, cfg["minimum_region_municipalities"])
    # This function ignores every 2024 value when fitting profiles.
    changed = panel.copy()
    changed.iloc[:, 12:] = 1e9
    changed_gp, changed_rp = fit_profiles(
        changed, regions, cfg["minimum_region_municipalities"]
    )
    np.testing.assert_array_equal(gp, changed_gp)
    assert all(np.array_equal(rp[k], changed_rp[k]) for k in rp)
    origin = pd.PeriodIndex(base.origin, freq="M")
    target = pd.PeriodIndex(base.target, freq="M")
    k = (origin.month - 1 + base.horizon.to_numpy()) // 12
    rates = np.array([growth_for_origin(x, source) for x in base.origin])
    weight = cfg["region_weight"]
    profiles = np.stack(
        [
            (1 - weight) * gp + weight * rp.get(regions.loc[i], gp)
            for i in base.territory_id
        ]
    )
    regional = (
        base.origin_actual.to_numpy()
        * profiles[np.arange(len(base)), target.month - 1]
        / profiles[np.arange(len(base)), origin.month - 1]
    )
    frames = [previous.copy()]
    for model, pred in [
        ("hierarchy05", regional),
        ("global_growth_bridge", base.predicted.to_numpy() * (1 + rates) ** k),
        ("hierarchy05_growth_bridge", regional * (1 + rates) ** k),
    ]:
        f = base.copy()
        f["model"] = model
        f["predicted"] = pred
        f["ae"] = abs(f.actual - f.predicted)
        f["yoy_predicted"] = 100 * (f.predicted / f.year_ago - 1)
        f["yoy_ae"] = abs(f.yoy_predicted - f.yoy_actual)
        f["mom_predicted"] = 100 * (f.predicted / f.origin_actual - 1)
        frames.append(f)
    result = pd.concat(frames, ignore_index=True)
    result["year_bridges"] = [
        year_bridges(o, h) for o, h in zip(result.origin, result.horizon)
    ]
    assert len(result) == 8 * len(base)
    assert (
        result.groupby(["territory_id", "origin", "target", "horizon"])
        .model.nunique()
        .eq(8)
        .all()
    )
    assert np.isfinite(result.predicted).all()
    selected = select_forecaster(result)
    selection = dict(
        selected_model=selected,
        validation_end="2024-06",
        evaluation_start="2024-07",
        selection_horizons=[1, 3, 6],
        rule="Equal mean of date-balanced MAE ratios to seasonal_pooled across available validation horizons; ties alphabetic; one model for all four horizons",
        validation_dates_by_horizon={
            str(h): int(g.target.nunique())
            for h, g in result[result.target <= "2024-06"].groupby("horizon")
        },
        independent_holdout=False,
        rule_defined_after_review_of_2024=True,
    )
    summary = summarize(result)
    monthly = (
        result.groupby(["model", "horizon", "target"])
        .agg(MAE=("ae", "mean"), pairs=("ae", "size"))
        .reset_index()
    )
    errors = (
        result.assign(log_error=np.log(result.predicted / result.actual))
        .groupby(["model", "horizon", "year_bridges"])
        .agg(
            pairs=("ae", "size"),
            median_log_error=("log_error", "median"),
            MAE=("ae", "mean"),
        )
        .reset_index()
    )
    OUT.mkdir(exist_ok=True, parents=True)
    result.to_parquet(OUT / "predictions.parquet", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    summarize(result[result.target >= "2024-07"]).to_csv(
        OUT / "late_summary.csv", index=False
    )
    (OUT / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )
    monthly.to_csv(OUT / "monthly.csv", index=False)
    errors.to_csv(OUT / "wrap_bias.csv", index=False)
    paths = [
        "configs/growth_bridge_review.json",
        "docs/protocols/GROWTH_BRIDGE_REVIEW.md",
        cfg["source_record"],
        source["evidence_capture"],
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "reports/full_cohort_review/predictions.parquet",
        "src/sberindex/forecasting/growth_bridge_review.py",
        "src/sberindex/forecasting/hierarchical_review.py",
        "src/sberindex/forecasting/full_cohort_review.py",
    ]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = dict(
        status="passed",
        pairs_per_model=len(base),
        models=8,
        source_growth_pct=source["growth_pct"],
        parameters_tuned=False,
        independent_time_validation=False,
        primary_model_replaced=False,
        future_profile_invariance=True,
        common_pairs=True,
        crossing_origin_count=int(base.loc[k > 0, "origin"].nunique()),
        macro_source_immutability_verified=False,
        input_sha256={p: sha(ROOT / p) for p in paths},
        output_sha256={
            p: sha(OUT / p)
            for p in [
                "predictions.parquet",
                "summary.csv",
                "monthly.csv",
                "wrap_bias.csv",
            ]
        },
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Сезонный профиль с поправкой годового роста",
        "",
        "Один региональный вес 0,5 и одна формула для h1/3/6/12. Национальный рост зарплаты 17,2% из опубликованного до origin доклада Росстата, без подбора по расходам 2024. Результат условен на этот перенос макро-показателя и прежний лаг 0.",
        "",
        "| h | Дат / пар | Сезонный профиль | Региональный 0,5 | Региональный + рост | Лучший проверенный Prophet |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for h, part in summary.groupby("horizon"):
        g = part.set_index("model")
        row = g.loc["hierarchy05_growth_bridge"]
        prophet = part[part.model.str.startswith("prophet")].MAE.min()
        lines.append(
            f"| {h} | {int(row.dates)} / {int(row.observations)} | {g.loc['seasonal_pooled', 'MAE']:.2f} | {g.loc['hierarchy05', 'MAE']:.2f} | {row.MAE:.2f} | {prophet:.2f} |"
        )
    lines += [
        "",
        "## Почему нужна поправка",
        "",
        "Если ряд имеет постоянный лог-тренд и повторяемую сезонность, сырой профиль одного года включает оба компонента. При замыкании календаря в отношении профилей теряется множитель годового роста. k считается числом переходов года, а не только условием target_month < origin_month: для h12 равный месяц также требует k=1. Тест на точной модели тренд × сезонность проверяет формулу.",
        "",
        "Поправка — новая проверяемая модель, не доказанная универсальная ошибка любой сезонной модели: перенос зарплат в расходы с коэффициентом 1 остаётся предположением. 17,2% — опубликованная оценка за октябрь 2023, не прогноз заработной платы и не оракульное g=0,12.",
        "",
        "## Доступность источника",
        "",
        "[Датированный анонс Росстата](https://rosstat.gov.ru/central-news?page=52&per_page=10&print=1) указывает 27.12.2023; [официальный доклад](https://rosstat.gov.ru/storage/mediabank/osn-11-2023.pdf) содержит 17,2%. Сохранены результаты извлечения из поискового индекса официальных страниц. Прямая загрузка документа/страницы завершилась SSL EOF/HTTP502; её сбой сохранён, полный исходный PDF не захвачен. Поэтому неизменность исторического файла не подтверждена. Доступность с 28.12 — консервативная политика publication+1 день, коэффициент затем заморожен.",
        "",
        "Все 60700 ключей на модель совпадают с full_cohort_review; прогнозы Prophet сохранены без изменения. На фактически наблюдаемых парах переход года встречается только из origin декабрь 2023: независимых повторов перехода нет. 2024 уже исследован, гипотеза предложена после его анализа. Статистическая значимость и новый независимый тест не заявляются.",
        "",
        "[Настройки](../../configs/growth_bridge_review.json), [протокол](../../docs/protocols/GROWTH_BRIDGE_REVIEW.md), [полные метрики](summary.csv), [смещение по переходам](wrap_bias.csv).",
        "",
        "Запуск: `PYTHONPATH=src python -m sberindex.forecasting.growth_bridge_review`.",
    ]
    lines += [
        "",
        "## Один выбор прогноза",
        "",
        f"В ретроспективной схеме выбран `{selected}` для всех четырёх горизонтов. Правило: на целях до июня 2024 вычислить для h1/3/6 MAE по датам, разделить на MAE сезонного контроля и выбрать модель с минимальным средним этих трёх отношений; h12 не участвует, поскольку ранних целей нет. Поздние цели июля–декабря отдельно сохранены в late_summary.csv. На h6 в ранней части всего одна дата: это ограничение отбора.",
        "",
        "Правило введено после изучения 2024 и второго отзыва, поэтому поздняя часть не названа независимым тестом. Тест инвариантности подтверждает, что функция выбора не читает её ошибки. Исторический operational выпуск не переписан новой моделью; выбранный исследовательский алгоритм имеет собственные конфигурацию, прогнозы и команду.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(summary[["horizon", "model", "MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
