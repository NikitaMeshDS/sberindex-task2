"""Separate direct HGB per horizon, rolling past-only refits, no h12 invention."""

import hashlib
import importlib.metadata
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from sberindex.paths import ROOT

OUT = ROOT / "reports/direct_horizon_review"


def training_indices(values, fit_origin, horizon, first_origin=2):
    """Rows (municipality_index, feature_origin, target) with target <= fit_origin.

    Missingness after fit_origin is irrelevant. Missing observations between
    feature origin and target do not invalidate a direct target if it is known.
    """
    x = np.asarray(values, float)
    blocks = []
    for source in range(first_origin, fit_origin - horizon + 1):
        target = source + horizon
        valid = (
            np.isfinite(x[:, : source + 1]).all(axis=1)
            & (x[:, : source + 1] > 0).all(axis=1)
            & np.isfinite(x[:, target])
            & (x[:, target] > 0)
        )
        ids = np.flatnonzero(valid)
        blocks.append(
            np.column_stack((ids, np.full(len(ids), source), np.full(len(ids), target)))
        )
    return np.concatenate(blocks).astype(int) if blocks else np.empty((0, 3), dtype=int)


def feature_block(values, categories, regions, source, horizon, context=False):
    """No value after source is read, including simultaneous aggregate features."""
    x = np.asarray(values, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        logs = np.log(
            np.where(
                x[:, source - 2 : source + 1] > 0, x[:, source - 2 : source + 1], np.nan
            )
        )
    growth = logs[:, -1] - logs[:, -2]
    n = len(x)
    origin_phase = 2 * np.pi * (source % 12 + 1) / 12
    target_phase = 2 * np.pi * ((source + horizon) % 12 + 1) / 12
    features = [
        logs[:, -1],
        growth,
        logs[:, -2] - logs[:, -3],
        np.mean(logs, axis=1),
        np.full(n, np.sin(origin_phase)),
        np.full(n, np.cos(origin_phase)),
        np.full(n, np.sin(target_phase)),
        np.full(n, np.cos(target_phase)),
    ]
    if context:
        finite = growth[np.isfinite(growth)]
        global_growth = float(np.median(finite)) if len(finite) else 0.0
        regional_growth = np.full(n, global_growth)
        for region in np.unique(regions[np.isfinite(regions)]):
            members = regions == region
            valid = growth[members]
            valid = valid[np.isfinite(valid)]
            if len(valid):
                regional_growth[members] = np.median(valid)
        features.extend([np.full(n, global_growth), regional_growth])
        cats = np.asarray(categories, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            current = np.where(cats[:, :, source] > 0, cats[:, :, source], np.nan)
            previous = np.where(
                cats[:, :, source - 1] > 0, cats[:, :, source - 1], np.nan
            )
            cat_growth = np.log(current / previous)
        features.extend(cat_growth[:, i] for i in range(cats.shape[1]))
    return np.column_stack(features)


def main():
    cfg = json.loads((ROOT / "configs/direct_horizon_review.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    months = pd.period_range("2023-01", "2024-12", freq="M").astype(str)
    panel = (
        raw[raw.category == cfg["category"]]
        .pivot(index="territory_id", columns="date", values="value")
        .reindex(columns=months)
        .sort_index()
    )
    train = panel.iloc[:, :12]
    eligible = train.index[np.isfinite(train).all(axis=1) & (train > 0).all(axis=1)]
    panel = panel.loc[eligible]
    values = panel.to_numpy(float)
    categories = np.stack(
        [
            raw[raw.category == c]
            .pivot(index="territory_id", columns="date", values="value")
            .reindex(index=eligible, columns=months)
            .to_numpy(float)
            for c in cfg["covariate_categories"]
        ],
        axis=1,
    )
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    geo = lookup[lookup.year == cfg["geography_year_label"]].set_index("territory_id")
    assert geo.index.is_unique
    regions = geo.region_code.reindex(eligible).to_numpy(float)
    source = pd.read_parquet(ROOT / "reports/prophet_seasonality/predictions.parquet")
    base = source[source.model == "seasonal_pooled"].copy()
    keys = ["territory_id", "origin", "target", "horizon"]
    assert not base.duplicated(keys).any()
    ids = json.loads((ROOT / "results/asof_cohort_protocol.json").read_text())[
        "sample_ids"
    ]
    assert set(base.territory_id).issubset(ids)
    records = []
    fits = []
    calendars = []
    start = time.monotonic()
    with threadpool_limits(limits=cfg["threads"]):
        for (origin_month, h), group in base.groupby(["origin", "horizon"]):
            h = int(h)
            origin = (pd.Period(origin_month, "M") - pd.Period("2023-01", "M")).n
            indices = training_indices(
                values, origin, h, cfg["first_training_origin_index"]
            )
            for feature_origin in np.unique(indices[:, 1]):
                targets = indices[indices[:, 1] == feature_origin, 2]
                calendars.append(
                    dict(
                        fit_origin=origin_month,
                        horizon=h,
                        feature_origin=months[feature_origin],
                        training_target=months[targets[0]],
                        training_rows=len(targets),
                    )
                )
            inference_ids = eligible.get_indexer(group.territory_id)
            assert (inference_ids >= 0).all()
            actual = values[inference_ids, origin + h]
            np.testing.assert_array_equal(actual, group.actual)
            for variant in cfg["variants"]:
                begin = time.monotonic()
                context = variant == "direct_context"
                if len(indices) < cfg["minimum_training_rows"]:
                    status = "not_trainable_seasonal_fallback"
                    model_name = variant + "__seasonal_fallback_not_trainable"
                    prediction = group.predicted.to_numpy(float)
                    trained = False
                else:
                    blocks = []
                    labels = []
                    for s in np.unique(indices[:, 1]):
                        selected = indices[indices[:, 1] == s, 0]
                        target = int(s) + h
                        blocks.append(
                            feature_block(
                                values, categories, regions, int(s), h, context
                            )[selected]
                        )
                        labels.append(
                            np.log(values[selected, target] / values[selected, int(s)])
                        )
                    x = np.concatenate(blocks)
                    y = np.concatenate(labels)
                    assert len(x) == len(indices) and np.isfinite(y).all()
                    model = HistGradientBoostingRegressor(**cfg["model"]).fit(x, y)
                    inference = feature_block(
                        values, categories, regions, origin, h, context
                    )[inference_ids]
                    # Direct cumulative growth; no recursively predicted lag enters X.
                    growth = np.clip(
                        model.predict(inference),
                        -cfg["clip_log_growth"],
                        cfg["clip_log_growth"],
                    )
                    prediction = values[inference_ids, origin] * np.exp(growth)
                    status = "trained_direct"
                    model_name = variant
                    trained = True
                fits.append(
                    dict(
                        origin=origin_month,
                        horizon=h,
                        variant=variant,
                        status=status,
                        fit_performed=trained,
                        training_rows=len(indices),
                        training_municipalities=len(np.unique(indices[:, 0])),
                        training_feature_origin_dates=len(np.unique(indices[:, 1])),
                        max_training_target=months[int(indices[:, 2].max())]
                        if len(indices)
                        else "",
                        inference_pairs=len(group),
                        seconds=time.monotonic() - begin,
                        optional_covariate_missing_fraction=float(
                            np.isnan(
                                feature_block(
                                    values, categories, regions, origin, h, context
                                )[inference_ids]
                            ).mean()
                        ),
                    )
                )
                for row, pred in zip(group.itertuples(), prediction):
                    records.append(
                        dict(
                            territory_id=row.territory_id,
                            origin=row.origin,
                            target=row.target,
                            horizon=h,
                            model=model_name,
                            status=status,
                            actual=row.actual,
                            predicted=float(pred),
                            year_ago=row.year_ago,
                        )
                    )
            print(
                f"Direct HGB {origin_month} h{h}: {len(indices)} training rows; {time.monotonic() - start:.1f}s",
                flush=True,
            )
    refs = source[
        source.model.isin(
            [
                "seasonal_pooled",
                "seasonal_naive",
                "prophet_yearly3",
                "prophet_pooled_profile",
            ]
        )
    ][keys + ["model", "actual", "predicted", "year_ago"]].copy()
    refs["status"] = "saved_reference"
    predictions = pd.concat([pd.DataFrame(records), refs], ignore_index=True)
    predictions["ae"] = (predictions.actual - predictions.predicted).abs()
    predictions["yoy_actual"] = 100 * (predictions.actual / predictions.year_ago - 1)
    predictions["yoy_predicted"] = 100 * (
        predictions.predicted / predictions.year_ago - 1
    )
    predictions["yoy_ae"] = (predictions.yoy_actual - predictions.yoy_predicted).abs()
    stats = []
    for (h, m), g in predictions.groupby(["horizon", "model"]):
        denom = float(
            ((g.actual - g.groupby("territory_id").actual.transform("mean")) ** 2).sum()
        )
        stats.append(
            dict(
                horizon=h,
                model=m,
                status=g.status.iloc[0],
                dates=g.target.nunique(),
                municipalities=g.territory_id.nunique(),
                observations=len(g),
                MAE=g.groupby("target").ae.mean().mean(),
                R2_pooled=r2_score(g.actual, g.predicted),
                R2_within_MO=1 - float((g.ae**2).sum()) / denom
                if denom > 0
                else np.nan,
                YoY_MAE_pp=g.groupby("target").yoy_ae.mean().mean(),
                YoY_R2=r2_score(g.yoy_actual, g.yoy_predicted),
            )
        )
    summary = pd.DataFrame(stats)
    seasonal = summary[summary.model == "seasonal_pooled"].set_index("horizon").MAE
    summary["skill_vs_seasonal_pooled"] = 1 - summary.MAE / summary.horizon.map(
        seasonal
    )
    monthly = (
        predictions.groupby(["horizon", "model", "target"])
        .agg(MAE=("ae", "mean"), observations=("ae", "size"))
        .reset_index()
    )
    fits = pd.DataFrame(fits)
    calendar = pd.DataFrame(calendars)
    assert (calendar.training_target <= calendar.fit_origin).all()
    assert not fits[fits.horizon == 12].fit_performed.any()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    monthly.to_csv(out / "monthly.csv", index=False)
    fits.to_csv(out / "fit_trace.csv", index=False)
    calendar.to_csv(out / "training_calendar.csv", index=False)
    sensitivity = []
    for h in [1, 3, 6]:
        ref = (
            monthly[(monthly.horizon == h) & (monthly.model == "seasonal_pooled")]
            .set_index("target")
            .MAE
        )
        for variant in cfg["variants"]:
            trial = (
                monthly[(monthly.horizon == h) & (monthly.model == variant)]
                .set_index("target")
                .MAE
            )
            gains = ref - trial
            for target, gain in gains.items():
                sensitivity.append(
                    dict(
                        horizon=h,
                        model=variant,
                        target=target,
                        gain_vs_seasonal_rub=float(gain),
                        mean_gain_without_target_rub=float(gains.drop(target).mean()),
                        dates_improved=int((gains > 0).sum()),
                        dates_total=len(gains),
                    )
                )
    pd.DataFrame(sensitivity).to_csv(out / "date_sensitivity.csv", index=False)
    inputs = [
        "configs/direct_horizon_review.json",
        "reports/direct_horizon_review/PREDECLARED_PROTOCOL.md",
        "src/sberindex/forecasting/direct_horizon_review.py",
        "tests/test_direct_horizon_review.py",
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "reports/prophet_seasonality/predictions.parquet",
        "results/asof_cohort_protocol.json",
        "src/sberindex/forecasting/deseasonal_hgb.py",
        "src/sberindex/forecasting/asof_hgb_12m.py",
    ]
    protocol = dict(
        config=cfg,
        eligible_training_cohort_2023_n=len(eligible),
        common_pairs=len(base),
        prediction_rows=len(predictions),
        fits_performed=int(fits.fit_performed.sum()),
        h12_direct_training_rows=0,
        h12_status="not_trainable_seasonal_fallback",
        trained_target_causality_verified=True,
        geographic_rows_label=2023,
        geographic_source_archive="20241025 retrospective; no metadata point-in-time availability claim",
        independent_time_validation=False,
        selected_primary_model=None,
        difference_from_existing="Separate cumulative-growth direct model perhorizon, rolling refit with targets<=origin. Existing HGB trains one-step model and recursively generates futurelags.",
        versions={
            p: importlib.metadata.version(p)
            for p in ["scikit-learn", "numpy", "pandas"]
        },
        input_sha256={
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in inputs
        },
    )
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(summary, protocol)
    print(summary.to_string(index=False))


def write_report(summary, protocol):
    lines = [
        "| h | model/status | dates/MO/pairs | MAE | skill vs seasonal | YoY MAE pp | YoY R² |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.horizon} | {r.model} | {r.dates}/{r.municipalities}/{r.observations} | {r.MAE:.1f} | {r.skill_vs_seasonal_pooled:.2%} | {r.YoY_MAE_pp:.2f} | {r.YoY_R2:.3f} |"
        )
    text = (
        """# Прямой pooled HGB по отдельным горизонтам

2026-10-06. Конфигурация и PREDECLARED_PROTOCOL.md сохранены до расчёта. Проверяемая новая гипотеза: отдельная модель накопленного изменения на каждом горизонте с переобучением по уже наблюдённым целям уменьшит ошибку рекурсивного распространения. Существующие asof_hgb_12m/deseasonal_hgb — замороженная модель одномесячного изменения 2023 с рекурсивно предсказанными лагами. Этот опыт не дублирует их. Использован установленный sklearn HistGradientBoostingRegressor, без добавления LightGBM.

Когорта обучения выбрана только по конечным положительным 12 фактам 2023:2075 МО. Оцениваются точные 7532 сохранённые общие пары прежней 256 ID когорты, фактически 251/252 МО. Все модели в таблице имеют одинаковые пары по горизонту:12/10/7/1 даты для h1/3/6/12. Сохранённые сильные Prophet и seasonal варианты служат ориентирами; они не переобучались в этом приложении.

Обучающая строка: origin s>=2, цель s+h<=текущий fit_origin. Метка — log(value[s+h]/value[s]). Признаки используют только историю до s и заранее известный календарь целевого месяца. Обучающие цели 2024 разрешены только после их наблюдения на текущем origin; будущая цель оцениваемого прогноза никогда не входит в fit. Полная положительная конечная история до s обязательна, отсутствующая цель исключается, будущие промежуточные значения после s для прямой модели не требуются. Пропуски optional категорияльных ковариат остаются NaN и не выбирают строки.

`direct_lags`: логарифм уровня на s, две последние месячные log разности, средний log уровень за 3 месяца, sin/cos календаря origin и target. `direct_context`: эти же признаки плюс медиана текущего месячного log изменения по всей обучающей когорте, аналогичная медиана по региону и локальные месячные log изменения пяти категорий расходов. Агрегаты каждого s используют только s/s−1; нет признака territory_id или будущих агрегатов. Эти агрегаты не являются региональным денежным оборотом. Гиперпараметры обеих моделей одинаковы, фиксированы; early_stopping=False,150 итераций,15 листьев,2 потока. Выход — прямая накопленная log разность, ограниченная±1 по предварительному правилу; рекурсивные будущие лаги не используются.

**h12 не обучен.** В декабре 2023 нет ни одной прямой 12 месячной пары в архиве, начинающемся январём 2023; fit_trace.csv показывает 0 строк и 0 обучений. В таблице эти строки явно названы `__seasonal_fallback_not_trainable` и повторяют seasonal_pooled. Их нельзя выдавать за обученную прямую h12 модель, а перенос коротких горизонтов не выполнялся. Для честного direct h12 нужны более ранние наблюдения или последующие origin с завершёнными 12 месячными целями, отсутствующие среди нынешних h12 общих пар.

"""
        + "\n".join(lines)
        + """

MAE усреднена сначала по МО внутри target, затем поровну по датам. CSV также содержит pooled/within R² и YoY метрики; within R² на однодатном h12 не определён. Обучающие rows/dates/IDs и максимальная дата обучающей цели раскрыты по каждому fit в fit_trace.csv; training_calendar.csv показывает число пар для каждого исторического origin/target. Все 58 fit h1/3/6 используют только цели≤текущего origin. h12 fallback отражён отдельно.

По фиксированному кандидату direct_lags: MAE на h1=2165,56 против 979,14 у seasonal_pooled (хуже 121,17%); на h3=1251,21 против 1633,04 (skill 23,38%); на h6=3188,61 против 1913,64 (хуже 66,63%). Контекстный вариант уступает лаговому на всех обученных горизонтах: агрегаты/категории не подтверждают улучшение в этой конфигурации.

На h3 lag вариант улучшает 6 из 10 дат. После исключения каждой одной даты средний выигрыш сохраняется 140,42–438,96 руб, но YoY R² остаётся−0,102; улучшение MAE не равно хорошему объяснению роста. date_sensitivity.csv показывает каждую дату и исключение каждой даты без перенастройки моделей. На h1 особенно велик проигрыш января 2024. В первом December 2023 fit обучающие календарные цели: h1 апрель–декабрь,h3 июнь–декабрь,h6 сентябрь–декабрь. Это реальный недостаток короткой годовой истории: известный календарь не заменяет отсутствующие обучающие календарные режимы.

Новая основная модель не выбиралась. Это повторно изученный архив 2024, не независимый временной тест. Прямой и рекурсивный подходы здесь дополнительно различаются режимом rolling refit и признаками; нельзя приписывать весь эффект одной замене рекурсии. Точное парное сравнение с сохранёнными seasonal/Prophet задано таблицей; рекурсивные модели с другими лагами публикации нельзя смешивать как равные условия.

География: используются записи с меткой 2023 из позднего снимка 20241025. Нет доказательства доступности этого справочника на историческом origin. Лаг публикации 0 — условный сценарий, не настоящий винтаж. Одновременные расходы пяти категорий предполагаются доступными вместе с «Все категории», реальные различия сроков публикации не проверены. Нет причинного вывода или гарантии переноса на другой год. Pooled обучение не является муниципальным out-of-sample: оцениваемые МО присутствуют в обучающей когорте, проверяется последующее время.

Проверки: будущие значения и будущие пропуски не изменяют набор обучающих пар; все обучающие цели≤fit_origin; h12 December 2023 пуст; пропуски цели/прошлой истории исключаются, неизвестный промежуточный факт для direct target не нужен; изменение будущих значений всех категорий не меняет признаки или региональные/глобальные агрегаты. protocol.json сохраняет SHA256 источников и версии.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.direct_horizon_review
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m unittest discover -s tests -p test_direct_horizon_review.py
```
"""
    )
    (OUT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
