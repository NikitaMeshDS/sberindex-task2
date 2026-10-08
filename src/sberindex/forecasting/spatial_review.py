"""Retrospective 2024 highway-neighbor sensitivity; never claims as-of graph validity."""

from datetime import datetime, timezone
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
from sberindex.forecasting.direct_horizon_review import training_indices, feature_block

OUT = ROOT / "reports/spatial_review"


def select_neighbors(highway, eligible_ids, k=5):
    """Explicit symmetric-distance assumption, equal-distance tie by neighbor ID.

    Preserve distinct zero-distance nodes. Fixed eligible IDs selected on2023,
    independent of any2024 expense values or missingness.
    """
    g = highway[
        highway.territory_id_x.isin(eligible_ids)
        & highway.territory_id_y.isin(eligible_ids)
        & (highway.territory_id_x != highway.territory_id_y)
        & np.isfinite(highway.distance)
        & (highway.distance >= 0)
    ].copy()
    reversed_g = g.rename(
        columns={"territory_id_x": "territory_id_y", "territory_id_y": "territory_id_x"}
    )
    both = pd.concat(
        [
            g[["territory_id_x", "territory_id_y", "distance"]],
            reversed_g[["territory_id_x", "territory_id_y", "distance"]],
        ],
        ignore_index=True,
    )
    both = both.sort_values(["territory_id_x", "distance", "territory_id_y"])
    both = both.drop_duplicates(["territory_id_x", "territory_id_y"])
    chosen = both.groupby("territory_id_x", sort=False).head(k).copy()
    chosen["rank"] = chosen.groupby("territory_id_x", sort=False).cumcount() + 1
    return chosen.reset_index(drop=True)


def neighbor_features(values, neighbors, source, alarm_threshold=0.15):
    """Mean growth(source), growth(source-1), lagged spike fraction, valid fraction.

    Spike flag is a fixed unvalidated monthly-growth proxy, not a real shock label.
    """
    indices = np.maximum(neighbors, 0)
    has_neighbor = neighbors >= 0

    def growth(t):
        current = values[indices, t]
        previous = values[indices, t - 1]
        valid = (
            has_neighbor
            & np.isfinite(current)
            & np.isfinite(previous)
            & (current > 0)
            & (previous > 0)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.log(np.where(valid, current / previous, np.nan))
        return result

    current = growth(source)
    past = growth(source - 1)

    def mean(x):
        valid = np.isfinite(x)
        count = valid.sum(axis=1)
        return np.divide(
            np.where(valid, x, 0.0).sum(axis=1),
            count,
            out=np.full(len(values), np.nan),
            where=count > 0,
        )

    flags = np.where(
        np.isfinite(past), (np.abs(past) > alarm_threshold).astype(float), np.nan
    )
    return np.column_stack(
        [
            mean(current),
            mean(past),
            mean(flags),
            np.isfinite(current).sum(axis=1) / neighbors.shape[1],
        ]
    )


def main():
    cfg = json.loads((ROOT / "configs/spatial_review.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
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
    values = panel.loc[eligible].to_numpy(float)
    connection = pd.read_parquet(ROOT / "data/connection.parquet")
    highway = connection[connection["type"] == "highway"].copy()
    lo = np.minimum(highway.territory_id_x, highway.territory_id_y)
    hi = np.maximum(highway.territory_id_x, highway.territory_id_y)
    unordered_duplicate_rows = int(
        pd.DataFrame({"lo": lo, "hi": hi}).duplicated(keep=False).sum()
    )
    assert (
        unordered_duplicate_rows == 0
        and not (highway.territory_id_x == highway.territory_id_y).any()
    )
    highway_nodes = len(set(highway.territory_id_x) | set(highway.territory_id_y))
    chosen = select_neighbors(highway, eligible, cfg["neighbor_count"])
    chosen.to_csv(OUT / "neighbors.csv", index=False)
    neighbors = np.full((len(eligible), cfg["neighbor_count"]), -1, dtype=int)
    for r in chosen.itertuples():
        neighbors[eligible.get_loc(r.territory_id_x), r.rank - 1] = eligible.get_loc(
            r.territory_id_y
        )
    source_metadata = next(
        s
        for s in json.loads((ROOT / "data_sources.json").read_text())["sources"]
        if s["path"] == "data/connection.parquet"
    )
    snapshot = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen_before_first_fit=True,
        graph_sha256=hashlib.sha256(
            (ROOT / "data/connection.parquet").read_bytes()
        ).hexdigest(),
        selected_neighbors_sha256=hashlib.sha256(
            (OUT / "neighbors.csv").read_bytes()
        ).hexdigest(),
        graph_rows=len(connection),
        type_counts=connection["type"].value_counts().to_dict(),
        highway_nodes=highway_nodes,
        highway_rows=len(highway),
        unordered_duplicate_rows=unordered_duplicate_rows,
        self_rows=0,
        stored_x_gt_y_rows=int((highway.territory_id_x > highway.territory_id_y).sum()),
        stored_x_lt_y_rows=int((highway.territory_id_x < highway.territory_id_y).sum()),
        potential_unordered_pairs=highway_nodes * (highway_nodes - 1) // 2,
        unordered_pair_coverage=len(highway)
        / (highway_nodes * (highway_nodes - 1) / 2),
        zero_distance_pairs=int((highway.distance == 0).sum()),
        source_metadata=source_metadata,
        distance_unit="unknown; distances used only for ordering",
        symmetrization="Explicit modeling assumption: one stored orientation per unordered highway pair; not directed flow or true adjacency",
        graph_use="2024 retrospective sensitivity; availability/validity in2023 not established",
        eligible_2023_ids=len(eligible),
        selected_neighbor_rows=len(chosen),
        ids_with_five_neighbors=int(
            ((neighbors >= 0).sum(axis=1) == cfg["neighbor_count"]).sum()
        ),
        ids_without_any_neighbor=int(((neighbors >= 0).sum(axis=1) == 0).sum()),
        ids_with_partial_neighbors=int(
            (
                ((neighbors >= 0).sum(axis=1) > 0)
                & ((neighbors >= 0).sum(axis=1) < cfg["neighbor_count"])
            ).sum()
        ),
    )
    (OUT / "graph_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    )
    del connection, highway, lo, hi
    source = pd.read_parquet(ROOT / "reports/prophet_seasonality/predictions.parquet")
    base = source[(source.model == "seasonal_pooled") & (source.horizon == 1)].copy()
    previous = pd.read_parquet(
        ROOT / "reports/direct_horizon_review/predictions.parquet"
    )
    previous = (
        previous[(previous.model == "direct_lags") & (previous.horizon == 1)]
        .set_index(["territory_id", "origin", "target"])
        .predicted
    )
    ids = json.loads((ROOT / "results/asof_cohort_protocol.json").read_text())[
        "sample_ids"
    ]
    assert set(base.territory_id).issubset(ids)
    records = []
    fit_rows = []
    cache = {}
    empty_categories = np.empty((len(values), 0, 24))
    regions = np.full(len(values), np.nan)

    def features(s, spatial):
        key = (s, spatial)
        if key not in cache:
            x = feature_block(values, empty_categories, regions, s, 1, False)
            if spatial:
                x = np.column_stack(
                    [
                        x,
                        neighbor_features(
                            values,
                            neighbors,
                            s,
                            cfg["lagged_proxy_alarm_abs_log_growth_threshold"],
                        ),
                    ]
                )
            cache[key] = x
        return cache[key]

    with threadpool_limits(limits=cfg["threads"]):
        for origin_month, group in base.groupby("origin"):
            origin = (pd.Period(origin_month, "M") - pd.Period("2023-01", "M")).n
            training = training_indices(
                values, origin, 1, cfg["first_training_origin_index"]
            )
            idx = eligible.get_indexer(group.territory_id)
            assert (idx >= 0).all()
            np.testing.assert_array_equal(values[idx, origin + 1], group.actual)
            for variant in cfg["variants"]:
                tick = time.monotonic()
                started = datetime.now(timezone.utc).isoformat()
                spatial = variant == "spatial_neighbors"
                x = []
                y = []
                for s in np.unique(training[:, 1]):
                    rows = training[training[:, 1] == s, 0]
                    x.append(features(int(s), spatial)[rows])
                    y.append(np.log(values[rows, int(s) + 1] / values[rows, int(s)]))
                model = HistGradientBoostingRegressor(**cfg["model"]).fit(
                    np.concatenate(x), np.concatenate(y)
                )
                pred = values[idx, origin] * np.exp(
                    np.clip(
                        model.predict(features(origin, spatial)[idx]),
                        -cfg["clip_log_growth"],
                        cfg["clip_log_growth"],
                    )
                )
                if not spatial:
                    saved = np.array(
                        [
                            previous.loc[(r.territory_id, r.origin, r.target)]
                            for r in group.itertuples()
                        ]
                    )
                    np.testing.assert_allclose(pred, saved, rtol=1e-12, atol=1e-8)
                fit_rows.append(
                    dict(
                        origin=origin_month,
                        model=variant,
                        training_rows=len(training),
                        training_municipalities=len(np.unique(training[:, 0])),
                        max_training_target=months[int(training[:, 2].max())],
                        fit_started_at_utc=started,
                        seconds=time.monotonic() - tick,
                        inference_pairs=len(group),
                        feature_missing_fraction=float(
                            np.isnan(features(origin, spatial)[idx]).mean()
                        ),
                    )
                )
                for r, prediction in zip(group.itertuples(), pred):
                    records.append(
                        dict(
                            territory_id=r.territory_id,
                            origin=r.origin,
                            target=r.target,
                            horizon=1,
                            model=variant,
                            actual=r.actual,
                            predicted=float(prediction),
                            year_ago=r.year_ago,
                        )
                    )
            print(
                f"Spatial retrospective {origin_month}: {len(training)} causal training rows",
                flush=True,
            )
    refs = base[
        [
            "territory_id",
            "origin",
            "target",
            "horizon",
            "model",
            "actual",
            "predicted",
            "year_ago",
        ]
    ]
    predictions = pd.concat([pd.DataFrame(records), refs], ignore_index=True)
    predictions["ae"] = (predictions.actual - predictions.predicted).abs()
    predictions["yoy_actual"] = 100 * (predictions.actual / predictions.year_ago - 1)
    predictions["yoy_predicted"] = 100 * (
        predictions.predicted / predictions.year_ago - 1
    )
    predictions["yoy_ae"] = (predictions.yoy_actual - predictions.yoy_predicted).abs()
    stats = []
    for m, g in predictions.groupby("model"):
        denom = float(
            ((g.actual - g.groupby("territory_id").actual.transform("mean")) ** 2).sum()
        )
        stats.append(
            dict(
                model=m,
                dates=g.target.nunique(),
                municipalities=g.territory_id.nunique(),
                observations=len(g),
                MAE=g.groupby("target").ae.mean().mean(),
                R2_pooled=r2_score(g.actual, g.predicted),
                R2_within_MO=1 - float((g.ae**2).sum()) / denom,
                YoY_MAE_pp=g.groupby("target").yoy_ae.mean().mean(),
                YoY_R2=r2_score(g.yoy_actual, g.yoy_predicted),
            )
        )
    summary = pd.DataFrame(stats)
    baseline = summary[summary.model == "spatial_base"].MAE.iloc[0]
    summary["skill_vs_no_neighbors"] = 1 - summary.MAE / baseline
    monthly = (
        predictions.groupby(["model", "target"])
        .agg(MAE=("ae", "mean"), observations=("ae", "size"))
        .reset_index()
    )
    a = monthly[monthly.model == "spatial_base"].set_index("target").MAE
    b = monthly[monthly.model == "spatial_neighbors"].set_index("target").MAE
    gains = a - b
    sensitivity = pd.DataFrame(
        [
            dict(
                target=t,
                gain_neighbors_rub=float(v),
                mean_gain_without_target_rub=float(gains.drop(t).mean()),
            )
            for t, v in gains.items()
        ]
    )
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    monthly.to_csv(OUT / "monthly.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(OUT / "fit_trace.csv", index=False)
    sensitivity.to_csv(OUT / "date_sensitivity.csv", index=False)
    sources = [
        "configs/spatial_review.json",
        "reports/spatial_review/PREDECLARED_PROTOCOL.md",
        "src/sberindex/forecasting/spatial_review.py",
        "tests/test_spatial_review.py",
        "src/sberindex/forecasting/direct_horizon_review.py",
        "data/connection.parquet",
        "data_sources.json",
        "data/consumption.parquet",
        "results/asof_cohort_protocol.json",
        "reports/prophet_seasonality/predictions.parquet",
        "reports/direct_horizon_review/predictions.parquet",
    ]
    protocol = dict(
        config=cfg,
        graph_snapshot=snapshot,
        common_pairs=len(base),
        prediction_rows=len(predictions),
        fits_performed=len(fit_rows),
        baseline_reproduces_saved_direct_lags=True,
        training_target_causality_verified=True,
        graph_retrospective_only=True,
        proxy_alarm_labels="Unvalidated fixed monthly-growth spike flags, lagged one month; not real shift labels or false-alarm metrics",
        selected_primary_model=None,
        independent_time_validation=False,
        versions={
            p: importlib.metadata.version(p)
            for p in ["scikit-learn", "numpy", "pandas"]
        },
        input_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
        },
    )
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(summary, snapshot, gains)
    print(summary.to_string(index=False))
    print(
        json.dumps(
            {
                k: snapshot[k]
                for k in [
                    "selected_neighbor_rows",
                    "ids_with_five_neighbors",
                    "ids_without_any_neighbor",
                    "unordered_pair_coverage",
                ]
            }
        )
    )


def write_report(summary, snapshot, gains):
    rows = [
        "| Модель | MAE | skill vs без соседей | within R² | YoY MAE pp | YoY R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        rows.append(
            f"| {r.model} | {r.MAE:.2f} | {r.skill_vs_no_neighbors:.2%} | {r.R2_within_MO:.3f} | {r.YoY_MAE_pp:.2f} | {r.YoY_R2:.3f} |"
        )
    baseline = summary[summary.model == "spatial_base"].MAE.iloc[0]
    neighbor = summary[summary.model == "spatial_neighbors"].MAE.iloc[0]
    text = (
        f"""# Соседи по highway: ретроспективная чувствительность

2026-10-06. Граф существует: data/connection.parquet,5942364 строки, highway 3303736 и railway 2638628. Для highway 2573 узла,3303736 уникальных неупорядоченных пар, без self loops/повторных пар,148 разных пар с distance=0. Записи имеют одну ориентацию пары: x>y 3253861 раз, x<y 49875 раз. Покрытие потенциальных пар≈{snapshot["unordered_pair_coverage"]:.3%}. Это почти полный каталог попарных дистанций, а не разреженная матрица непосредственной дорожной смежности. Симметризация — явно заявленное предположение модели; направление транспортного потока не выводится. Единица distance не установлена, километры не приписываются; используется только порядок.

**Граф — снимок 2024, только ретроспективная чувствительность.** data_sources.json имеет period=2024snapshot,usage=retrospective_interpretation. Доступность/валидность в 2023 не доказана. Поэтому past-only расходы не делают весь опыт полноценным as-of: в структуре графа есть поздняя информация. Числа не доказывают реальное раннее предупреждение о шоке. Внешние события не используются как выдуманные метки.

Конфигурация и предварительный протокол заданы до метрик. До первого fit сохранены SHA256 графа и фиксированный neighbors.csv в graph_snapshot.json. Для каждого МО выбраны 5 ближайших по distance среди 2075 пригодных по 2023 МО, tie по ID. Нулевые дистанции разных ID сохранены. Будущие факты/пропуски 2024 не выбирают соседей. Сохранено {snapshot["selected_neighbor_rows"]} соседних записей; пять соседей имеют {snapshot["ids_with_five_neighbors"]} МО, без соседей {snapshot["ids_without_any_neighbor"]} МО; неполный список имеют {snapshot["ids_with_partial_neighbors"]} МО.

Одинаковые 3012 общих h1 пары,251 МО,12 дат сохранённой 256 ID когорты. spatial_base воспроизводит каждый сохранённый прогноз direct_lags; spatial_neighbors добавляет только четыре признака: среднее месячное log изменение соседей на s, такое же на s−1, долю соседних spike-флагов на s−1 и долю доступных текущих изменений относительно пяти соседних слотов (учитывает и отсутствующие связи). Spike-флаг: abs(monthlyloggrowth)>0,15, фиксирован заранее. Это не валидированный детектор структурного изменения: сезонные пики могут вызывать флаг; не рассчитываются его precision/F1/ложные тревоги на неразмеченных расходах. Доля прошлого флага не является доступной новостью до внешнего шока.

Все расходные признаки используют только≤s. Обучающая цель s+1≤fit_origin; обе HGB модели переобучаются на одном наборе causal training pairs каждый origin. Пропущенные значения соседей игнорируются внутри агрегата; при полном отсутствии значение NaN, доля доступных 0, прогнозная пара не удаляется. Фиксированные 150 итераций/15 листьев и остальные настройки одинаковы;24 обучения,2 потока,без выбора по 2024. Данные считаются опубликованными с лагом 0 — условный сценарий.

"""
        + "\n".join(rows)
        + f"""

Изменение MAE соседнего варианта относительно baseline: {(neighbor / baseline - 1) * 100:.2f}% (положительное означает ухудшение). В этой фиксированной конфигурации гипотеза улучшения не подтверждена; это не доказательство бесполезности всех пространственных моделей. Соседний вариант улучшает {int((gains > 0).sum())}/12 дат относительно spatial_base. При исключении каждой одной даты средний выигрыш соседнего варианта находится в диапазоне {float(((gains.sum() - gains) / (len(gains) - 1)).min()):.2f}–{float(((gains.sum() - gains) / (len(gains) - 1)).max()):.2f} руб. date_sensitivity.csv раскрывает все даты. Сильный сезонный контроль также показан: сравнение двух HGB не доказывает превосходство над сезонным профилем. Новая основная модель не назначается; это уже изученный 2024 архив, не независимый тест.

Графовая структура позднего снимка, календарные режимы короткой обучающей истории, отсутствие расходных винтажей и реальных меток ограничивают выводы. Модель проверяет информацию в соседних расходных изменениях, не причинное распространение шока по дорогам. Для as-of исследования требуется источник графа с подтверждённой исторической доступностью и семантикой дистанции/ориентации.

Тесты: будущие изменения всех соседних рядов не меняют прошлые признаки; одностороннее хранение пары корректно симметризуется по заявленному предположению, tie детерминирован; лагgedflags проверены на явном числовом примере с пропуском. protocol.json сохраняет источники/версии,neighbors.csv—фиксированную географию,fit_trace.csv—максимальные обучающие цели и timestamp после записи snapshot.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.spatial_review
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m unittest discover -s tests -p test_spatial_review.py
```
"""
    )
    (OUT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
