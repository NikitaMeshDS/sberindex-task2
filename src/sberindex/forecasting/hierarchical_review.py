"""Fixed global/region seasonal mixtures on saved 256-ID common pairs."""

import hashlib
import json
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sberindex.paths import ROOT

OUT = ROOT / "reports/hierarchical_review"


def fit_profiles(panel, regions, minimum_n=10):
    """Fit only first 12 (2023) observations; ignore every future column."""
    train = panel.iloc[:, :12]
    eligible = train.index[np.isfinite(train).all(axis=1) & (train > 0).all(axis=1)]
    train = train.loc[eligible]
    global_profile = train.sum().to_numpy(float)
    global_profile = global_profile / global_profile.mean()
    regional = {}
    aligned = regions.reindex(eligible)
    for region in aligned.dropna().unique():
        members = aligned.index[aligned == region]
        if len(members) < minimum_n:
            continue
        profile = train.loc[members].sum().to_numpy(float)
        regional[int(region)] = profile / profile.mean()
    return global_profile, regional


def predict_profile(
    origin_value, origin_index, target_index, region, global_profile, regional, weight
):
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between zero and one")
    profile = (1 - weight) * global_profile + weight * regional.get(
        region, global_profile
    )
    return float(origin_value * profile[target_index % 12] / profile[origin_index % 12])


def main():
    config = json.loads((ROOT / "configs/hierarchical_review.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (
        raw[raw.category == config["category"]]
        .pivot(index="territory_id", columns="date", values="value")
        .sort_index()
    )
    panel = panel.reindex(
        columns=pd.period_range("2023-01", "2024-12", freq="M").astype(str)
    )
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    geo = lookup[lookup.year == config["geography_year_preferred"]].set_index(
        "territory_id"
    )
    assert geo.index.is_unique
    regions = geo.region_code.reindex(panel.index)
    global_profile, regional = fit_profiles(
        panel, regions, config["minimum_region_municipalities"]
    )
    source = pd.read_parquet(ROOT / "reports/prophet_seasonality/predictions.parquet")
    base = source[source.model == "seasonal_pooled"].copy()
    assert not base.duplicated(["territory_id", "origin", "target", "horizon"]).any()
    ids = json.loads((ROOT / "results/asof_cohort_protocol.json").read_text())[
        "sample_ids"
    ]
    assert set(base.territory_id).issubset(ids)
    records = []
    for row in base.itertuples():
        origin = (pd.Period(row.origin, "M") - pd.Period("2023-01", "M")).n
        target = (pd.Period(row.target, "M") - pd.Period("2023-01", "M")).n
        assert target - origin == row.horizon and row.horizon in config["horizons"]
        assert panel.loc[row.territory_id].iloc[target] == row.actual
        region = regions.loc[row.territory_id]
        for weight in config["region_weights"]:
            pred = predict_profile(
                panel.loc[row.territory_id].iloc[origin],
                origin,
                target,
                region,
                global_profile,
                regional,
                weight,
            )
            if weight == 0:
                assert np.isclose(pred, row.predicted, atol=1e-8, rtol=1e-10)
            records.append(
                dict(
                    territory_id=row.territory_id,
                    region_code=region,
                    origin=row.origin,
                    target=row.target,
                    horizon=row.horizon,
                    model=f"region_weight_{weight:g}",
                    region_weight=weight,
                    actual=row.actual,
                    predicted=pred,
                    year_ago=panel.loc[row.territory_id].iloc[target - 12],
                    region_fallback=region not in regional,
                )
            )
    predictions = pd.DataFrame(records)
    predictions["ae"] = (predictions.actual - predictions.predicted).abs()
    predictions["yoy_actual"] = 100 * (predictions.actual / predictions.year_ago - 1)
    predictions["yoy_predicted"] = 100 * (
        predictions.predicted / predictions.year_ago - 1
    )
    predictions["yoy_ae"] = (predictions.yoy_actual - predictions.yoy_predicted).abs()
    stats = []
    for (h, model), g in predictions.groupby(["horizon", "model"]):
        denom = float(
            ((g.actual - g.groupby("territory_id").actual.transform("mean")) ** 2).sum()
        )
        stats.append(
            dict(
                horizon=h,
                model=model,
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
                region_fallback_observations=int(g.region_fallback.sum()),
            )
        )
    summary = pd.DataFrame(stats)
    global_mae = summary[summary.model == "region_weight_0"].set_index("horizon").MAE
    summary["skill_vs_global"] = 1 - summary.MAE / summary.horizon.map(global_mae)
    monthly = (
        predictions.groupby(["horizon", "model", "target"])
        .agg(MAE=("ae", "mean"), observations=("ae", "size"))
        .reset_index()
    )
    train = panel.iloc[:, :12]
    eligible = train.index[np.isfinite(train).all(axis=1) & (train > 0).all(axis=1)]
    counts = (
        regions.reindex(eligible)
        .value_counts()
        .rename_axis("region_code")
        .reset_index(name="eligible_2023_n")
    )
    counts["regional_profile_used"] = counts.region_code.isin(regional)
    inputs = [
        "configs/hierarchical_review.json",
        "reports/hierarchical_review/PREDECLARED_PROTOCOL.md",
        "src/sberindex/forecasting/hierarchical_review.py",
        "tests/test_hierarchical_review.py",
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "data/connection.parquet",
        "data_sources.json",
        "reports/prophet_seasonality/predictions.parquet",
        "results/asof_cohort_protocol.json",
    ]
    audit = dict(
        config=config,
        fit_year=2023,
        eligible_2023_n=len(eligible),
        requested_sample_ids=len(ids),
        common_pairs=len(base),
        prediction_rows=len(predictions),
        global_profile_matches_saved_baseline=True,
        region_profiles_n=len(regional),
        unmapped_eligible_2023_n=int(regions.reindex(eligible).isna().sum()),
        independent_time_validation=False,
        selected_winner=None,
        versions={
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn"]
        },
        geographic_snapshot="Rows labelled2023 in municipal_lookup; retrospective source archive20241025, availability in2023 not established",
        spatial_graph="Existing data/connection.parquet has5942364rows with territory_id_x/y,distance,type. data_sources.json labels2024snapshot andretrospective_interpretation; historical2023availability and distanceunit not established. Not used by this seasonal experiment; a separate retrospective spatial sensitivity may use it.",
        input_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in inputs
        },
    )
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    monthly.to_csv(OUT / "monthly.csv", index=False)
    counts.to_csv(OUT / "region_coverage.csv", index=False)
    (OUT / "protocol.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "| h | regional weight | dates/MO/pairs | MAE | skill vs global | YoY MAE pp | YoY R² |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.horizon} | {r.model.removeprefix('region_weight_')} | {r.dates}/{r.municipalities}/{r.observations} | {r.MAE:.1f} | {r.skill_vs_global:.3%} | {r.YoY_MAE_pp:.2f} | {r.YoY_R2:.3f} |"
        )
    (OUT / "REPORT.md").write_text(
        """# Фиксированное региональное сглаживание сезонности

Расчёт 2026-10-06. Предварительный протокол и конфигурация сохранены до метрик. Ни один вес не выбран победителем по 2024; все 0/.25/.5/.75 показаны. Это отдельный опыт на прежних 256 ID общих парах, без изменения Prophet или полного когортного приложения.

Профили — нормированные суммы расходов МО по календарным месяцам 2023: глобальный и региональный. Это сумма показателей на жителя, а не агрегированный денежный оборот. Арифметическое смешивание профилей с фиксированным региональным весом; прогноз — наблюдённый уровень на origin × target/origin профиль. Все профили заморожены на 2023. Последующие факты могут быть использованы только как уже наблюдённый уровень origin; цель не используется. Регион с менее 10 пригодных 2023 МО возвращается к глобальному профилю. Глобальный контроль по каждой паре совпадает с сохранённым seasonal_pooled.

"""
        + f"Профили обучены на {len(eligible)} пригодных по 2023 МО; {len(regional)} региональных профилей. Запрошенная когорта {len(ids)} ID, фактически {len(base)} общих пар; отсутствующие в исходных парах факты не восстановлены.\n\n"
        + "\n".join(lines)
        + """

На h1 вес 0,50 уменьшил MAE на 5,33%, на h3 — на 7,35%; это фиксированный кандидат, а не выбранный победитель. На h6 вес 0,25 улучшил MAE на 1,66%,0,75 ухудшил на 3,13%. На h12 все прогнозы совпадают: target и origin имеют один календарный месяц, отношение любого замороженного сезонного профиля равно 1. Отрицательный YoY R² на h3/6/12 сохраняется, поэтому снижение MAE не означает качественного объяснения динамики роста.

MAE: сначала среднее по МО внутри target, затем одинаковый вес датам. Таблица h1/3/6/12 имеет 12/10/7/1 дат; h12 не проверяет устойчивость во времени, within-MO R² при одной дате неопределён. CSV содержит также pooled/within R², число fallback и все исходные ошибки. Обобщение на новые годы/реальные события не доказано:2024 уже изучался, веса фиксированы после предыдущих исследований и не являются новым независимым тестом.

География: записи справочника с годом 2023, но источник municipal versions имеет поздний снимок 20241025. Метка года не доказывает доступность справочника на origin; пространственный или географический результат считается ретроспективной гипотезой. Граф действительно есть: data/connection.parquet содержит 5942364 строки territory_id_x/y,distance,type, включая highway. data_sources.json указывает 2024snapshot и usage=retrospective_interpretation; доступность графа в 2023 и единица distance не установлены. Отсутствует подтверждённая пригодность именно для исторического as-of2023 анализа, а не сам граф. Этот опыт сезонности граф не использовал; отдельная пространственная чувствительность может использовать его только как явно ретроспективный снимок, без приписывания distance километрам или утверждения раннего предупреждения.

Проверка: meaningful тест заменяет все 2024 значения пропусками и проверяет неизменность всех обученных профилей. Другой тест проверяет fallback и точное поведение нулевого регионального веса. protocol.json сохраняет SHA256 источников и число пар; predictions.parquet содержит каждую пару/вес, monthly.csv — каждую дату, region_coverage.csv — размер региональных обучающих групп.

Воспроизведение:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.hierarchical_review
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m unittest discover -s tests -p test_hierarchical_review.py
```
"""
    )
    print(summary.to_string(index=False))
    print(
        json.dumps(
            {
                k: audit[k]
                for k in [
                    "eligible_2023_n",
                    "common_pairs",
                    "prediction_rows",
                    "region_profiles_n",
                    "unmapped_eligible_2023_n",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
