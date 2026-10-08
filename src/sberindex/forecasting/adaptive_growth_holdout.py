"""Post-review candidate: causal regional three-month YoY, unchanged seasonal profile."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sberindex.forecasting.growth_bridge_review import growth_for_origin, year_bridges
from sberindex.forecasting.selected_model_holdout import forecast_selected
from sberindex.paths import ROOT


def regional_growth_rate(panel, regions, origin, fallback, minimum_n=10):
    """Median of three monthly regional medians; all three need minimum_n valid MO."""
    training = panel.reindex(
        columns=pd.period_range("2023-01", "2023-12", freq="M").astype(str)
    )
    eligible = training.index[
        np.isfinite(training).all(axis=1) & (training > 0).all(axis=1)
    ]
    aligned = regions.reindex(eligible)
    months = pd.period_range(end=origin, periods=3, freq="M")
    rates = {}
    counts = {}
    for region in aligned.dropna().unique():
        members = aligned.index[aligned == region]
        medians = []
        monthly_counts = []
        for month in months:
            current = panel.reindex(index=members, columns=[str(month)]).iloc[:, 0]
            prior = panel.reindex(index=members, columns=[str(month - 12)]).iloc[:, 0]
            good = (
                np.isfinite(current) & np.isfinite(prior) & (current > 0) & (prior > 0)
            )
            monthly_counts.append(int(good.sum()))
            medians.append(
                float((current[good] / prior[good] - 1).median())
                if good.sum() >= minimum_n
                else np.nan
            )
        counts[region] = monthly_counts
        if np.isfinite(medians).all():
            rates[region] = float(np.median(medians))
    records = []
    for tid in panel.index:
        region = regions.get(tid, np.nan)
        regional = region in rates
        records.append(
            {
                "territory_id": int(tid),
                "region_code": region,
                "origin": origin,
                "growth_rate": rates.get(region, fallback),
                "growth_source": "regional_three_month_yoy"
                if regional
                else "documented_macro_fallback",
                "window_start": str(months[0]),
                "window_end": str(months[-1]),
                "minimum_valid_monthly_peers": min(counts.get(region, [0])),
            }
        )
    return pd.DataFrame(records)


def main():
    cfg_path = ROOT / "configs/adaptive_growth_holdout.json"
    cfg = json.loads(cfg_path.read_text())
    selected_path = ROOT / "configs/growth_bridge_review.json"
    selected = json.loads(selected_path.read_text())
    source = json.loads((ROOT / selected["source_record"]).read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    raw = raw[(raw.category == cfg["category"]) & (raw.date <= cfg["origin"])]
    panel = raw.pivot(index="territory_id", columns="date", values="value").sort_index()
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    regions = lookup[lookup.year == 2023].set_index("territory_id").region_code
    fallback = growth_for_origin(cfg["origin"], source)
    rates = regional_growth_rate(
        panel, regions, cfg["origin"], fallback, cfg["minimum_monthly_peers"]
    )
    forecasts = forecast_selected(
        panel,
        regions,
        cfg["origin"],
        cfg["horizons"],
        0.0,
        selected["region_weight"],
        selected["minimum_region_municipalities"],
    )
    forecasts = forecasts.merge(
        rates, on=["territory_id", "origin"], validate="many_to_one"
    )
    k = np.array(
        [year_bridges(o, h) for o, h in zip(forecasts.origin, forecasts.horizon)]
    )
    forecasts["predicted"] *= (1 + forecasts.growth_rate.to_numpy()) ** k
    forecasts["model"] = "hierarchy05_adaptive_regional_growth_candidate"
    out = ROOT / cfg["output_directory"]
    if out.exists():
        raise FileExistsError("Never overwrite an existing frozen candidate")
    out.mkdir(parents=True)
    forecasts.to_csv(out / "predictions.csv", index=False, float_format="%.12g")
    rates[rates.territory_id.isin(forecasts.territory_id)].to_csv(
        out / "growth_rates.csv", index=False
    )
    # Cached diagnostic on studied 2024, no refitting or new benchmark selection.
    previous = pd.read_parquet(
        ROOT / "reports/growth_bridge_review/predictions.parquet"
    )
    base = previous[previous.model == "hierarchy05"].copy()
    diagnostic = []
    for origin, group in base.groupby("origin", sort=True):
        rr = regional_growth_rate(
            panel,
            regions,
            origin,
            growth_for_origin(origin, source),
            cfg["minimum_monthly_peers"],
        ).set_index("territory_id")
        rr = rr.reindex(group.territory_id)
        copy = group.copy()
        bridges = np.array([year_bridges(origin, h) for h in group.horizon])
        copy["predicted"] *= (1 + rr.growth_rate.to_numpy()) ** bridges
        copy["ae"] = abs(copy.actual - copy.predicted)
        copy["model"] = "hierarchy05_adaptive_regional_growth_candidate"
        diagnostic.append(copy)
    diagnostic = pd.concat(diagnostic, ignore_index=True)
    diagnostic.to_parquet(out / "diagnostic_2024.parquet", index=False)
    controls = previous[
        previous.model.isin(["hierarchy05_growth_bridge", "prophet_pooled_profile"])
    ]
    summary = (
        pd.concat([diagnostic, controls])
        .groupby(["model", "horizon", "target"])
        .ae.mean()
        .groupby(["model", "horizon"])
        .mean()
        .rename("MAE")
        .reset_index()
    )
    summary.to_csv(out / "diagnostic_2024_summary.csv", index=False)
    original = pd.read_csv(
        ROOT / "reports/selected_model_holdout_20261007/predictions.csv"
    )
    matched = forecasts.merge(
        original,
        on=["territory_id", "origin", "target", "horizon"],
        suffixes=("_adaptive", "_fixed"),
        validate="one_to_one",
    )
    assert len(matched) == len(forecasts) == len(original)
    matched["change_pct"] = 100 * (
        matched.predicted_adaptive / matched.predicted_fixed - 1
    )
    changes = (
        matched.groupby("horizon")
        .change_pct.agg(["median", "min", "max"])
        .reset_index()
    )
    changes.to_csv(out / "change_from_fixed.csv", index=False)
    inputs = [
        "configs/adaptive_growth_holdout.json",
        "configs/growth_bridge_review.json",
        selected["source_record"],
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "src/sberindex/forecasting/adaptive_growth_holdout.py",
        "src/sberindex/forecasting/selected_model_holdout.py",
        "src/sberindex/forecasting/growth_bridge_review.py",
        "src/sberindex/forecasting/hierarchical_review.py",
        "reports/growth_bridge_review/predictions.parquet",
        "reports/selected_model_holdout_20261007/predictions.csv",
    ]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    m = {
        "status": "awaiting_official_target_data",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": forecasts.model.iloc[0],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_tree_modified": True,
        "rows": len(forecasts),
        "municipalities": int(forecasts.territory_id.nunique()),
        "growth_rule": "Median of last3 monthly regional medians of positive finite MO YoY; 2023 eligibility; >=10 valid MO in each month; otherwise documented macro fallback",
        "input_sha256": {p: sha(ROOT / p) for p in inputs},
        "output_sha256": {
            p: sha(out / p)
            for p in [
                "predictions.csv",
                "growth_rates.csv",
                "diagnostic_2024.parquet",
                "diagnostic_2024_summary.csv",
                "change_from_fixed.csv",
            ]
        },
        "primary_model_replaced": False,
        "original_freeze_overwritten": False,
        "municipal_2025_target_values_read": False,
        "known_2025_aggregate_context": True,
        "rule_proposed_after_2024_results_and_2025_aggregate_comment": True,
        "independent_evaluation_performed": False,
        "reporting_lag_assumption_months": 0,
    }
    (out / "freeze_manifest.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2) + "\n"
    )
    med = float(forecasts.drop_duplicates("territory_id").growth_rate.median())
    (out / "REPORT.md").write_text(
        f"""# Адаптивный рост: второй кандидат на 2025\n\nСохранено {len(forecasts)} прогнозов для {m["municipalities"]} МО на h1/3/6/12 из декабря 2024. Старые прогнозы с g=17,2% не изменены. Медианная ставка по оцениваемым МО: {med:.2%}.\n\nДля каждого из трёх последних известных месяцев считаем медиану годового роста расходов пригодных МО региона, затем медиану трёх значений. В каждом месяце нужно минимум десять положительных конечных пар; иначе используется документированный макропоказатель 17,2%. Профиль 2023 и региональный вес 0,5 сохранены. Пропуски не заполняются будущими фактами; ограничений на положительный/отрицательный рост и подбора коэффициента нет.\n\nПравило предложено после анализа 2024 и знакомства с сообщением об агрегированных расходах декабря 2025. Поэтому новый кандидат не является предварительно зарегистрированным независимым тестом. Агрегат 2025 не используется в формуле, но его известность раскрыта; муниципальные факты 2025 не читались. Публикация декабрьского факта предполагает лаг 0. Исторические версии географии не подтверждены. Ни одна версия не гарантирует предсказания дальнейшего замедления в 2025.\n\n`diagnostic_2024_summary.csv` — проверка на уже изученном архиве, не новый независимый тест. На h12 2024 оба правила совпадают: в декабре 2023 нет годовой истории, используется Росстат. После получения сопоставимых фактов 2025 сравнивать оба замороженных файла на точных одинаковых ключах; не выбирать правило по национальному агрегату.\n"""
    )
    print("Adaptive freeze:", len(forecasts), "median g:", med)
    print(summary.to_string(index=False))
    print(changes.to_string(index=False))


if __name__ == "__main__":
    main()
