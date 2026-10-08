"""Unchanged growth bridge across six labels; resumable frozen Prophet checks."""

import argparse
import concurrent.futures
import hashlib
import json
import time

import numpy as np
import pandas as pd

from sberindex.forecasting.full_cohort_review import (
    eligible_ids,
    evaluation_pairs,
    refit,
)
from sberindex.forecasting.growth_bridge_review import growth_for_origin
from sberindex.forecasting.hierarchical_review import fit_profiles, predict_profile
from sberindex.paths import ROOT

OUT = ROOT / "reports/category_growth_review"
MONTHS = pd.period_range("2023-01", "2024-12", freq="M").astype(str)
KEYS = ["territory_id", "origin", "target", "horizon"]


def category_panel(raw, category):
    return (
        raw[raw.category == category]
        .pivot(index="territory_id", columns="date", values="value")
        .reindex(columns=MONTHS)
        .sort_index()
    )


def fixed_cohort(raw):
    """Freeze the total-category 2023-only cohort across all category comparisons."""
    return eligible_ids(category_panel(raw, "Все категории"))


def category_summary(frame):
    rows = []
    for (category, h, model), g in frame.groupby(["category", "horizon", "model"]):
        rows.append(
            {
                "category": category,
                "horizon": h,
                "model": model,
                "dates": g.target.nunique(),
                "municipalities": g.territory_id.nunique(),
                "pairs": len(g),
                "MAE": g.groupby("target").ae.mean().mean(),
            }
        )
    return pd.DataFrame(rows)


def write_results(full, sample, cfg, started):
    OUT.mkdir(exist_ok=True, parents=True)
    full.to_parquet(OUT / "full_bridge_predictions.parquet", index=False)
    category_summary(full).to_csv(OUT / "full_bridge_summary.csv", index=False)
    if len(sample):
        sample.to_parquet(OUT / "sample_predictions.parquet", index=False)
        category_summary(sample).to_csv(OUT / "sample_summary.csv", index=False)
    legacy = pd.read_parquet(ROOT / "results/category_predictions.parquet")
    legacy = (
        legacy[legacy.model == "prophet"].copy().rename(columns={"model": "old_model"})
    )
    match = legacy.merge(
        full[full.model.isin(["hierarchy05_growth_bridge", "hierarchy05"])],
        on=["category"] + KEYS,
        suffixes=("_old", ""),
    )
    if not np.array_equal(match.actual_old, match.actual):
        raise ValueError("legacy actual mismatch")
    rows = []
    for (category, h, model), g in match.groupby(["category", "horizon", "model"]):
        a = (
            g.assign(ae=abs(g.actual - g.predicted_old))
            .groupby("target")
            .ae.mean()
            .mean()
        )
        b = g.groupby("target").ae.mean().mean()
        rows.append(
            {
                "category": category,
                "horizon": h,
                "model": model,
                "pairs": len(g),
                "dates": g.target.nunique(),
                "municipalities": g.territory_id.nunique(),
                "legacy_prophet_MAE": a,
                "MAE": b,
                "reduction_pct": 100 * (1 - b / a),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "matched_legacy_summary.csv", index=False)
    summary = category_summary(full)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for h, ax in zip(cfg["horizons"], axes.flat):
        tab = summary[summary.horizon == h].pivot(
            index="category", columns="model", values="MAE"
        )
        reduction = 100 * (1 - tab.hierarchy05_growth_bridge / tab.hierarchy05)
        ax.barh(reduction.index, reduction.values)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"h={h}")
        ax.set_xlabel("MAE reduction from fixed 17.2% versus g=0, %")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"fixed_growth_effect.{ext}", dpi=160)
    lines = [
        "# Неизменённая модель на шести категориях",
        "",
        "Фактические категории: "
        + ", ".join(sorted(full.category.unique()))
        + ". Во всех категориях один источник роста 17,2%, региональный вес 0,5, минимальный регион 10 МО; g=0 — диагностический контроль. Когорта 2075 МО выбрана исключительно по положительным конечным значениям «Все категории» за 2023. Дополнительная проверка категории использует только её 2023; пропуски origin/history/target учитываются отдельно, поэтому число оцениваемых МО меняется. Это проверка устойчивости в тех же данных, не независимая генерализация.",
        "",
        "MAE — среднее по датам средних ошибок МО; результаты полного bridge/g=0 ниже.",
        "",
        "| Категория | h | МО / пары / даты | MAE g=0 | MAE g=17,2% | Изменение MAE, % |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for (cat, h), g in summary.groupby(["category", "horizon"]):
        a = g[g.model == "hierarchy05"].iloc[0]
        b = g[g.model == "hierarchy05_growth_bridge"].iloc[0]
        lines.append(
            f"| {cat} | {h} | {b.municipalities} / {b.pairs} / {b.dates} | {a.MAE:.3f} | {b.MAE:.3f} | {100 * (b.MAE / a.MAE - 1):+.2f} |"
        )
    if len(sample):
        ss = category_summary(sample)
        lines += [
            "",
            "Сопоставимые замороженные Prophet-варианты: фиксированная 256-МО выборка исходного протокола, все доступные даты/четыре горизонта. Пары моделей внутри категории строго одинаковы; три варианта и параметры не подбираются. «Все категории» берётся из готового полного артефакта. sample_summary.csv — отдельная выборочная оценка, не полная панель.",
            "",
            "| Категория | h | МО / пары | Снижение MAE bridge к pooled Prophet, % |",
            "|---|---:|---|---:|",
        ]
        for (cat, h), g in ss.groupby(["category", "horizon"]):
            b = g[g.model == "hierarchy05_growth_bridge"]
            a = g[g.model == "prophet_pooled_profile"]
            if len(a) and len(b):
                a = a.iloc[0]
                b = b.iloc[0]
                lines.append(
                    f"| {cat} | {h} | {b.municipalities} / {b.pairs} | {100 * (1 - b.MAE / a.MAE):+.2f} |"
                )
    lines += [
        "",
        "matched_legacy_summary.csv дополнительно сопоставляет 128-МО старый Prophet с bridge на точно тех же шести поздних датах h1/3/6. Этот старый Prophet не равен замороженной усиленной pooled_profile-модели.",
        "",
        "Повторное использование 2024 и поствыбор сохраняются. Рост зарплаты переносится на каждую категорию с эластичностью 1 без эконометрической оценки; это сильная гипотеза, которую результаты могут опровергать. h12 имеет одну дату, временная значимость не оценивается. Основная модель не заменена.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    inputs = [
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "results/asof_cohort_protocol.json",
        "results/category_predictions.parquet",
        "reports/growth_bridge_review/predictions.parquet",
        "configs/category_growth_review.json",
        "configs/full_cohort_review.json",
        cfg["source_record"],
        "src/sberindex/forecasting/category_growth_review.py",
        "src/sberindex/forecasting/full_cohort_review.py",
        "src/sberindex/forecasting/prophet_seasonality.py",
        "src/sberindex/forecasting/hierarchical_review.py",
        "docs/protocols/CATEGORY_GROWTH_REVIEW.md",
    ]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = {
        "status": "passed_full_bridge_sample_completed"
        if len(sample)
        else "passed_full_bridge_prophet_pending",
        "seconds": time.monotonic() - started,
        "frozen_cohort_2023": 2075,
        "category_labels": sorted(full.category.unique()),
        "parameters_tuned": False,
        "independent_generalization": False,
        "independent_time_validation": False,
        "growth_pct": 17.2,
        "reporting_lag_assumption": 0,
        "prophet_full_new_categories_complete": False,
        "input_sha256": {p: sha(ROOT / p) for p in inputs},
        "output_sha256": {
            p.name: sha(p)
            for p in OUT.iterdir()
            if p.is_file() and p.name != "audit.json"
        },
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )


def checkpoint_manifest(directory, identity):
    """Reject resumptions with changed scientific inputs or backend configuration."""
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("checkpoint scientific inputs/configuration changed")
    elif any(directory.glob("*.parquet")):
        raise ValueError("existing checkpoints lack an input manifest")
    else:
        path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-only", action="store_true")
    parser.add_argument("--full-pooled", action="store_true")
    parser.add_argument("--sample-all-variants", action="store_true")
    args = parser.parse_args()
    args.full_pooled = args.full_pooled or not args.sample_all_variants
    started = time.monotonic()
    cfg = json.loads((ROOT / "configs/category_growth_review.json").read_text())
    pcfg = json.loads((ROOT / "configs/full_cohort_review.json").read_text())
    pcfg["workers"] = cfg["workers"]
    if args.full_pooled:
        pcfg["models"] = ["prophet_pooled_profile"]
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    ids = fixed_cohort(raw)
    geo = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    regions = geo[geo.year == 2023].set_index("territory_id").region_code
    source = json.loads((ROOT / cfg["source_record"]).read_text())
    rates = {o: growth_for_origin(MONTHS[o], source) for o in range(11, 23)}
    samples = set(
        json.loads((ROOT / "results/asof_cohort_protocol.json").read_text())[
            "sample_ids"
        ]
    )
    frames = []
    coverage = []
    panels = {}
    profiles = {}
    valid_pairs = {}
    for category in sorted(raw.category.unique()):
        panel = category_panel(raw, category)
        panels[category] = panel
        # Category-specific eligibility uses2023 only, within the fixed total cohort.
        eligible = ids.intersection(eligible_ids(panel))
        gp, rp = fit_profiles(
            panel.loc[eligible], regions, cfg["minimum_region_municipalities"]
        )
        profiles[category] = gp
        mutated = panel.loc[eligible].copy()
        mutated.iloc[:, 12:] = 1e9
        gp2, rp2 = fit_profiles(mutated, regions, cfg["minimum_region_municipalities"])
        np.testing.assert_array_equal(gp, gp2)
        assert all(np.array_equal(rp[k], rp2[k]) for k in rp)
        rows = []
        for city in eligible:
            values = panel.loc[city].to_numpy(float)
            pairs, reasons = evaluation_pairs(values, cfg["horizons"])
            valid_pairs[(category, int(city))] = pairs
            coverage.append(
                dict(
                    category=category,
                    territory_id=int(city),
                    eligible_2023=True,
                    **reasons,
                )
            )
            for origin, h in pairs:
                target = origin + h
                pred = predict_profile(
                    values[origin], origin, target, regions.loc[city], gp, rp, 0.5
                )
                rate = rates[origin]
                info = {
                    "category": category,
                    "territory_id": int(city),
                    "origin": MONTHS[origin],
                    "target": MONTHS[target],
                    "horizon": h,
                    "actual": values[target],
                }
                for model, p in [
                    ("hierarchy05", pred),
                    (
                        "hierarchy05_growth_bridge",
                        pred * (1 + rate) ** ((origin % 12 + h) // 12),
                    ),
                ]:
                    rows.append(
                        dict(
                            **info, model=model, predicted=p, ae=abs(values[target] - p)
                        )
                    )
        frames.append(pd.DataFrame(rows))
    full = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    write_results(full, pd.DataFrame(), cfg, started)
    print(
        f"Full bridge completed: {len(full)} predictions, {time.monotonic() - started:.1f}s",
        flush=True,
    )
    if args.bridge_only:
        return
    checkpoints = OUT / (
        "full_pooled_checkpoints" if args.full_pooled else "sample_checkpoints"
    )
    checkpoints.mkdir(exist_ok=True)
    identity_inputs = [
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "results/asof_cohort_protocol.json",
        cfg["source_record"],
        "src/sberindex/forecasting/prophet_seasonality.py",
        "src/sberindex/forecasting/full_cohort_review.py",
        "src/sberindex/forecasting/hierarchical_review.py",
    ]
    identity = {
        "prophet_config": pcfg,
        "category_config": cfg,
        "full_pooled": args.full_pooled,
        "input_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in identity_inputs
        },
    }
    checkpoint_manifest(checkpoints, identity)
    forecasts = []
    launched_fits = 0
    saved = pd.read_parquet(ROOT / "reports/growth_bridge_review/predictions.parquet")
    saved = saved[saved.model.isin(pcfg["models"])]
    if not args.full_pooled:
        saved = saved[saved.territory_id.isin(samples)]
    forecasts.append(
        saved.assign(category="Все категории")[
            ["category"] + KEYS + ["actual", "model", "predicted", "ae"]
        ]
    )
    for category in sorted(raw.category.unique()):
        if category == "Все категории":
            continue
        candidates = sorted(
            int(i)
            for i in ids
            if (category, int(i)) in valid_pairs and (args.full_pooled or i in samples)
        )
        jobs = []
        meta = {}
        for city in candidates:
            path = checkpoints / f"{category}_{city}.parquet"
            if path.exists():
                forecasts.append(pd.read_parquet(path))
                continue
            pairs = valid_pairs[(category, city)]
            values = panels[category].loc[city].to_numpy(float)
            for origin in sorted({o for o, _ in pairs}):
                horizons = [h for o, h in pairs if o == origin]
                job = (
                    city,
                    origin,
                    values[: origin + 1],
                    profiles[category],
                    max(horizons),
                    pcfg,
                )
                jobs.append(job)
                meta[(city, origin)] = horizons
        print(
            f"{category}: {len(jobs)} origins x {len(pcfg['models'])} Prophet variants pending",
            flush=True,
        )
        cityrows = {}
        pending = {
            city: sum(j[0] == city for j in jobs)
            for city in candidates
            if not (checkpoints / f"{category}_{city}.parquet").exists()
        }
        launched_fits += len(jobs) * len(pcfg["models"])
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            futures = [pool.submit(refit, job) for job in jobs]
            for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
                (city, origin), preds = future.result()
                values = panels[category].loc[city].to_numpy(float)
                records = cityrows.setdefault(city, [])
                for h in meta[(city, origin)]:
                    info = {
                        "category": category,
                        "territory_id": city,
                        "origin": MONTHS[origin],
                        "target": MONTHS[origin + h],
                        "horizon": h,
                        "actual": values[origin + h],
                    }
                    for model, p in preds.items():
                        records.append(
                            dict(
                                **info,
                                model=model,
                                predicted=float(p[h - 1]),
                                ae=abs(values[origin + h] - p[h - 1]),
                            )
                        )
                pending[city] -= 1
                if pending[city] == 0:
                    result = pd.DataFrame(cityrows.pop(city))
                    path = checkpoints / f"{category}_{city}.parquet"
                    temp = path.with_suffix(".tmp.parquet")
                    result.to_parquet(temp, index=False)
                    temp.replace(path)
                    forecasts.append(result)
                if n % 250 == 0 or n == len(jobs):
                    print(
                        f"{category}: {n}/{len(jobs)} origins, total elapsed {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    prophet = pd.concat(forecasts, ignore_index=True)
    bridge = full.merge(
        prophet[prophet.model == pcfg["models"][0]][["category"] + KEYS],
        on=["category"] + KEYS,
        validate="many_to_one",
    )
    # All frozen variants have the exact same keys within this evaluation.
    for model in pcfg["models"]:
        if len(prophet[prophet.model == model]) != len(bridge) // 2:
            raise ValueError("incomplete Prophet keys")
    result = pd.concat([bridge, prophet], ignore_index=True)
    if args.full_pooled:
        result.to_parquet(OUT / "full_pooled_predictions.parquet", index=False)
        category_summary(result).to_csv(OUT / "full_pooled_summary.csv", index=False)
        fulltext = [
            "\n## Полный замороженный pooled_profile Prophet\n",
            "Один усиленный comparator выбран заранее: prophet_pooled_profile, без изменения параметров или категорийных настроек. Существующий основной прогноз не заменён. Все оцениваемые пары каждой категории совпадают.\n",
            "| Категория | h | МО / пары / даты | Prophet MAE | Bridge MAE | Снижение MAE, % |",
            "|---|---:|---|---:|---:|---:|",
        ]
        summary = category_summary(result)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for h, axis in zip(cfg["horizons"], axes.flat):
            table = summary[summary.horizon == h].pivot(
                index="category", columns="model", values="MAE"
            )
            reduction = 100 * (
                1 - table.hierarchy05_growth_bridge / table.prophet_pooled_profile
            )
            axis.barh(reduction.index, reduction.values)
            axis.axvline(0, color="black", lw=0.8)
            axis.set_title(f"h={h}")
            axis.set_xlabel("Bridge MAE reduction versus frozen pooled Prophet, %")
        figure.tight_layout()
        for ext in ["png", "svg"]:
            figure.savefig(OUT / f"full_pooled_comparison.{ext}", dpi=160)
        for (cat, h), g in summary.groupby(["category", "horizon"]):
            a = g[g.model == "prophet_pooled_profile"].iloc[0]
            b = g[g.model == "hierarchy05_growth_bridge"].iloc[0]
            fulltext.append(
                f"| {cat} | {h} | {b.municipalities} / {b.pairs} / {b.dates} | {a.MAE:.3f} | {b.MAE:.3f} | {100 * (1 - b.MAE / a.MAE):+.2f} |"
            )
        with (OUT / "REPORT.md").open("a") as report:
            report.write("\n".join(fulltext) + "\n")
        audit = json.loads((OUT / "audit.json").read_text())
        audit.update(
            status="passed",
            prophet_full_new_categories_complete=True,
            seconds=time.monotonic() - started,
            prophet_frozen_variants_full=["prophet_pooled_profile"],
            workers=cfg["workers"],
        )
        timing_path = OUT / "fit_run_timing.json"
        if not timing_path.exists():
            timing_path.write_text(
                json.dumps(
                    {
                        "original_full_fit_seconds": time.monotonic() - started,
                        "new_prophet_fits": launched_fits,
                    },
                    indent=2,
                )
                + "\n"
            )
        audit.update(
            new_prophet_fits_this_run=launched_fits,
            full_category_origin_count=len(
                prophet[prophet.category != "Все категории"][
                    ["category", "territory_id", "origin"]
                ].drop_duplicates()
            ),
            fit_run_timing=json.loads(timing_path.read_text()),
        )
        audit["input_sha256"] = {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in audit["input_sha256"]
        }
        audit["output_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.iterdir()
            if p.is_file() and p.name != "audit.json"
        }
        (OUT / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
        )
    else:
        write_results(full, result, cfg, started)
    print(category_summary(result).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
