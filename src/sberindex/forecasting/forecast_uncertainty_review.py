"""Paired, date-balanced cross-sectional uncertainty of frozen forecasts."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t

from sberindex.paths import ROOT

KEYS = ["territory_id", "origin", "target", "horizon"]


def paired_losses(frame, reference, challenger):
    a = frame[frame.model == reference][KEYS + ["actual", "predicted"]]
    b = frame[frame.model == challenger][KEYS + ["actual", "predicted"]]
    if a.duplicated(KEYS).any() or b.duplicated(KEYS).any():
        raise ValueError("duplicate forecast keys")
    joined = a.merge(
        b,
        on=KEYS,
        suffixes=("_a", "_b"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined._merge.eq("both").all() or not np.array_equal(
        joined.actual_a, joined.actual_b
    ):
        raise ValueError(
            "forecasts must have exactly the same evaluation pairs and actuals"
        )
    joined["loss_a"] = abs(joined.actual_a - joined.predicted_a)
    joined["loss_b"] = abs(joined.actual_b - joined.predicted_b)
    return joined


def date_balanced_metric(frame, weights=None):
    weights = np.ones(len(frame)) if weights is None else np.asarray(weights, float)
    work = frame.assign(w=weights, wa=weights * frame.loss_a, wb=weights * frame.loss_b)
    sums = work.groupby("target")[["w", "wa", "wb"]].sum()
    valid = sums.w > 0
    if not valid.all():
        return np.array([np.nan, np.nan])
    a, b = (sums[["wa", "wb"]].div(sums.w, axis=0).mean()).to_numpy()
    return np.array([b - a, 100 * (1 - b / a)])


def cluster_bootstrap(frame, cluster, draws=2500, seed=20261007):
    """Whole cluster resampling preserves every observed date and unbalanced counts."""
    grouped = frame.groupby([cluster, "target"]).agg(
        a=("loss_a", "sum"), b=("loss_b", "sum"), n=("loss_a", "size")
    )
    clusters = sorted(frame[cluster].unique())
    dates = sorted(frame.target.unique())
    arrays = [
        grouped[name]
        .unstack()
        .reindex(index=clusters, columns=dates)
        .fillna(0)
        .to_numpy()
        for name in ["a", "b", "n"]
    ]
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, draws, 100):
        w = rng.multinomial(
            len(clusters),
            np.full(len(clusters), 1 / len(clusters)),
            size=min(100, draws - start),
        )
        denom = w @ arrays[2]
        a = np.mean((w @ arrays[0]) / denom, axis=1)
        b = np.mean((w @ arrays[1]) / denom, axis=1)
        values.append(np.column_stack([b - a, 100 * (1 - b / a)]))
    sampled = np.concatenate(values)
    if not np.isfinite(sampled).all():
        raise ValueError("bootstrap draw has an empty target date")
    return sampled, len(clusters)


def dm_hac(differences, horizon):
    """Exploratory HLN corrected DM with Bartlett HAC; conservative T gate."""
    d = np.asarray(differences, float)
    n, lag = len(d), horizon - 1
    result = {
        "T": n,
        "hac_lag": lag,
        "dm_stat": np.nan,
        "p_value": np.nan,
        "temporal_ci_low": np.nan,
        "temporal_ci_high": np.nan,
    }
    if n < max(8, 2 * horizon + 1):
        return dict(
            **result,
            dm_status="undefined_single_date"
            if n < 2
            else "not_reported_minimum_dates_policy",
        )
    centered = d - d.mean()
    lrv = float(centered @ centered / n)
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k] / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    if not np.isfinite(lrv) or lrv <= 0 or correction <= 0:
        return dict(**result, dm_status="undefined_degenerate_hac")
    se = np.sqrt(lrv / n) / correction
    statistic = float(d.mean() / se)
    critical = t.ppf(0.975, n - 1)
    result.update(
        dm_stat=statistic,
        p_value=float(2 * t.sf(abs(statistic), n - 1)),
        temporal_ci_low=float(d.mean() - critical * se),
        temporal_ci_high=float(d.mean() + critical * se),
    )
    return dict(**result, dm_status="exploratory_small_T_postselection")


def holm(pvalues):
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    indices = np.flatnonzero(np.isfinite(p))
    order = indices[np.argsort(p[indices])]
    if len(order):
        out[order] = np.minimum(
            1, np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1))
        )
    return out


def main():
    config = json.loads((ROOT / "configs/forecast_uncertainty_review.json").read_text())
    frame = pd.read_parquet(ROOT / config["predictions"])
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    regions = lookup[lookup.year == 2023].set_index("territory_id").region_code
    rows = []
    for h in config["horizons"]:
        for prophet in config["comparators"]:
            paired = paired_losses(
                frame[frame.horizon == h], prophet, config["challenger"]
            )
            mapped = paired.territory_id.map(regions)
            if mapped.isna().any():
                raise ValueError("region missing; cannot silently discard pairs")
            paired["region_code"] = mapped
            point = date_balanced_metric(paired)
            dm = dm_hac(
                paired.assign(d=paired.loss_b - paired.loss_a)
                .groupby("target")
                .d.mean()
                .to_numpy(),
                h,
            )
            for cluster in ["territory_id", "region_code"]:
                samples, n = cluster_bootstrap(
                    paired, cluster, config["draws"], config["seed"]
                )
                lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
                rows.append(
                    dict(
                        horizon=h,
                        comparator=prophet,
                        cluster=cluster,
                        clusters=n,
                        dates=paired.target.nunique(),
                        pairs=len(paired),
                        delta_MAE=point[0],
                        reduction_pct=point[1],
                        delta_low=lo[0],
                        delta_high=hi[0],
                        reduction_low=lo[1],
                        reduction_high=hi[1],
                        **dm,
                    )
                )
    result = pd.DataFrame(rows)
    temporal = result[result.cluster == "territory_id"].copy()
    temporal["p_holm"] = holm(temporal.p_value)
    # Requested after inspecting results: sensitivity, never retroactive preregistration.
    focused = temporal.comparator.eq("prophet_pooled_profile") & temporal.horizon.isin(
        [1, 3]
    )
    temporal["p_holm_two_horizons_exploratory"] = np.nan
    temporal.loc[focused, "p_holm_two_horizons_exploratory"] = holm(
        temporal.loc[focused, "p_value"]
    )
    out = ROOT / "reports/forecast_uncertainty_review"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "cluster_intervals.csv", index=False)
    temporal.to_csv(out / "temporal_tests.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = []
    for i, row in result.iterrows():
        ax.errorbar(
            row.reduction_pct,
            i,
            xerr=[
                [row.reduction_pct - row.reduction_low],
                [row.reduction_high - row.reduction_pct],
            ],
            fmt="o",
            color="tab:blue" if row.cluster == "territory_id" else "tab:orange",
        )
        labels.append(
            f"h{row.horizon} {row.comparator.replace('prophet_', '')} / {'MO' if row.cluster == 'territory_id' else 'region'}"
        )
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("MAE reduction versus Prophet, % (95% conditional cluster CI)")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(out / f"conditional_intervals.{ext}", dpi=160)
    text = [
        "# Неопределённость сравнения замороженных прогнозов",
        "",
        "Положительное снижение MAE означает преимущество hierarchy05_growth_bridge; delta MAE = bridge − Prophet. Наблюдаемое снижение MAE к pooled Prophet: 41,53% / 23,34% / 28,49% / 55,80% на h1/3/6/12. После Holm ни одно оцениваемое временное сравнение h1/h3 не достигает уровня 5%. Все даты имеют одинаковый вес; пары строго совпадают. 2500 парных выборок целых МО и отдельно регионов сохраняют все даты каждого кластера. Это условная поперечная неопределённость при фиксированных датах 2024, а не независимое временное подтверждение. Региональная чувствительность допускает зависимость внутри региона, но не общероссийский шок.",
        "",
        "| h | Prophet | Снижение, % | 95% МО | 95% регион |",
        "|---|---|---:|---|---|",
    ]
    for (h, m), g in result.groupby(["horizon", "comparator"]):
        a = g[g.cluster == "territory_id"].iloc[0]
        b = g[g.cluster == "region_code"].iloc[0]
        text.append(
            f"| {h} | {m} | {a.reduction_pct:.2f} | [{a.reduction_low:.2f}; {a.reduction_high:.2f}] | [{b.reduction_low:.2f}; {b.reduction_high:.2f}] |"
        )
    text += [
        "",
        "DM по средним разностям абсолютных ошибок на датах: Bartlett HAC lag=h−1, коррекция Harvey–Leybourne–Newbold, двустороннее t(T−1); Holm по шести оцениваемым сравнениям h1/h3. Минимум T=max(8,2h+1) — консервативное правило данного аудита, не теорема. h6: T=7 и lag=5; тест и временной интервал не сообщаются по заранее заданному правилу минимального числа дат. h12: T=1; временная дисперсия и тест не определены. h1/h3 с T=12/10 остаются исследовательскими: поствыбор параметров и повторное использование 2024 не устраняются p-value.",
        "",
        "Дополнительная чувствительность, добавленная после просмотра результатов: Holm только для pooled Prophet на h1/h3 даёт 0,03406 и 0,30684. h1 ниже 5%, h3 — выше. Исходная поправка по шести тестам сохранена (h1 pooled: 0,10218). Нельзя задним числом объявить узкое семейство заранее заданным; сравнение по категориям не регистрирует гипотезы общего прогноза. T=10 на h3 ограничивает информацию, но недостаточная мощность отдельно не оценена.\n\nСм. temporal_tests.csv для p и двух вариантов Holm. Существующая основная модель не заменена; bootstrap не повторяет выбор модели.",
    ]
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    inputs = [
        config["predictions"],
        "results/municipal_lookup.csv",
        "configs/forecast_uncertainty_review.json",
        "docs/protocols/FORECAST_UNCERTAINTY_REVIEW.md",
        "src/sberindex/forecasting/forecast_uncertainty_review.py",
    ]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "draws": config["draws"],
                "seed": config["seed"],
                "independent_time_validation": False,
                "input_sha256": {p: sha(ROOT / p) for p in inputs},
                "output_sha256": {
                    p.name: sha(p) for p in out.iterdir() if p.name != "audit.json"
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        temporal[
            ["horizon", "comparator", "reduction_pct", "p_value", "p_holm", "dm_status"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
