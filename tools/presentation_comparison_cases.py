"""Export matched cached comparisons and causal illustrative municipality intervals.

No model fitting, tuning, test-error case selection, or transfer of other model bands.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/presentation_comparison_cases"
KEYS = ["territory_id", "origin", "target", "horizon"]
GROWTH = "reports/growth_bridge_review/predictions.parquet"
DIRECT = "reports/direct_horizon_review/predictions.parquet"
FOUNDATION = "reports/foundation_covariate_review/predictions.parquet"
INPUTS = [
    "tools/presentation_comparison_cases.py",
    "tests/test_presentation_comparison_cases.py",
    GROWTH,
    DIRECT,
    FOUNDATION,
    "data/consumption.parquet",
    "results/municipal_lookup.csv",
    "reports/growth_bridge_review/selection.json",
]
MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "prophet_disabled": "Prophet: yearly disabled",
    "prophet_pooled_profile": "Prophet + pooled seasonal profile",
    "prophet_yearly3": "Prophet: yearly Fourier order 3",
    "seasonal_pooled": "Seasonal profile",
    "hierarchy05": "Hierarchy 0.5",
    "direct_lags": "Direct HGB (lags)",
    "chronos2_univariate": "Chronos-2 univariate",
    "chronos2_profile": "Chronos-2 + seasonal covariate",
    "hierarchy05_growth_bridge": "Selected hierarchy 0.5 + growth bridge",
}
GROWTH_MODELS = [
    m
    for m in MODEL_LABELS
    if m not in {"direct_lags", "chronos2_univariate", "chronos2_profile"}
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def metrics(frame: pd.DataFrame) -> dict:
    """Equal target-month MAE; pooled SSE / within-MO target variation R2."""
    error = frame.actual - frame.predicted
    sse = float((error**2).sum())
    pooled_ss = float(((frame.actual - frame.actual.mean()) ** 2).sum())
    centered = frame.actual - frame.groupby("territory_id").actual.transform("mean")
    within_ss = float((centered**2).sum())
    per_mo = frame.groupby("territory_id").actual.agg(["size", "nunique"])
    valid_yoy = frame.year_ago.gt(0)
    yoy = 100 * error[valid_yoy].abs() / frame.year_ago[valid_yoy]
    yoy_month = yoy.groupby(frame.loc[valid_yoy, "target"]).mean()
    return {
        "MAE": float(error.abs().groupby(frame.target).mean().mean()),
        "R2_pooled": 1 - sse / pooled_ss if pooled_ss > 0 else None,
        "R2_within_MO": 1 - sse / within_ss if within_ss > 0 else None,
        "within_MO_denominator": within_ss,
        "within_MO_undefined_count": int((per_mo["nunique"] <= 1).sum()),
        "YoY_MAE_pp": float(yoy_month.mean()),
        "observations": len(frame),
        "municipalities": int(frame.territory_id.nunique()),
        "dates": int(frame.target.nunique()),
        "target_start": frame.target.min(),
        "target_end": frame.target.max(),
        "origin_start": frame.origin.min(),
        "origin_end": frame.origin.max(),
    }


def finite_sample_quantile(scores: np.ndarray, coverage: float = 0.8) -> float:
    """Conformal-style rank correction; no exchangeability guarantee is claimed."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("No finite calibration residuals")
    rank = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    return float(np.partition(scores, rank - 1)[rank - 1])


def calibration_for_origin(
    predictions: pd.DataFrame, origin: str, scale: pd.Series, coverage: float = 0.8
) -> dict | None:
    """Use three completed months ending at origin; origin expense is observed."""
    months = pd.period_range(end=origin, periods=3, freq="M").astype(str).tolist()
    cal = predictions[predictions.target.isin(months)].copy()
    if set(cal.target.unique()) != set(months):
        return None
    normalized = (cal.actual - cal.predicted).abs() / cal.territory_id.map(scale)
    q = finite_sample_quantile(normalized.to_numpy(), coverage)
    return dict(
        origin=origin,
        calibration_start=months[0],
        calibration_end=months[-1],
        calibration_months=3,
        observations=len(cal),
        municipalities=int(cal.territory_id.nunique()),
        normalized_radius=q,
        nominal_coverage=coverage,
        max_calibration_target=cal.target.max(),
    )


def matched_rows(frames: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    key_sets = [set(map(tuple, f[KEYS].to_numpy())) for f in frames.values()]
    common = set.intersection(*key_sets)
    keys = pd.DataFrame(sorted(common), columns=KEYS)
    matched = {
        name: f.merge(keys, on=KEYS, validate="one_to_one")
        for name, f in frames.items()
    }
    truth = None
    for name, f in matched.items():
        actual = f.set_index(KEYS).actual.sort_index()
        if truth is None:
            truth = actual
        elif not np.allclose(truth.to_numpy(), actual.to_numpy(), atol=0, rtol=0):
            raise ValueError(f"Actuals mismatch for {name}")
    return matched, keys


def build(root: Path = ROOT) -> dict:
    growth = pd.read_parquet(root / GROWTH)
    direct = pd.read_parquet(root / DIRECT)
    foundation = pd.read_parquet(root / FOUNDATION)
    foundation = foundation[foundation.category.eq("Все категории")]
    frames = {m: growth[growth.model.eq(m)].copy() for m in GROWTH_MODELS}
    direct_model = direct[
        direct.model.isin(
            ["direct_lags", "direct_lags__seasonal_fallback_not_trainable"]
        )
    ].copy()
    direct_model["saved_model"] = direct_model.model
    frames["direct_lags"] = direct_model
    for m in ["chronos2_univariate", "chronos2_profile"]:
        frames[m] = foundation[foundation.model.eq(m)].copy()
    matched, pairs = matched_rows(frames)
    rows = []
    for cohort, groups in [
        ("common_cached_pairs", matched),
        ("full_growth_cohort", {m: frames[m] for m in GROWTH_MODELS}),
    ]:
        for model, frame in groups.items():
            for horizon, group in frame.groupby("horizon"):
                is_fallback = model == "direct_lags" and horizon == 12
                computed = metrics(group)
                row = dict(
                    cohort=cohort,
                    model=model,
                    label=MODEL_LABELS[model],
                    horizon=int(horizon),
                    status="seasonal_fallback_not_trained"
                    if is_fallback
                    else "saved_prediction",
                    **computed,
                )
                if is_fallback:
                    row["fallback_MAE"] = row["MAE"]
                    row["fallback_R2_pooled"] = row["R2_pooled"]
                    row["MAE"] = None
                    row["R2_pooled"] = None
                    row["R2_within_MO"] = None
                row["source"] = (
                    DIRECT
                    if model == "direct_lags"
                    else FOUNDATION
                    if model.startswith("chronos2")
                    else GROWTH
                )
                rows.append(row)
    raw = pd.read_parquet(root / "data/consumption.parquet")
    raw = raw[raw.category.eq("Все категории")]
    scale = (
        raw[raw.date.between("2023-01", "2023-12")].groupby("territory_id").value.mean()
    )
    lookup = pd.read_csv(root / "results/municipal_lookup.csv")
    lookup = (
        lookup[lookup.year.eq(2023)]
        .drop_duplicates("territory_id")
        .set_index("territory_id")
    )
    selected = frames["hierarchy05_growth_bridge"]
    h1 = selected[selected.horizon.eq(1)].copy()
    complete = h1.groupby("territory_id").target.nunique()
    eligible = lookup.loc[
        lookup.index.intersection(complete[complete.eq(12)].index)
    ].copy()
    eligible["mean_2023"] = scale.reindex(eligible.index)
    cities = eligible[
        eligible.municipal_district_status.eq("административный_центр_субъекта")
    ]
    ufa = cities[cities.municipal_district_name_short.eq("Уфа")]
    if len(ufa) != 1:
        raise ValueError("Fixed large-city Ufa example is not uniquely identified")
    city_id = int(ufa.index[0])
    districts = eligible[
        eligible.municipal_district_type.isin(
            ["муниципальный округ", "муниципальный район"]
        )
    ]
    small_id = int(districts.sort_values(["mean_2023"], kind="stable").index[0])
    case_specs = [
        ("regional_capital", city_id),
        ("low_expense_municipality", small_id),
        ("orsk_stress_case", 1673),
    ]
    calibrations = [
        calibration_for_origin(h1, o, scale) for o in sorted(h1.origin.unique())
    ]
    calibrations = [c for c in calibrations if c is not None]
    cal_by_origin = {c["origin"]: c for c in calibrations}
    cases = []
    case_csv = []
    for role, tid in case_specs:
        geo = lookup.loc[tid]
        point = h1[h1.territory_id.eq(tid)].sort_values("target")
        if len(point) != 12:
            raise ValueError(f"Case {tid} lacks complete saved h1 predictions")
        months = []
        for r in point.itertuples():
            cal = cal_by_origin.get(r.origin)
            radius = cal["normalized_radius"] * scale.loc[tid] if cal else None
            month = dict(
                target=r.target,
                origin=r.origin,
                actual=float(r.actual),
                predicted=float(r.predicted),
                lower=max(0.0, r.predicted - radius) if radius is not None else None,
                upper=float(r.predicted + radius) if radius is not None else None,
                calibration_start=cal["calibration_start"] if cal else None,
                calibration_end=cal["calibration_end"] if cal else None,
            )
            months.append(month)
            case_csv.append(
                dict(
                    territory_id=tid,
                    role=role,
                    model="hierarchy05_growth_bridge",
                    **month,
                )
            )
        cases.append(
            dict(
                role=role,
                territory_id=tid,
                name=geo.municipal_district_name_short,
                full_name=geo.municipal_district_name,
                region=geo.region_name,
                mean_expense_2023=float(scale.loc[tid]),
                model="hierarchy05_growth_bridge",
                horizon=1,
                title=f"{geo.municipal_district_name_short}: rolling h1, 2024",
                months=months,
            )
        )
    caveats = [
        "Common rows use identical municipality/origin/target/horizon pairs and exactly equal saved actuals. No model was refitted.",
        "Full growth cohort and common cached cohort are separate panels; do not rank full and common values against each other.",
        "Direct HGB is the predeclared lags variant. At h12 it was not trainable; the seasonal fallback is separately stored and the HGB MAE cell is null.",
        "Chronos-2 covariates means the saved univariate seasonal-profile variant, not the grouped category variant.",
        "Pooled seasonal/profile training pools differ across research runs. Here baseline/Prophet/hierarchy predictions are the saved full-cohort growth-run forecasts, restricted to common evaluation pairs.",
        "MAE averages within each target month then weights target months equally. Within-MO R2 = 1 - pooled prediction SSE / sum of within-municipality centered actual squares; it is not the mean municipality R2.",
        "h12 has one target month per municipality: all within-MO R2 are undefined, represented by null. Constant-target municipality counts are explicit.",
        "Cases are predetermined by metadata and 2023 mean expenditure, not forecast error: fixed Ufa regional capital, lowest 2023 expenditure municipal district/okrug, and fixed Orsk.",
        "The expenditure proxy is per-consumer expense in rubles, not population or city size. Ufa is labelled regional capital; low expense is not evidence of small population.",
        "Example intervals are NEW illustrative global scaled residual bands for the selected growth curve. They use only preceding three completed target months of that SAME saved model, 2023 scale, and an 80% rank-corrected residual quantile. Zero reporting lag is assumed.",
        "Intervals are posthoc illustrations on reused 2024; temporal/dependent residuals and model selection preclude independent coverage guarantees. They are not copied from blend_75, and they are not a validated production uncertainty model.",
        "Jan–Mar have insufficient three-month saved residual history: bands are absent. h1 points are rolling one-month forecasts, not a single Jan-origin 12-month forecast.",
    ]
    out = root / "reports/presentation_comparison_cases"
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "common_pairs.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(case_csv).to_csv(out / "case_months.csv", index=False)
    pd.DataFrame(calibrations).to_csv(out / "calibration.csv", index=False)
    sources = {p: sha256(root / p) for p in INPUTS}
    payload = dict(
        schema_version=1,
        comparison_rows=rows,
        cases=cases,
        interval_calibration=calibrations,
        source_caveats=caveats,
        source_sha256=sources,
        common_pair_counts=pairs.groupby("horizon").size().to_dict(),
        common_pairs_sha256=sha256(out / "common_pairs.csv"),
        metrics_definition="Equal target-month MAE; within-MO R2 uses pooled SSE over within-MO actual variation",
        interval_method="global_scaled_2023_previous_3_completed_months_80pct_rank_quantile",
        case_selection="fixed Ufa regional capital; municipal district minimum 2023 mean expenditure; fixed Orsk",
    )
    (out / "evidence.json").write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    report = "# Matched presentation comparison and forecast cases\n\n"
    report += "\n\n".join(caveats) + "\n\n## Common cached pairs: MAE (rubles)\n\n"
    report += "| Model | h1 | h3 | h6 | h12 |\n| --- | --- | --- | --- | --- |\n"
    for model, label in MODEL_LABELS.items():
        cells = []
        for h in [1, 3, 6, 12]:
            r = next(
                r
                for r in rows
                if r["cohort"] == "common_cached_pairs"
                and r["model"] == model
                and r["horizon"] == h
            )
            cells.append(
                f"{r['MAE']:.1f}"
                if r["MAE"] is not None
                else "not trained (fallback only)"
            )
        report += "| " + " | ".join([label, *cells]) + " |\n"
    report += (
        "\n## Cases\n\n"
        + "\n".join(
            f"- {c['role']}: {c['name']} (ID {c['territory_id']}), mean 2023 expense {c['mean_expense_2023']:.1f} RUB."
            for c in cases
        )
        + "\n"
    )
    (out / "REPORT.md").write_text(report)
    audit = dict(
        status="passed",
        models_refitted=False,
        cached_common_actuals_exactly_equal=True,
        no_saved_blend_interval_transferred=True,
        interval_calibration_max_target_at_or_before_origin=all(
            c["max_calibration_target"] <= c["origin"] for c in calibrations
        ),
        independent_interval_validation=False,
        case_selection_uses_test_error=False,
        input_sha256=sources,
        output_sha256={
            p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != "audit.json"
        },
    )
    (out / "audit.json").write_text(json.dumps(safe(audit), indent=2) + "\n")
    return payload


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            dict(
                output=str(OUT),
                rows=len(result["comparison_rows"]),
                cases=[c["name"] for c in result["cases"]],
                common_pair_counts=result["common_pair_counts"],
            ),
            ensure_ascii=False,
        )
    )
