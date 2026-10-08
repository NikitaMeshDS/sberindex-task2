"""Fixed early-vs-late h2 comparison under one-month reporting delay."""

import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sberindex.paths import ROOT
from sberindex.forecasting.calendar_audit import features, predict_asof
from sberindex.forecasting.foundation_seasonal_audit import known_2023_profile
from sberindex.forecasting.foundation_delay_audit import timing, history_context
from sberindex.forecasting.asof_distribution_audit import (
    paired_metrics,
    validate_actuals,
)

OUT = ROOT / "results"
EARLY = [str(t) for t in pd.period_range("2024-02", "2024-06", freq="M")]
LATE = ["2024-09", "2024-10", "2024-11", "2024-12"]
MODELS = ["hgb_frozen", "seasonal_pooled", "prophet", "blend_75"]
KEYS = ["territory_id", "origin", "target", "horizon"]
CACHE = [
    "operational_early_predictions.parquet",
    "operational_early_protocol.json",
    "operational_early_runtime.csv",
]
INPUTS = [
    "data/consumption.parquet",
    "config.json",
    "results/asof_cohort_protocol.json",
    "results/foundation_delay_predictions.parquet",
    "results/foundation_delay_protocol.json",
    "docs/protocols/OPERATIONAL_EARLY_EXPERIMENT.md",
    "src/sberindex/forecasting/operational_early_audit.py",
    "src/sberindex/forecasting/prophet_backend.py",
    "src/sberindex/forecasting/calendar_audit.py",
    "src/sberindex/forecasting/foundation_seasonal_audit.py",
    "src/sberindex/forecasting/foundation_delay_audit.py",
    "src/sberindex/forecasting/asof_distribution_audit.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def period_for(target):
    if target in EARLY:
        return "early"
    if target in LATE:
        return "late"
    raise ValueError(f"Target outside fixed windows: {target}")


def training_arrays(panel):
    dates = pd.period_range("2023-01", periods=12, freq="M")
    train = panel.loc[:, dates.astype(str)]
    good = np.isfinite(train.to_numpy(float)).all(axis=1) & train.gt(0).all(axis=1)
    values = train.loc[good].to_numpy(float)
    x = np.concatenate(
        [features(np.log(values[:, :t]), dates[t], False) for t in range(3, 12)]
    )
    y = np.concatenate(
        [np.log(values[:, t]) - np.log(values[:, t - 1]) for t in range(3, 12)]
    )
    return x, y, train.index[good].to_numpy()


def inputs(root=ROOT):
    raw = pd.read_parquet(root / "data/consumption.parquet")
    panel = (
        raw[raw.category.eq("Все категории")]
        .pivot(index="territory_id", columns="date", values="value")
        .sort_index()
    )
    profile, eligible = known_2023_profile(panel)
    ids = json.loads((root / "results/asof_cohort_protocol.json").read_text())[
        "sample_ids"
    ]
    assert len(ids) == 256 and len(eligible) == 2075 and set(ids).issubset(eligible)
    return panel, profile, ids, raw


def late_predictions(root=ROOT):
    source = pd.read_parquet(root / "results/foundation_delay_predictions.parquet")
    return source[source.model.isin(MODELS) & source.reporting_delay_months.eq(1)][
        KEYS + ["model", "actual", "predicted"]
    ].copy()


def scored(frame, panel):
    frame = frame[KEYS + ["model", "actual", "predicted"]].copy()
    frame["territory_id"] = frame.territory_id.astype("int64")
    frame["horizon"] = frame.horizon.astype("int64")
    frame["period"] = frame.target.map(period_for)
    frame["reporting_delay_months"] = 1
    frame["business_horizon_months"] = 1
    frame["decision_month"] = (pd.PeriodIndex(frame.target, freq="M") - 1).astype(str)
    frame["absolute_error"] = (frame.actual - frame.predicted).abs()
    means = panel.loc[:, panel.columns.str.startswith("2023-")].mean(axis=1)
    frame["normalized_error_2023"] = frame.absolute_error / frame.territory_id.map(
        means
    )
    return frame.sort_values(["period", "model", "target", "territory_id"]).reset_index(
        drop=True
    )


def summaries(frame):
    summary, monthly, paired, sensitivity = [], [], [], []
    for (period, model), group in frame.groupby(["period", "model"], sort=True):
        labels = {
            "period": period,
            "model": model,
            "business_horizon_months": 1,
            "reporting_delay_months": 1,
            "effective_horizon": 2,
        }
        summary.append(
            {
                **labels,
                "MAE_date_balanced": float(
                    group.groupby("target").absolute_error.mean().mean()
                ),
                "MAE_pooled": float(group.absolute_error.mean()),
                "R2_pooled": float(r2_score(group.actual, group.predicted)),
                "WAPE_pct": float(
                    100 * group.absolute_error.sum() / group.actual.sum()
                ),
                "NMAE_2023_pct": float(
                    100 * group.groupby("target").normalized_error_2023.mean().mean()
                ),
                "dates": group.target.nunique(),
                "municipalities": group.territory_id.nunique(),
                "observations": len(group),
            }
        )
        for target, part in group.groupby("target"):
            monthly.append(
                {
                    **labels,
                    "target": target,
                    "MAE": float(part.absolute_error.mean()),
                    "municipalities": len(part),
                }
            )
        references = (
            {"seasonal_pooled", "prophet", "blend_75"}
            if model == "hgb_frozen"
            else {"seasonal_pooled"}
        )
        for reference in sorted(references - {model}):
            base = (
                frame[frame.period.eq(period) & frame.model.eq(reference)]
                .sort_values(KEYS)
                .reset_index(drop=True)
            )
            part = group.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(part[KEYS], base[KEYS])
            np.testing.assert_array_equal(part.actual, base.actual)
            part["difference"] = part.absolute_error - base.absolute_error
            paired.append({**labels, "reference": reference, **paired_metrics(part)})
            for omitted in sorted(group.target.unique()):
                sensitivity.append(
                    {
                        **labels,
                        "reference": reference,
                        "excluded_target": omitted,
                        **paired_metrics(part[part.target.ne(omitted)]),
                    }
                )
    return {
        name: pd.DataFrame(rows)
        for name, rows in [
            ("summary", summary),
            ("monthly", monthly),
            ("paired", paired),
            ("sensitivity", sensitivity),
        ]
    }


def coverage(panel, ids, frame):
    sample = panel.loc[ids]
    rows = []
    for target in EARLY + LATE:
        decision, origin, horizon = timing(target, 1)
        active = history_context(sample, origin, np.ones(12)).index
        valid = sample.loc[active, target].notna()
        scored_ids = frame[
            frame.target.eq(target) & frame.model.eq("seasonal_pooled")
        ].territory_id
        assert set(scored_ids) == set(active[valid])
        rows.append(
            {
                "period": period_for(target),
                "target": target,
                "decision_month": decision,
                "origin": origin,
                "sample_ids": len(ids),
                "history_eligible": len(active),
                "incomplete_history": len(ids) - len(active),
                "unobserved_targets_in_eligible": int((~valid).sum()),
                "scored_pairs_per_model": len(scored_ids),
            }
        )
    return pd.DataFrame(rows)


def infer():
    from sberindex.forecasting.prophet_backend import legacy_predict
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    panel, profile, ids, _ = inputs()
    config = json.loads((ROOT / "config.json").read_text())
    x, y, training_ids = training_arrays(panel)
    assert len(training_ids) == 2075 and len(y) == 18675
    months = pd.period_range("2023-01", periods=24, freq="M")
    runtime = []
    records = []
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        hgb = HistGradientBoostingRegressor(
            max_iter=config["hgb_max_iter"],
            max_leaf_nodes=config["hgb_max_leaf_nodes"],
            learning_rate=config["hgb_learning_rate"],
            min_samples_leaf=config["hgb_min_samples_leaf"],
            l2_regularization=config["hgb_l2_regularization"],
            random_state=config["random_seed"],
        ).fit(x, y)
        runtime.append(
            {
                "model": "hgb_frozen",
                "stage": "fit_2023",
                "target": "",
                "seconds": time.perf_counter() - start,
                "inference_ids": len(training_ids),
            }
        )
        late = late_predictions()
        for origin, part in late[late.model.eq("hgb_frozen")].groupby("origin"):
            index = months.get_loc(pd.Period(origin, freq="M"))
            cities = part.territory_id.to_numpy()
            forecasts = predict_asof(
                hgb, panel.loc[cities].to_numpy(float), index, 2, months, False
            )
            np.testing.assert_allclose(forecasts[:, -1], part.predicted, rtol=1e-10)
        for target in EARLY:
            start = time.perf_counter()
            decision, origin, horizon = timing(target, 1)
            index = months.get_loc(pd.Period(origin, freq="M"))
            active = history_context(panel.loc[ids], origin, np.ones(12))
            forecast_hgb = predict_asof(
                hgb, panel.loc[active.index].to_numpy(float), index, 2, months, False
            )[:, -1]
            for city, prediction_hgb in zip(active.index, forecast_hgb):
                values = panel.loc[city].to_numpy(float)
                prediction = legacy_predict(
                    values[: index + 1],
                    months[: index + 1].to_timestamp(),
                    months[index + 1 : index + 3].to_timestamp(),
                )
                prophet = max(0.0, float(prediction[-1]))
                seasonal = float(
                    values[index]
                    * profile[pd.Period(target, freq="M").month - 1]
                    / profile[pd.Period(origin, freq="M").month - 1]
                )
                actual = panel.loc[city, target]
                if not np.isfinite(actual):
                    continue
                for name, predicted in [
                    ("hgb_frozen", prediction_hgb),
                    ("prophet", prophet),
                    ("seasonal_pooled", seasonal),
                    ("blend_75", 0.75 * seasonal + 0.25 * prophet),
                ]:
                    records.append(
                        {
                            "territory_id": int(city),
                            "origin": origin,
                            "target": target,
                            "horizon": 2,
                            "model": name,
                            "actual": float(actual),
                            "predicted": float(predicted),
                        }
                    )
            runtime.append(
                {
                    "model": "combined_early",
                    "stage": "fit_prophet_and_forecast",
                    "target": target,
                    "seconds": time.perf_counter() - start,
                    "inference_ids": len(active),
                }
            )
            print(f"Early target {target}: {len(active)} past-eligible IDs", flush=True)
    frame = scored(pd.concat([pd.DataFrame(records), late], ignore_index=True), panel)
    frame.to_parquet(OUT / CACHE[0], index=False)
    pd.DataFrame(runtime).to_csv(OUT / CACHE[2], index=False)
    protocol = {
        "input_sha256": {p: sha(ROOT / p) for p in INPUTS},
        "cached_artifact_sha256": {p: sha(OUT / p) for p in [CACHE[0], CACHE[2]]},
        "sample_ids": ids,
        "training_ids": training_ids.astype(int).tolist(),
        "training_rows": len(y),
        "training_target_months": [str(m) for m in months[3:12]],
        "early_targets": EARLY,
        "late_targets": LATE,
        "models": MODELS,
        "blend_weight": 0.75,
        "blend_selection": "Transferred unchanged from early h1 comparison; no h2 selection",
        "decision_rule": "end t, latest t-1, target t+1, effective h2",
        "torch_inference": False,
        "late_hgb_reproduction": "Matched every saved late h2 HGB prediction at rtol1e-10",
        "limits": "Reused2024; early vs late dates and eligible IDs differ; hypothetical publication lag; no selection, independence or regime-change claim.",
    }
    (OUT / CACHE[1]).write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    refresh()


def refresh(root=ROOT):
    protocol = json.loads((root / "results" / CACHE[1]).read_text())
    assert set(protocol["input_sha256"]) == set(INPUTS)
    for p, h in protocol["input_sha256"].items():
        assert sha(root / p) == h, p
    assert set(protocol["cached_artifact_sha256"]) == {CACHE[0], CACHE[2]}
    for p, h in protocol["cached_artifact_sha256"].items():
        assert sha(root / "results" / p) == h, p
    frame = pd.read_parquet(root / "results" / CACHE[0])
    panel, _, ids, _ = inputs(root)
    for name, table in summaries(frame).items():
        table.to_csv(root / "results" / f"operational_early_{name}.csv", index=False)
    coverage(panel, ids, frame).to_csv(
        root / "results/operational_early_coverage.csv", index=False
    )


def verify():
    protocol = json.loads((OUT / CACHE[1]).read_text())
    assert set(protocol["input_sha256"]) == set(INPUTS)
    for p, h in protocol["input_sha256"].items():
        assert sha(ROOT / p) == h, p
    assert set(protocol["cached_artifact_sha256"]) == {CACHE[0], CACHE[2]}
    for p, h in protocol["cached_artifact_sha256"].items():
        assert sha(OUT / p) == h, p
    panel, profile, ids, raw = inputs()
    x, y, train_ids = training_arrays(panel)
    changed = panel.copy()
    changed.loc[:, changed.columns > "2023-12"] = np.nan
    changed_x, changed_y, changed_ids = training_arrays(changed)
    np.testing.assert_array_equal(x, changed_x)
    np.testing.assert_array_equal(y, changed_y)
    np.testing.assert_array_equal(train_ids, changed_ids)
    assert (
        protocol["sample_ids"] == ids and protocol["training_rows"] == len(y) == 18675
    )
    np.testing.assert_array_equal(train_ids, protocol["training_ids"])
    assert (
        protocol["blend_weight"] == 0.75
        and protocol["early_targets"] == EARLY
        and protocol["late_targets"] == LATE
    )
    frame = pd.read_parquet(OUT / CACHE[0])
    assert (
        set(frame.model) == set(MODELS) and not frame.duplicated(KEYS + ["model"]).any()
    )
    assert np.isfinite(
        frame[
            ["actual", "predicted", "absolute_error", "normalized_error_2023"]
        ].to_numpy()
    ).all()
    assert frame.predicted.ge(0).all() and frame.horizon.eq(2).all()
    after = scored(frame, panel)
    pd.testing.assert_frame_equal(frame, after)
    for target, part in frame.groupby("target"):
        decision, origin, horizon = timing(target, 1)
        assert part.decision_month.eq(decision).all() and part.origin.eq(origin).all()
        assert (
            part.period.eq(period_for(target)).all()
            and part.business_horizon_months.eq(1).all()
            and part.reporting_delay_months.eq(1).all()
        )
        base = (
            part[part.model.eq("seasonal_pooled")]
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        for model, group in part.groupby("model"):
            group = group.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(base[KEYS], group[KEYS])
            np.testing.assert_array_equal(base.actual, group.actual)
            validate_actuals(group, raw[raw.category.eq("Все категории")])
        value = (
            panel.loc[base.territory_id, origin].to_numpy(float)
            * profile[pd.Period(target, freq="M").month - 1]
            / profile[pd.Period(origin, freq="M").month - 1]
        )
        # Reused late forecasts divide before multiplying; tolerate only roundoff.
        np.testing.assert_allclose(base.predicted, value, rtol=1e-12, atol=1e-9)
        wide = part.pivot(index=KEYS, columns="model", values="predicted")
        np.testing.assert_allclose(
            wide.blend_75, 0.75 * wide.seasonal_pooled + 0.25 * wide.prophet, rtol=1e-12
        )
    late = (
        frame[frame.period.eq("late")]
        .sort_values(KEYS + ["model"])
        .reset_index(drop=True)
    )
    old = late_predictions().sort_values(KEYS + ["model"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(late[KEYS], old[KEYS], check_dtype=False)
    np.testing.assert_array_equal(late.predicted, old.predicted)
    for name, expected in summaries(frame).items():
        pd.testing.assert_frame_equal(
            pd.read_csv(OUT / f"operational_early_{name}.csv"),
            expected,
            check_dtype=False,
            rtol=1e-11,
            atol=1e-9,
        )
    pd.testing.assert_frame_equal(
        pd.read_csv(OUT / "operational_early_coverage.csv"),
        coverage(panel, ids, frame),
        check_dtype=False,
    )
    return True


if __name__ == "__main__":
    import sys

    if "--refresh" in sys.argv:
        refresh()
    elif "--verify" in sys.argv:
        print(verify())
    else:
        infer()
