"""Frozen operational candidate, delayed residual audit and detector decision.

Build: python tools/operational_workflow.py
Issue: python tools/operational_workflow.py --decision 2024-12 --lag 1 --output forecast.csv
Historical publication vintages remain unknown; lags are scenarios.
"""

from pathlib import Path
import argparse
import hashlib
import json
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import residual_feature_experiment as residual

OUT = ROOT / "reports/operational_workflow"
KEYS = [
    "territory_id",
    "decision_month",
    "origin",
    "target",
    "business_horizon",
    "reporting_lag",
]
INPUTS = [
    "data/consumption.parquet",
    "data/external/cbr_rate_decisions_2023_2024.csv",
    "results/asof_cohort_protocol.json",
    "results/bocpd_runs.csv",
    "results/bocpd_protocol.json",
    "configs/operational_workflow.json",
    "configs/residual_features.json",
    "configs/detectors.json",
    "docs/protocols/OPERATIONAL_WORKFLOW.md",
    "tools/operational_workflow.py",
    "src/sberindex/forecasting/prophet_backend.py",
    "tools/residual_feature_experiment.py",
]


def timing(decision, horizon, lag):
    if horizon < 1 or lag < 0:
        raise ValueError("Positive horizon and nonnegative lag required")
    d = pd.Period(decision, freq="M")
    return str(d - lag), str(d + horizon), horizon + lag


def training_available(origin, training_end):
    return pd.Period(origin, freq="M") >= pd.Period(training_end, freq="M")


def prophet_forecast(history, horizon):
    from sberindex.forecasting.prophet_backend import legacy_predict

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    dates = pd.date_range("2023-01-01", periods=len(history) + horizon, freq="MS")
    return np.maximum(
        0.0, legacy_predict(history, dates[: len(history)], dates[len(history) :])
    )


def forecast_row(
    values, profile, decision, lag, horizon, weight, predictor=prophet_forecast
):
    origin, target, effective = timing(decision, horizon, lag)
    row = dict(
        decision_month=decision,
        origin=origin,
        target=target,
        business_horizon=horizon,
        reporting_lag=lag,
        effective_horizon=effective,
        model="blend_75",
        predicted=np.nan,
    )
    if not 0 <= weight <= 1:
        raise ValueError("Invalid mixture weight")
    if not training_available(origin, "2023-12"):
        return {**row, "status": "training_not_available"}
    i = (pd.Period(origin, freq="M") - pd.Period("2023-01", freq="M")).n
    history = values[: i + 1]
    if (
        i >= len(values)
        or len(history) != i + 1
        or not (np.isfinite(history).all() and (history > 0).all())
    ):
        return {**row, "status": "missing_or_nonpositive_history"}
    base = history[-1] * profile[(i + effective) % 12] / profile[i % 12]
    predicted = weight * base + (1 - weight) * float(
        predictor(history, effective)[effective - 1]
    )
    return {**row, "predicted": predicted, "status": "ok"}


def validate_seeds(selection, evaluation):
    if (
        not selection
        or not evaluation
        or set(selection) & set(evaluation)
        or len(set(selection)) != len(selection)
        or len(set(evaluation)) != len(evaluation)
    ):
        raise ValueError("Nonempty unique disjoint seed lists required")


def choose_detector(table, budget, minimum):
    if budget < 0 or not 0 <= minimum <= 1:
        raise ValueError("Invalid operational requirements")
    eligible = table[
        (table.maximum_burden <= budget) & (table.minimum_detection >= minimum)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["minimum_detection", "maximum_burden", "method"], ascending=[False, True, True]
    ).iloc[0]["method"]


def detector_decision(config):
    c = config["detector"]
    validate_seeds(c["selection_seeds"], c["evaluation_seeds"])
    f = pd.read_csv(ROOT / "results/bocpd_runs.csv")
    f = f[
        f.category.eq(c["category"])
        & f.scaling.eq(c["scaling"])
        & f.reporting_delay_months.eq(c["reporting_delay_months"])
        & f["shape"].eq(c["shape"])
    ]
    hit = {
        1: "new_by_first_calendar_month",
        2: "new_by_second_calendar_month",
        3: "new_by_third_calendar_month",
    }[c["deadline_months"]]
    tables = []
    for part, seeds in [
        ("selection", c["selection_seeds"]),
        ("evaluation", c["evaluation_seeds"]),
    ]:
        g = f[f.seed.isin(seeds)].copy()
        assert len(g) == len(seeds) * 2 * 5 and set(g.shift_pct) == {-20, 20}
        by = (
            g.groupby(["method", "shift_pct"])
            .agg(
                detection=(hit, "mean"),
                burden=("original_control_per100_delivered_months", "mean"),
            )
            .reset_index()
        )
        t = (
            by.groupby("method")
            .agg(
                minimum_detection=("detection", "min"), maximum_burden=("burden", "max")
            )
            .reset_index()
        )
        t["partition"] = part
        tables.append(t)
        by["partition"] = part
        by.to_csv(OUT / f"detector_{part}_signs.csv", index=False)
    selection = tables[0]
    chosen = choose_detector(selection, c["max_burden_per100"], c["minimum_detection"])
    result = {
        "selected_method": chosen,
        "status": "selected" if chosen else "no_admissible_method",
        "requirements": c,
        "evaluation_used_for_selection": False,
        "limits": "Exploratory split of synthetic assignments on already studied2024 months and same municipalities. Not independent time/event validation. Budget/deadline engineering assumptions; original alarms are not proven false positives.",
    }
    if chosen:
        metrics = tables[1].set_index("method").loc[chosen]
        result["separate_assignment_evaluation"] = {
            k: float(metrics[k]) for k in ["minimum_detection", "maximum_burden"]
        }
        result["evaluation_meets_requirements"] = bool(
            metrics.minimum_detection >= c["minimum_detection"]
            and metrics.maximum_burden <= c["max_burden_per100"]
        )
    pd.concat(tables, ignore_index=True).to_csv(
        OUT / "detector_partitions.csv", index=False
    )
    (OUT / "detector_choice.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def summaries(frame):
    f = frame.copy()
    f["error"] = (f.actual - f.predicted).abs()
    f["period"] = np.where(f.target < "2024-07", "early", "late")
    rows = []
    monthly = []
    paired = []
    sensitivity = []
    for (lag, h, period, model), g in f.groupby(
        ["reporting_lag", "business_horizon", "period", "model"]
    ):
        labels = dict(
            reporting_lag=int(lag), business_horizon=int(h), period=period, model=model
        )
        rows.append(
            {
                **labels,
                "MAE_date_balanced": g.groupby("target").error.mean().mean(),
                "R2_pooled": r2_score(g.actual, g.predicted),
                "dates": g.target.nunique(),
                "observations": len(g),
            }
        )
        for target, part in g.groupby("target"):
            monthly.append(
                {
                    **labels,
                    "target": target,
                    "MAE": part.error.mean(),
                    "observations": len(part),
                }
            )
        if model.startswith("residual_"):
            for ref in ["seasonal_pooled", "prophet", "blend_75"]:
                base = f[
                    (f.reporting_lag == lag)
                    & (f.business_horizon == h)
                    & (f.period == period)
                    & (f.model == ref)
                ]
                p = g.merge(
                    base[KEYS + ["actual", "error"]],
                    on=KEYS,
                    suffixes=("", "_ref"),
                    validate="one_to_one",
                )
                assert len(p) == len(g)
                np.testing.assert_array_equal(p.actual, p.actual_ref)
                p["difference"] = p.error - p.error_ref
                l = {**labels, "reference": ref}
                paired.append(
                    {
                        **l,
                        "MAE_difference": p.groupby("target").difference.mean().mean(),
                        "date_win_fraction": p.groupby("target")
                        .difference.mean()
                        .lt(0)
                        .mean(),
                    }
                )
                if p.target.nunique() > 1:
                    for omitted in sorted(p.target.unique()):
                        sensitivity.append(
                            {
                                **l,
                                "excluded_target": omitted,
                                "MAE_difference": p[p.target.ne(omitted)]
                                .groupby("target")
                                .difference.mean()
                                .mean(),
                            }
                        )
    return {
        k: pd.DataFrame(v)
        for k, v in [
            ("summary", rows),
            ("monthly", monthly),
            ("paired", paired),
            ("sensitivity", sensitivity),
        ]
    }


def audit_forecasts(config):
    values, ids, profile, events = residual.load()
    sample = json.loads((ROOT / "results/asof_cohort_protocol.json").read_text())[
        "sample_ids"
    ]
    positions = [list(ids).index(city) for city in sample]
    v = values[positions]
    c = json.loads((ROOT / "configs/residual_features.json").read_text())
    estimators = {}
    for variant in ["own", "categories", "policy"]:
        x, y = residual.training_arrays(values, profile, events, variant)
        model = HistGradientBoostingRegressor(
            max_iter=c["hgb_max_iter"],
            max_leaf_nodes=c["hgb_max_leaf_nodes"],
            min_samples_leaf=c["hgb_min_samples_leaf"],
            l2_regularization=c["hgb_l2_regularization"],
            learning_rate=c["hgb_learning_rate"],
            random_state=c["seed"],
        )
        model.fit(x, y)
        estimators[variant] = model
    records = []
    coverage = []
    prophet_cache = {}
    decisions = [str(x) for x in pd.period_range("2023-12", "2024-11", freq="M")]
    for lag in config["reporting_lags"]:
        for h in config["horizons"]:
            for decision in decisions:
                origin, target, effective = timing(decision, h, lag)
                i = (pd.Period(origin, freq="M") - pd.Period("2023-01", freq="M")).n
                t = i + effective
                if not training_available(origin, config["training_end"]):
                    status = "training_not_available"
                elif t >= 24:
                    status = "target_outside_archive"
                else:
                    status = "eligible"
                valid = np.zeros(len(sample), bool)
                target_ok = np.zeros(len(sample), bool)
                if status == "eligible":
                    history = v[:, 0, : i + 1]
                    valid = np.isfinite(history).all(axis=1) & (history > 0).all(axis=1)
                    target_ok = np.isfinite(v[:, 0, t]) & (v[:, 0, t] > 0)
                paired = valid & target_ok
                coverage.append(
                    dict(
                        decision_month=decision,
                        origin=origin,
                        target=target,
                        business_horizon=h,
                        reporting_lag=lag,
                        effective_horizon=effective,
                        status=status,
                        cohort_n=len(sample),
                        valid_history_n=int(valid.sum()),
                        observed_target_n=int(target_ok.sum()),
                        scored_n=int(paired.sum()),
                    )
                )
                if not paired.any():
                    continue
                vv = v[paired]
                city_ids = np.asarray(sample)[paired]
                # A Prophet fit per ID/origin reused for all horizons/lag views. It never sees the target.
                p = []
                for city, city_values in zip(city_ids, vv[:, 0]):
                    key = (int(city), origin)
                    if key not in prophet_cache:
                        prophet_cache[key] = prophet_forecast(city_values[: i + 1], 14)
                    p.append(prophet_cache[key][effective - 1])
                seasonal = residual.restore(
                    vv[:, 0, i], i, effective, profile, np.zeros(len(vv)), 0.5
                )
                outputs = {
                    "prophet": np.asarray(p),
                    "seasonal_pooled": seasonal,
                    "blend_75": config["seasonal_weight"] * seasonal
                    + (1 - config["seasonal_weight"]) * np.asarray(p),
                }
                for variant, model in estimators.items():
                    x = residual.feature_matrix(
                        vv, i, effective, profile, events, variant
                    )
                    outputs["residual_hgb_" + variant] = residual.restore(
                        vv[:, 0, i],
                        i,
                        effective,
                        profile,
                        model.predict(x),
                        c["max_absolute_log_correction"],
                    )
                for model, predicted in outputs.items():
                    records.extend(
                        dict(
                            territory_id=int(city),
                            decision_month=decision,
                            origin=origin,
                            target=target,
                            business_horizon=h,
                            reporting_lag=lag,
                            effective_horizon=effective,
                            actual=float(actual),
                            predicted=float(pred),
                            model=model,
                        )
                        for city, actual, pred in zip(city_ids, vv[:, 0, t], predicted)
                    )
            print("Delayed comparison", lag, h, "complete", flush=True)
    frame = pd.DataFrame(records).sort_values(KEYS + ["model"]).reset_index(drop=True)
    frame.to_parquet(OUT / "delayed_predictions.parquet", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    for name, table in summaries(frame).items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    return values, ids, profile


def release_panel(frame, ids):
    if frame.duplicated(["territory_id", "date"]).any():
        raise ValueError("Duplicate ID/month")
    months = pd.PeriodIndex(frame.date, freq="M")
    if (months.astype(str) != frame.date).any() or (
        months < pd.Period("2023-01", freq="M")
    ).any():
        raise ValueError("Expected YYYY-MM months from2023")
    return frame.pivot(index="territory_id", columns="date", values="value").reindex(
        index=ids,
        columns=pd.period_range("2023-01", months.max(), freq="M").astype(str),
    )


def issue(config, decision, lag, path, input_path=None):
    values, ids, profile, _ = residual.load()
    sample = json.loads((ROOT / "results/asof_cohort_protocol.json").read_text())[
        "sample_ids"
    ]
    panel = None
    if input_path:
        raw = pd.read_parquet(input_path)
        if "category" in raw:
            raw = raw[raw.category.eq("Все категории")]
        panel = release_panel(raw, sample)
        frozen = values[[list(ids).index(city) for city in sample], 0, :12]
        supplied = panel.reindex(
            columns=pd.period_range("2023-01", "2023-12", freq="M").astype(str)
        ).to_numpy(float)
        if not np.array_equal(frozen, supplied):
            raise ValueError(
                "Training2023 was revised or missing: register a new version explicitly"
            )
    rows = []
    for city in sample:
        v = (
            panel.loc[city].to_numpy(float)
            if panel is not None
            else values[list(ids).index(city), 0]
        )
        cache = {}

        def predict(history, h):
            if "p" not in cache:
                cache["p"] = prophet_forecast(history, max(config["horizons"]) + lag)
            return cache["p"]

        for horizon in config["horizons"]:
            rows.append(
                {
                    "territory_id": int(city),
                    **forecast_row(
                        v,
                        profile,
                        decision,
                        lag,
                        horizon,
                        config["seasonal_weight"],
                        predict,
                    ),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if input_path:
        provenance = {
            "input_path": str(input_path.absolute()),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "decision": decision,
            "lag": lag,
            "config": config,
            "limits": "Caller must supply same target definition, units, territory IDs and truthful availability metadata. This CLI imposes the declared uniform lag; it cannot establish historical publication dates.",
        }
        path.with_suffix(".provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
        )


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in INPUTS}


def verify():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["input_sha256"] == hashes()
    for name, digest in protocol["artifact_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
    f = pd.read_parquet(OUT / "delayed_predictions.parquet")
    assert not f.duplicated(KEYS + ["model"]).any()
    assert np.isfinite(f[["actual", "predicted"]]).all().all()
    models = {
        "prophet",
        "seasonal_pooled",
        "blend_75",
        "residual_hgb_own",
        "residual_hgb_categories",
        "residual_hgb_policy",
    }
    assert set(f.model) == models
    assert f.groupby(KEYS).model.nunique().eq(6).all()
    assert (pd.PeriodIndex(f.origin, freq="M") + f.effective_horizon.to_numpy()).astype(
        str
    ).tolist() == f.target.tolist()
    assert (f.effective_horizon == f.business_horizon + f.reporting_lag).all()
    assert (f.origin >= "2023-12").all()
    assert f[(f.business_horizon == 12) & (f.reporting_lag > 0)].empty
    for name, table in summaries(f).items():
        pd.testing.assert_frame_equal(
            table,
            pd.read_csv(OUT / f"{name}.csv"),
            check_dtype=False,
            atol=1e-9,
            rtol=1e-10,
        )
    cover = pd.read_csv(OUT / "coverage.csv")
    counts = (
        f[f.model.eq("prophet")]
        .groupby(["decision_month", "business_horizon", "reporting_lag"])
        .size()
    )
    for row in cover.itertuples():
        assert row.scored_n == counts.get(
            (row.decision_month, row.business_horizon, row.reporting_lag), 0
        )
    config = json.loads((ROOT / "configs/operational_workflow.json").read_text())
    choice = json.loads((OUT / "detector_choice.json").read_text())
    parts = pd.read_csv(OUT / "detector_partitions.csv")
    assert (
        choose_detector(
            parts[parts.partition.eq("selection")],
            config["detector"]["max_burden_per100"],
            config["detector"]["minimum_detection"],
        )
        == choice["selected_method"]
    )
    print(
        "Operational workflow verified: hashes, common pairs, timing, coverage and selection",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--decision")
    parser.add_argument("--lag", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="Complete same-definition expenditure history including unchanged2023",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/operational_workflow.json").read_text())
    if args.verify:
        verify()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        if args.decision:
            if args.output is None:
                parser.error("--output required with --decision")
            issue(
                config,
                args.decision,
                config["default_lag"] if args.lag is None else args.lag,
                args.output,
                args.input,
            )
            return
        source = hashes()
        audit_forecasts(config)
        choice = detector_decision(config)
        issue(
            config,
            config["demonstration_decision"],
            config["default_lag"],
            OUT / "forecast_release.csv",
        )
    assert source == hashes()
    protocol = {
        "input_sha256": source,
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.iterdir())
            if p.suffix in [".csv", ".parquet"] or p.name == "detector_choice.json"
        },
        "config": config,
        "limits": "Retrospective studied2024. Hypothetical publication lags; no independent test, future shock prediction or HGB promotion. Fixed2023 fit; effective horizons>9 extrapolate residual training range; policy available conservatively through last expenditure origin. Main mixture transferred .75 weight including unvalidated h12. Detector assignment split does not separate dates or municipalities.",
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    print("Detector choice:", choice, flush=True)
    verify()


if __name__ == "__main__":
    main()
