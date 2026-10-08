"""Frozen, matched calendar ablations; see CALENDAR_EXPERIMENT.md."""
import calendar
from datetime import date
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from sberindex.paths import ROOT

OUT = ROOT / "results"
KEYS = ["territory_id", "origin", "target", "horizon"]


def calendar_features(month):
    """Days then Monday through Sunday counts; no spending or holiday data."""
    year, number = map(int, str(month)[:7].split("-"))
    days = calendar.monthrange(year, number)[1]
    counts = [0] * 7
    for day in range(1, days + 1):
        counts[date(year, number, day).weekday()] += 1
    return np.asarray([days, *counts], dtype=float)


def seasonal_day_forecast(anchor, origin, target, profile):
    origin, target = pd.Period(origin, freq="M"), pd.Period(target, freq="M")
    days_2023 = np.asarray([calendar.monthrange(2023, m)[1] for m in range(1, 13)])
    daily = np.asarray(profile) / days_2023
    return (np.asarray(anchor) / calendar_features(origin)[0] *
            daily[target.month - 1] / daily[origin.month - 1] * calendar_features(target)[0])


def features(log_history, target, with_calendar):
    month = pd.Period(target, freq="M").month
    n = len(log_history)
    matrix = np.column_stack([
        log_history[:, -1], log_history[:, -1] - log_history[:, -2],
        log_history[:, -2] - log_history[:, -3],
        np.full(n, np.sin(2 * np.pi * month / 12)),
        np.full(n, np.cos(2 * np.pi * month / 12))])
    if with_calendar:
        matrix = np.column_stack([matrix, np.tile(calendar_features(target), (n, 1))])
    return matrix


def predict_asof(model, spending, origin_index, horizon, months, with_calendar):
    """Accept the panel for testing, but access only three observed lag months."""
    trajectory = np.log(spending[:, origin_index - 2:origin_index + 1]).copy()
    assert np.isfinite(trajectory).all()
    predicted = []
    for target in range(origin_index + 1, origin_index + horizon + 1):
        growth = np.clip(model.predict(features(trajectory, months[target], with_calendar)), -.5, .5)
        trajectory = np.column_stack([trajectory, trajectory[:, -1] + growth])
        predicted.append(np.exp(trajectory[:, -1]))
    return np.column_stack(predicted)


def analytic_tests():
    np.testing.assert_array_equal(calendar_features("2024-02"), [29, 4, 4, 4, 5, 4, 4, 4])
    assert calendar_features("2023-02")[0] == 28
    daily_constant_profile = [calendar.monthrange(2023, m)[1] for m in range(1, 13)]
    np.testing.assert_allclose(seasonal_day_forecast(31., "2024-01", "2024-02", daily_constant_profile), 29.)

    class KnownPredictor:
        def predict(self, x):
            return np.log(x[:, 5] / 31.) if x.shape[1] > 5 else np.full(len(x), .01)

    months = pd.period_range("2023-01", periods=24, freq="M")
    spending = np.full((2, 24), 31.)
    changed = spending.copy()
    changed[:, 13:] = np.nan
    for with_calendar in [False, True]:
        a = predict_asof(KnownPredictor(), spending, 12, 6, months, with_calendar)
        b = predict_asof(KnownPredictor(), changed, 12, 6, months, with_calendar)
        np.testing.assert_allclose(a, b)
    expected = 31. * np.cumprod([calendar_features(m)[0] / 31. for m in months[13:19]])
    np.testing.assert_allclose(a[0], expected)
    return True


def metrics(frame):
    ae = np.abs(frame.actual - frame.predicted)
    return {"observations": len(frame), "municipalities": frame.territory_id.nunique(),
            "dates": frame.target.nunique(), "MAE": float(ae.mean()),
            "R2": float(r2_score(frame.actual, frame.predicted)),
            "WAPE_pct": float(100 * ae.sum() / frame.actual.sum())}


def summaries(predictions):
    frame = predictions.copy()
    frame["period"] = np.where(frame.horizon == 12, "12m_single_date",
                                np.where(frame.origin < "2024-06", "early", "late"))
    summary = pd.DataFrame([{**dict(zip(["period", "horizon", "model"], key)), **metrics(g)}
                            for key, g in frame.groupby(["period", "horizon", "model"])])
    monthly = pd.DataFrame([{**dict(zip(["period", "origin", "target", "horizon", "model"], key)), **metrics(g)}
                            for key, g in frame.groupby(["period", "origin", "target", "horizon", "model"])])
    deltas = []
    for before, after in [("seasonal_pooled", "seasonal_day_corrected"),
                          ("hgb_frozen", "hgb_frozen_calendar")]:
        for (period, horizon), group in summary.groupby(["period", "horizon"]):
            s = group.set_index("model")
            m = monthly[(monthly.period == period) & (monthly.horizon == horizon)]
            m = m.pivot(index=["origin", "target"], columns="model", values="MAE")
            change = m[after] - m[before]
            # Algebraically identical non-leap forecasts can differ by roundoff.
            change = change.mask(np.isclose(change, 0., atol=1e-9, rtol=0.), 0.)
            deltas.append({"period": period, "horizon": horizon, "baseline": before,
                           "candidate": after, "MAE_delta": s.loc[after, "MAE"] - s.loc[before, "MAE"],
                           "MAE_delta_pct": 100 * (s.loc[after, "MAE"] / s.loc[before, "MAE"] - 1),
                           "WAPE_pp_delta": s.loc[after, "WAPE_pct"] - s.loc[before, "WAPE_pct"],
                           "dates_improved": int((change < 0).sum()),
                           "dates_worsened": int((change > 0).sum()), "dates_tied": int((change == 0).sum())})
    return summary, monthly, pd.DataFrame(deltas)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analytic_tests()  # Run before any experimental training or score calculation.
    config = json.loads((ROOT / "config.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = raw.loc[raw.category == "Все категории"].pivot(index="territory_id", columns="date", values="value").sort_index()
    eligible = panel.iloc[:, :12].notna().all(axis=1)
    ids = panel.index[eligible]
    assert len(ids) == 2075
    train = panel.loc[ids].iloc[:, :12].to_numpy(float)
    assert (train > 0).all()
    profile = train.sum(axis=0)
    months = pd.period_range("2023-01", periods=24, freq="M")
    original = pd.read_parquet(OUT / "asof_cohort_predictions.parquet")
    pairs = original.loc[original.model == "prophet", KEYS + ["actual"]].copy()
    records = [original]
    day = pairs.copy()
    day["model"] = "seasonal_day_corrected"
    day["predicted"] = [float(seasonal_day_forecast(panel.loc[r.territory_id, r.origin], r.origin, r.target, profile))
                        for r in pairs.itertuples()]
    records.append(day)
    trained = {}
    with threadpool_limits(limits=2):
        for with_calendar in [False, True]:
            name = "hgb_frozen_calendar" if with_calendar else "hgb_frozen"
            x = np.concatenate([features(np.log(train[:, :target]), months[target], with_calendar) for target in range(3, 12)])
            y = np.concatenate([np.log(train[:, target]) - np.log(train[:, target - 1]) for target in range(3, 12)])
            model = HistGradientBoostingRegressor(
                max_iter=config["hgb_max_iter"], max_leaf_nodes=config["hgb_max_leaf_nodes"],
                learning_rate=config["hgb_learning_rate"], min_samples_leaf=config["hgb_min_samples_leaf"],
                l2_regularization=config["hgb_l2_regularization"], random_state=config["random_seed"])
            model.fit(x, y)
            trained[name] = {"training_rows": len(y), "features": x.shape[1], "iterations": model.n_iter_}
            for origin, group in pairs.groupby("origin"):
                origin_index = months.get_loc(pd.Period(origin, freq="M"))
                territory_ids = group.territory_id.unique()
                spending = panel.loc[territory_ids].to_numpy(float)
                forecasts = predict_asof(model, spending, origin_index, int(group.horizon.max()), months, with_calendar)
                lookup = {int(v): i for i, v in enumerate(territory_ids)}
                result = group.copy()
                result["model"] = name
                result["predicted"] = [forecasts[lookup[int(r.territory_id)], int(r.horizon) - 1] for r in group.itertuples()]
                records.append(result)
            print(f"Calendar audit trained {name}: {trained[name]}", flush=True)
    predictions = pd.concat(records, ignore_index=True)
    predictions.to_parquet(OUT / "calendar_predictions.parquet", index=False)
    summary, monthly, deltas = summaries(predictions)
    summary.to_csv(OUT / "calendar_summary.csv", index=False)
    monthly.to_csv(OUT / "calendar_monthly.csv", index=False)
    deltas.to_csv(OUT / "calendar_deltas.csv", index=False)
    report = {"training_ids": len(ids), "models": trained, "scored_pairs_per_model": len(pairs),
              "config": config, "future_spending_in_features": False, "holiday_labels": False,
              "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                               [ROOT / "data/consumption.parquet", ROOT / "config.json", OUT / "asof_cohort_predictions.parquet",
                                ROOT / "docs/protocols/CALENDAR_EXPERIMENT.md"]},
              "limits": "Retrospective explored-2024 audit; calendar and seasonality confounded; 6m/12m each only one target date; no late selection."}
    (OUT / "calendar_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    verify()
    print(deltas.to_string(index=False))


def verify():
    """Cheap analytic, pair, metric and provenance checks for project verification."""
    analytic_tests()
    predictions = pd.read_parquet(OUT / "calendar_predictions.parquet")
    original = pd.read_parquet(OUT / "asof_cohort_predictions.parquet")
    assert set(predictions.model) == {"prophet", "seasonal_pooled", "seasonal_day_corrected", "hgb_frozen", "hgb_frozen_calendar"}
    assert not predictions.duplicated(KEYS + ["model"]).any()
    assert np.isfinite(predictions[["actual", "predicted"]].to_numpy()).all()
    reference = original[original.model == "prophet"].set_index(KEYS).sort_index()
    for name, group in predictions.groupby("model"):
        group = group.set_index(KEYS).sort_index()
        assert group.index.equals(reference.index), name
        np.testing.assert_allclose(group.actual, reference.actual)
        if name in {"prophet", "seasonal_pooled"}:
            before = original[original.model == name].set_index(KEYS).sort_index()
            np.testing.assert_array_equal(group.predicted, before.predicted)
    summary, monthly, deltas = summaries(predictions)
    for expected, filename in [(summary, "calendar_summary.csv"), (monthly, "calendar_monthly.csv"), (deltas, "calendar_deltas.csv")]:
        pd.testing.assert_frame_equal(expected, pd.read_csv(OUT / filename), check_dtype=False, rtol=1e-10)
    hgb = pd.read_csv(OUT / "asof_hgb_12m.csv").set_index(KEYS).sort_index()
    frozen = predictions[(predictions.model == "hgb_frozen") & (predictions.horizon == 12)].set_index(KEYS).sort_index()
    assert frozen.index.equals(hgb.index)
    np.testing.assert_allclose(frozen.predicted, hgb.predicted, rtol=1e-10)
    report = json.loads((OUT / "calendar_audit.json").read_text())
    for path, checksum in report["input_sha256"].items():
        assert digest(ROOT / path) == checksum, path
    assert report["training_ids"] == 2075
    return True


if __name__ == "__main__":
    main()
