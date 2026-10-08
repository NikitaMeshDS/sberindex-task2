"""Summarize matched, observed forecasts from the 2023-known cohort."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sberindex.paths import ROOT
OUT = ROOT / "results"


def main():
    raw = pd.read_parquet(OUT / "asof_cohort_predictions.parquet")
    protocol = json.loads((OUT / "asof_cohort_protocol.json").read_text())
    assert set(raw.model) == {"prophet", "seasonal_pooled"}
    assert not raw.duplicated(["territory_id", "origin", "horizon", "model"]).any()
    wide = raw.pivot(index=["territory_id", "origin", "target", "horizon"],
                     columns="model", values=["actual", "predicted"])
    assert wide.notna().all().all()
    np.testing.assert_allclose(wide[("actual", "prophet")],
                               wide[("actual", "seasonal_pooled")])
    frame = wide.reset_index()
    frame.columns = ["_".join(str(x) for x in pair if x) for pair in frame.columns]
    early = frame[(frame.horizon == 1) & (frame.origin >= "2024-01") &
                  (frame.target <= "2024-06")]
    weights = json.loads((ROOT / "config.json").read_text())["blend_candidate_seasonal_weights"]
    validation = pd.DataFrame([{
        "seasonal_weight": weight,
        "MAE": float(np.abs(early.actual_prophet -
                  (weight*early.predicted_seasonal_pooled +
                   (1-weight)*early.predicted_prophet)).mean()),
        "observations": len(early),
        "dates": early.target.nunique(),
    } for weight in weights])
    validation.to_csv(OUT / "asof_cohort_validation.csv", index=False)
    chosen = float(validation.loc[validation.MAE.idxmin(), "seasonal_weight"])

    rows = []
    matched = []
    for horizon in [1, 3, 6, 12]:
        subset = frame[frame.horizon == horizon]
        if horizon < 12:
            subset = subset[subset.origin >= "2024-06"]
        if subset.empty:
            continue
        predictions = {"prophet": subset.predicted_prophet.to_numpy(),
                       "seasonal_pooled": subset.predicted_seasonal_pooled.to_numpy()}
        if horizon < 12:
            predictions["blend_selected"] = (
                chosen*predictions["seasonal_pooled"] +
                (1-chosen)*predictions["prophet"])
        y = subset.actual_prophet.to_numpy()
        for model, forecast in predictions.items():
            ae = np.abs(y-forecast)
            rows.append({"horizon": horizon, "model": model,
                         "dates": subset.target.nunique(),
                         "municipalities": subset.territory_id.nunique(),
                         "observations": len(subset),
                         "MAE": float(ae.mean()),
                         "R2": float(r2_score(y, forecast)),
                         "WAPE_pct": float(100*ae.sum()/y.sum())})
            matched.extend({"territory_id": int(territory_id), "origin": origin,
                            "target": target, "horizon": horizon, "model": model,
                            "actual": float(actual), "predicted": float(pred),
                            "absolute_error": float(error)}
                           for territory_id,origin,target,actual,pred,error in
                           zip(subset.territory_id,subset.origin,subset.target,y,forecast,ae))
    hgb = pd.read_csv(OUT / "asof_hgb_12m.csv")
    twelve = frame[frame.horizon == 12]
    hgb = twelve[["territory_id", "origin", "target", "horizon", "actual_prophet"]].merge(
        hgb, on=["territory_id", "origin", "target", "horizon"], validate="one_to_one")
    np.testing.assert_allclose(hgb.actual_prophet, hgb.actual)
    assert len(hgb) == len(twelve)
    errors = np.abs(hgb.actual-hgb.predicted)
    rows.append({"horizon": 12, "model": "global_hgb_asof",
                 "dates": hgb.target.nunique(),
                 "municipalities": hgb.territory_id.nunique(),
                 "observations": len(hgb),
                 "MAE": float(errors.mean()),
                 "R2": float(r2_score(hgb.actual, hgb.predicted)),
                 "WAPE_pct": float(100*errors.sum()/hgb.actual.sum())})
    matched.extend({"territory_id": int(r.territory_id), "origin": r.origin,
                    "target": r.target, "horizon": 12, "model": "global_hgb_asof",
                    "actual": float(r.actual), "predicted": float(r.predicted),
                    "absolute_error": float(abs(r.actual-r.predicted))}
                   for r in hgb.itertuples())
    foundation = pd.read_parquet(OUT / "asof_foundation_predictions.parquet")
    assert set(foundation.model) == {"chronos_bolt_tiny", "chronos_2"}
    keys = ["territory_id", "origin", "target", "horizon"]
    for (horizon, model), group in foundation.groupby(["horizon", "model"]):
        if horizon < 12:
            group = group[group.origin >= "2024-06"]
        q = frame[frame.horizon == horizon]
        if horizon < 12:
            q = q[q.origin >= "2024-06"]
        joined = q[keys + ["actual_prophet"]].merge(group, on=keys,
                 validate="one_to_one")
        assert len(joined) == len(q) == len(group), (horizon, model)
        np.testing.assert_allclose(joined.actual_prophet, joined.actual)
        errors = np.abs(joined.actual-joined.predicted)
        rows.append({"horizon": int(horizon), "model": model,
                     "dates": joined.target.nunique(),
                     "municipalities": joined.territory_id.nunique(),
                     "observations": len(joined), "MAE": float(errors.mean()),
                     "R2": float(r2_score(joined.actual, joined.predicted)),
                     "WAPE_pct": float(100*errors.sum()/joined.actual.sum())})
        matched.extend({"territory_id": int(r.territory_id), "origin": r.origin,
                        "target": r.target, "horizon": int(horizon), "model": model,
                        "actual": float(r.actual), "predicted": float(r.predicted),
                        "absolute_error": float(abs(r.actual-r.predicted))}
                       for r in joined.itertuples())
    summary = pd.DataFrame(rows).sort_values(["horizon", "MAE"])
    summary.to_csv(OUT / "asof_cohort_comparison.csv", index=False)
    pd.DataFrame(matched).to_parquet(OUT / "asof_cohort_scored.parquet", index=False)

    selected_ids = protocol["sample_ids"]
    consumption = pd.read_parquet(ROOT / "data/consumption.parquet")
    panel = (consumption.loc[consumption.category == "Все категории"]
             .pivot(index="territory_id", columns="date", values="value"))
    selected = panel.loc[selected_ids]
    report = {
        "selected_ids": len(selected_ids),
        "selected_with_complete_2024": int(selected.iloc[:, 12:].notna().all(axis=1).sum()),
        "selected_with_no_2024": int(selected.iloc[:, 12:].notna().any(axis=1).eq(False).sum()),
        "selected_weight_from_early_2024": chosen,
        "limits": "Observed targets only, fixed 2023-known sample. The 2024 archive, model family, and candidate weights were previously studied; this is not independent validation.",
    }
    (OUT / "asof_cohort_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
