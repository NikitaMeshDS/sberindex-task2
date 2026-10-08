"""Independently check saved expansion comparisons, keys and temporal cutoffs."""

import hashlib
import json

import numpy as np
import pandas as pd

from sberindex.paths import ROOT

KEYS = ["territory_id", "origin", "target", "horizon"]


def check_comparison(predictions, summary, extra_keys=()):
    group = [*extra_keys, "model", "horizon"]
    if predictions.duplicated([*extra_keys, "model", *KEYS]).any():
        raise ValueError("duplicate model evaluation key")
    if not np.isfinite(predictions[["actual", "predicted"]]).all().all():
        raise ValueError("nonfinite forecast or observation")
    identity = predictions.groupby([*extra_keys, *KEYS]).actual.agg(["min", "max"])
    if not np.array_equal(identity["min"], identity["max"]):
        raise ValueError("models compared with different actuals")
    computed = (
        predictions.assign(loss=abs(predictions.actual - predictions.predicted))
        .groupby([*group, "target"])
        .loss.mean()
        .groupby(group)
        .mean()
    )
    claimed = summary.set_index(group).MAE
    joined = computed.to_frame("computed").join(claimed, how="outer")
    if joined.isna().any().any():
        raise ValueError("missing computed or claimed model result")
    np.testing.assert_allclose(joined.computed, joined.MAE, rtol=1e-12, atol=1e-8)
    return {
        "rows": len(predictions),
        "comparisons": len(joined),
        "maximum_MAE_difference": float(abs(joined.computed - joined.MAE).max()),
    }


def main():
    block = ROOT / "reports/foundation_expansion_review"
    common = pd.read_parquet(block / "matched_predictions.parquet")
    result = {
        "foundation_common": check_comparison(
            common, pd.read_csv(block / "summary.csv")
        )
    }
    reference = pd.read_csv(
        ROOT / "reports/presentation_comparison_cases/common_pairs.csv"
    )[KEYS]
    for _, frame in common.groupby("model"):
        paired = frame[KEYS].merge(
            reference, how="outer", indicator=True, validate="one_to_one"
        )
        if not paired._merge.eq("both").all():
            raise ValueError("foundation model has different evaluation keys")
    result["foundation_fullpanel"] = check_comparison(
        pd.read_parquet(block / "fullpanel_matched_predictions.parquet"),
        pd.read_csv(block / "fullpanel_summary.csv"),
    )
    training = json.loads((block / "bolt_epoch1_training.json").read_text())
    if (
        training["train_end"] > "2023-12"
        or training["validation_end"] > "2023-12"
        or training["sampled_training_municipalities"] != 2075
    ):
        raise ValueError("fine-tuning cutoff or full-panel training scope violated")
    result["training_cutoff"] = training["train_end"]
    categories = ROOT / "reports/category_growth_review"
    if (categories / "full_pooled_predictions.parquet").exists():
        predictions = pd.read_parquet(categories / "full_pooled_predictions.parquet")
        result["categories"] = check_comparison(
            predictions,
            pd.read_csv(categories / "full_pooled_summary.csv"),
            ["category"],
        )
    else:
        result["categories"] = "pending_full_comparator"
    result.update(
        status="passed",
        independent_temporal_validation=False,
        scope="Independent arithmetic and evaluation-key checks; not repeated model fitting",
    )
    sources = [
        "tools/verify_expansion_evidence.py",
        "reports/foundation_expansion_review/summary.csv",
        "reports/foundation_expansion_review/fullpanel_summary.csv",
        "reports/foundation_expansion_review/bolt_epoch1_training.json",
    ]
    result["input_sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
    }
    (ROOT / "reports/expansion_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
