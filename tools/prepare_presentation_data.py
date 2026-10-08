"""Prepare presentation evidence from the frozen research tables, no new selection."""

import json
import math
from pathlib import Path
import pandas as pd
from sberindex.paths import ROOT


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    summary = pd.read_csv(ROOT / "reports/growth_bridge_review/summary.csv")
    quality = []
    for h, g in summary.groupby("horizon"):
        selected = g[g.model == "hierarchy05_growth_bridge"].iloc[0]
        default = g[g.model == "prophet_disabled"].iloc[0]
        best = g[g.model.str.startswith("prophet")].MAE.min()
        quality.append(
            dict(
                h=int(h),
                MAE=float(selected.MAE),
                default=float(default.MAE),
                best=float(best),
                gain_pct=float(100 * (1 - selected.MAE / best)),
                dates=int(selected.dates),
                pairs=int(selected.observations),
            )
        )
    sensitivity = pd.read_csv(ROOT / "reports/growth_sensitivity_review/summary.csv")
    detectors = pd.read_csv(ROOT / "reports/short_history_review/comparison.csv")
    detectors = detectors[
        (detectors.split == "evaluation") & (detectors["shape"] == "all")
    ]
    foundation = pd.read_csv(ROOT / "reports/foundation_covariate_review/summary.csv")
    foundation = foundation[
        (foundation.category == "Все категории") & (foundation.horizon == 1)
    ]
    payload = dict(
        jury_guide=json.loads((ROOT / "artifacts/jury_guide.json").read_text()),
        adaptive_growth=json.loads(
            (
                ROOT / "reports/adaptive_growth_holdout_20261007/freeze_manifest.json"
            ).read_text()
        ),
        recovery_news=json.loads(
            (ROOT / "reports/news_body_recovery_20261007/audit.json").read_text()
        ),
        recovery_news_burden=pd.read_csv(
            ROOT / "reports/news_body_recovery_20261007/monitoring_burden.csv"
        ).to_dict("records"),
        comparison_cases=json.loads(
            (ROOT / "reports/presentation_comparison_cases/evidence.json").read_text()
        ),
        verified_claims=json.loads(
            (ROOT / "reports/presentation_claim_checks/evidence.json").read_text()
        ),
        expansion=dict(
            fullpanel=pd.read_csv(
                ROOT / "reports/foundation_expansion_review/fullpanel_summary.csv"
            ).to_dict("records"),
            foundation=pd.read_csv(
                ROOT / "reports/foundation_expansion_review/summary.csv"
            ).to_dict("records"),
            categories=pd.read_csv(
                ROOT / "reports/category_growth_review/full_pooled_summary.csv"
            ).to_dict("records"),
            uncertainty=pd.read_csv(
                ROOT / "reports/forecast_uncertainty_review/cluster_intervals.csv"
            ).to_dict("records"),
            yoy=pd.read_csv(
                ROOT / "reports/yoy_shift_review/monitoring_burden.csv"
            ).to_dict("records"),
            news=json.loads(
                (
                    ROOT
                    / "reports/news_event_extraction_review/expanded_batch/audit.json"
                ).read_text()
            ),
            news_burden=pd.read_csv(
                ROOT
                / "reports/news_event_extraction_review/expanded_batch/monitoring_burden.csv"
            ).to_dict("records"),
        ),
        news_forecast_diagnostic=pd.read_csv(
            ROOT
            / "reports/news_event_extraction_review/fixed_news_forecast_diagnostic.csv"
        ).to_dict("records"),
        news_intervals=pd.read_csv(
            ROOT / "reports/news_interval_review/summary.csv"
        ).to_dict("records"),
        quality=quality,
        sensitivity=sensitivity.to_dict("records"),
        detectors=detectors.to_dict("records"),
        foundation=foundation.to_dict("records"),
        burden=pd.read_csv(
            ROOT / "reports/peer_residual_review/monitoring_burden.csv"
        ).to_dict("records"),
        orsk=pd.read_csv(ROOT / "reports/peer_residual_review/case_yoy.csv").to_dict(
            "records"
        ),
        news_lead=pd.read_csv(
            ROOT / "reports/peer_residual_review/news_lead_sensitivity.csv"
        ).to_dict("records"),
    )
    out = ROOT / "artifacts/presentation_data.json"
    out.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print("Evidence exported:", out)


if __name__ == "__main__":
    main()
