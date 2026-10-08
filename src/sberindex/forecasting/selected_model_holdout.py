"""Freeze the selected unchanged model without reading any 2025 target values."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sberindex.forecasting.growth_bridge_review import bridge_prediction
from sberindex.forecasting.hierarchical_review import fit_profiles
from sberindex.paths import ROOT


def forecast_selected(panel, regions, origin, horizons, rate, weight=0.5, minimum_n=10):
    history = pd.period_range("2023-01", "2023-12", freq="M").astype(str)
    if origin not in panel.columns:
        raise ValueError("Anchor month missing")
    train = panel.reindex(columns=history)
    eligible = train.index[np.isfinite(train).all(axis=1) & (train > 0).all(axis=1)]
    gp, rp = fit_profiles(train, regions, minimum_n)
    records = []
    for territory in eligible:
        value = panel.loc[territory, origin]
        if not np.isfinite(value) or value <= 0:
            continue
        region = regions.get(territory, np.nan)
        profile = (1 - weight) * gp + weight * rp.get(region, gp)
        for horizon in horizons:
            if int(horizon) != horizon or horizon < 1:
                raise ValueError("Positive integer horizon required")
            records.append(
                {
                    "territory_id": int(territory),
                    "origin": origin,
                    "target": str(pd.Period(origin, "M") + horizon),
                    "horizon": int(horizon),
                    "model": "hierarchy05_growth_bridge",
                    "origin_value": float(value),
                    "predicted": bridge_prediction(
                        value, profile, pd.Period(origin, "M").month - 1, horizon, rate
                    ),
                }
            )
    return pd.DataFrame(records)


def main():
    cfg_path = ROOT / "configs/selected_model_holdout.json"
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / cfg["output_directory"]
    if out.exists():
        raise FileExistsError(
            "Frozen directory already exists: never overwrite a freeze"
        )
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    # Remove future targets before pivoting; no observed 2025 values enter this run.
    raw = raw[(raw.category == cfg["category"]) & (raw.date <= cfg["origin"])]
    panel = raw.pivot(index="territory_id", columns="date", values="value").sort_index()
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    regions = lookup[lookup.year == 2023].set_index("territory_id").region_code
    selected = json.loads((ROOT / "configs/growth_bridge_review.json").read_text())
    source = json.loads((ROOT / selected["source_record"]).read_text())
    predictions = forecast_selected(
        panel,
        regions,
        cfg["origin"],
        cfg["horizons"],
        source["growth_pct"] / 100,
        selected["region_weight"],
        selected["minimum_region_municipalities"],
    )
    out.mkdir(parents=True)
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.12g")
    inputs = [
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "configs/growth_bridge_review.json",
        "configs/selected_model_holdout.json",
        selected["source_record"],
        "src/sberindex/forecasting/selected_model_holdout.py",
        "src/sberindex/forecasting/hierarchical_review.py",
        "src/sberindex/forecasting/growth_bridge_review.py",
    ]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {
        "status": "awaiting_official_target_data",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_tree_modified": True,
        "input_sha256": {name: sha(ROOT / name) for name in inputs},
        "output_sha256": {"predictions.csv": sha(out / "predictions.csv")},
        "model": "hierarchy05_growth_bridge",
        "origin": cfg["origin"],
        "horizons": cfg["horizons"],
        "target_months": sorted(predictions.target.unique()),
        "rows": len(predictions),
        "municipalities": int(predictions.territory_id.nunique()),
        "fit_year": 2023,
        "growth_rate": 0.172,
        "target_values_read": False,
        "metrics_computed": False,
        "independent_evaluation_performed": False,
        "reporting_lag_assumption_months": 0,
        "limitations": "Computed in 2026 after repeated inspection of 2024. Not issued in 2024. Same frozen 2023 profile and 17.2% growth reused for 2025; no retuning or 2025 facts. December anchor availability assumes lag 0. Geography vintage unverified. Does not prove independent future performance.",
    }
    (out / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "REPORT.md").write_text(
        f"""# Прогноз выбранной модели на 2025 год\n\nЗафиксировано {len(predictions)} прогнозов для {manifest["municipalities"]} МО из последнего известного месяца — декабря 2024. Горизонты: 1, 3, 6, 12 месяцев; цели: январь, март, июнь, декабрь 2025.\n\nМодель и параметры не изменены: профиль 2023, региональный вес 0,5, рост 17,2%. Источник Росстата 2023 используется повторно как фиксированное допущение, а не актуальная оценка роста 2025. Целевые расходы 2025 не читались, MAE не рассчитана. Это историческая симуляция, вычисленная в 2026 году, не прогноз, реально выданный в 2024.\n\n`freeze_manifest.json` хранит точное время вычисления и SHA256 входов, кода и файла прогнозов. Повтор команды не перезаписывает фиксацию. Старый ансамбль в `reports/next_holdout` сохранён без изменений. После получения сопоставимых официальных фактов оценивать этот файл отдельно, на его точных ключах, без переобучения и изменения коэффициента. Доступность декабрьского факта предполагает нулевую задержку; исторические версии географии не подтверждены.\n"""
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ["rows", "municipalities", "target_months", "frozen_at_utc"]
            }
        )
    )


if __name__ == "__main__":
    main()
