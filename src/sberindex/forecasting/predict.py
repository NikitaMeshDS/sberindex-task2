"""Configurable inference of the selected seasonal model on monthly input data."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sberindex.forecasting.selected_model_holdout import forecast_selected


def integer(value, name, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(f"{name}: требуется целое число >= {minimum}")
    return int(value)


def ids(values, name):
    numeric = pd.to_numeric(values, errors="raise")
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0).any()
        or (numeric % 1 != 0).any()
    ):
        raise ValueError(f"{name}: ожидаются неотрицательные целочисленные ID")
    return numeric.astype("int64")


def make_forecast(raw, lookup, cfg):
    """Filter to the declared information cutoff before fitting any profiles."""
    required = {"territory_id", "date", "category", "value"}
    if not required.issubset(raw.columns):
        raise ValueError(
            f"Таблица расходов: отсутствуют {sorted(required - set(raw.columns))}"
        )
    if not {"territory_id", "region_code"}.issubset(lookup.columns):
        raise ValueError("Справочник должен содержать territory_id и region_code")
    origin = str(pd.Period(cfg["origin"], freq="M"))
    year = integer(cfg["profile_year"], "profile_year")
    if pd.Period(f"{year}-12", "M") > pd.Period(origin, "M"):
        raise ValueError(
            "Год сезонного профиля должен полностью завершиться к месяцу прогноза"
        )
    horizons = [integer(h, "horizon") for h in cfg["horizons"]]
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("Нужен непустой список уникальных горизонтов")
    rate, weight = float(cfg["growth_rate"]), float(cfg["region_weight"])
    minimum = integer(
        cfg["minimum_region_municipalities"], "minimum_region_municipalities"
    )
    if (
        not np.isfinite(rate)
        or rate <= -1
        or not np.isfinite(weight)
        or not 0 <= weight <= 1
    ):
        raise ValueError("growth_rate должен быть > -1; region_weight — от 0 до 1")
    as_of = pd.Timestamp(cfg.get("as_of_date", pd.Period(origin, "M").end_time.date()))
    if pd.isna(as_of) or as_of.date() < pd.Period(origin, "M").end_time.date():
        raise ValueError("as_of_date должна быть не раньше конца месяца origin")
    if cfg.get("growth_available_from"):
        growth_date = pd.Timestamp(cfg["growth_available_from"])
        if pd.isna(growth_date) or growth_date > as_of:
            raise ValueError(
                "Источник роста ещё не доступен на as_of_date либо дата некорректна"
            )
    raw = raw.copy()
    dates = pd.to_datetime(raw["date"], format="mixed", errors="raise")
    if dates.isna().any():
        raise ValueError("Пустая дата расхода")
    raw["date"] = dates.dt.to_period("M").astype(str)
    future_count = int((raw.date > origin).sum())
    history = raw[(raw.date <= origin) & (raw.category == cfg["category"])].copy()
    unavailable = 0
    if "available_from" in history.columns:
        available = pd.to_datetime(
            history.available_from, format="mixed", errors="raise"
        )
        if available.isna().any():
            raise ValueError("available_from задан не для всех исторических строк")
        unavailable = int((available > as_of).sum())
        history = history.loc[available <= as_of].copy()
    history["territory_id"] = ids(history.territory_id, "territory_id")
    history["value"] = pd.to_numeric(history.value, errors="raise")
    if history.duplicated(["territory_id", "date"]).any():
        raise ValueError(
            "Повтор territory_id / месяц / категория: проверьте агрегацию данных"
        )
    if origin not in set(history.date):
        raise ValueError("Нет доступных расходов за origin")
    lookup = lookup.copy()
    if "year" in lookup.columns:
        available_years = pd.to_numeric(lookup.year, errors="raise")
        preferred = cfg.get("geography_year")
        if preferred is None:
            candidates = available_years[available_years <= pd.Period(origin, "M").year]
            if candidates.empty:
                raise ValueError(
                    "Нет справочника территории, датированного не позже origin"
                )
            preferred = int(candidates.max())
        preferred = integer(preferred, "geography_year")
        if preferred > pd.Period(origin, "M").year:
            raise ValueError("geography_year не должен быть позже origin")
        lookup = lookup[available_years == preferred].copy()
        if lookup.empty:
            raise ValueError("В справочнике нет geography_year")
    lookup["territory_id"] = ids(lookup.territory_id, "lookup.territory_id")
    if lookup.territory_id.duplicated().any():
        raise ValueError("Справочник содержит повтор territory_id")
    nonmissing = lookup.region_code.notna()
    lookup.loc[nonmissing, "region_code"] = ids(
        lookup.loc[nonmissing, "region_code"], "region_code"
    )
    regions = lookup.set_index("territory_id").region_code
    panel = history.pivot(
        index="territory_id", columns="date", values="value"
    ).sort_index()
    months = pd.period_range(f"{year}-01", f"{year}-12", freq="M").astype(str)
    train = panel.reindex(columns=months)
    eligible = np.isfinite(train).all(axis=1) & (train > 0).all(axis=1)
    if not eligible.any():
        raise ValueError(
            "Нет МО с полными 12 положительными наблюдениями в profile_year"
        )
    forecasts = forecast_selected(
        panel, regions, origin, horizons, rate, weight, minimum, fit_year=year
    )
    if forecasts.empty or not np.isfinite(forecasts.predicted).all():
        raise ValueError("Нет пригодных МО с положительным уровнем origin для прогноза")
    eligible_regions = regions.reindex(train.index[eligible])
    counts = eligible_regions.value_counts()
    forecasts["region_code"] = forecasts.territory_id.map(regions)
    forecasts["regional_profile_used"] = (
        forecasts.region_code.map(counts).fillna(0) >= minimum
    )
    audit = {
        "origin": origin,
        "as_of_date": str(as_of.date()),
        "profile_year": year,
        "future_rows_excluded": future_count,
        "unavailable_rows_excluded": unavailable,
        "input_municipalities": int(history.territory_id.nunique()),
        "profile_eligible_municipalities": int(eligible.sum()),
        "forecast_municipalities": int(forecasts.territory_id.nunique()),
        "municipalities_without_forecast": int(
            history.territory_id.nunique() - forecasts.territory_id.nunique()
        ),
        "global_fallback_forecasts": int((~forecasts.regional_profile_used).sum()),
        "availability_assumption": "Row availability checked"
        if "available_from" in raw.columns
        else "No publication dates: all history through origin assumed available at as_of_date",
        "geography_availability": "Geography publication dates not verified; user-provided vintage",
        "metrics_computed": False,
        "new_data_accuracy_validated": False,
    }
    return forecasts, audit


def read_table(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Входные таблицы должны быть CSV или Parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    paths = {
        name: (cfg_path.parent / cfg[name]).resolve()
        for name in ["input_path", "lookup_path", "output_directory"]
    }
    if paths["output_directory"].exists():
        raise FileExistsError(
            "output_directory уже существует; задайте новую папку, чтобы сохранить предыдущий прогноз"
        )
    forecasts, audit = make_forecast(
        read_table(paths["input_path"]), read_table(paths["lookup_path"]), cfg
    )
    out = paths["output_directory"]
    out.mkdir(parents=True)
    forecasts.to_csv(out / "predictions.csv", index=False, float_format="%.12g")
    (out / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    )
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit.update(
        {
            "model": "hierarchy05_growth_bridge",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(forecasts),
            "input_sha256": {
                str(cfg_path): sha(cfg_path),
                str(paths["input_path"]): sha(paths["input_path"]),
                str(paths["lookup_path"]): sha(paths["lookup_path"]),
            },
            "code_sha256": {
                p.name: sha(p)
                for p in [
                    Path(__file__),
                    Path(__file__).with_name("selected_model_holdout.py"),
                    Path(__file__).with_name("hierarchical_review.py"),
                    Path(__file__).with_name("growth_bridge_review.py"),
                ]
            },
            "output_sha256": sha(out / "predictions.csv"),
        }
    )
    (out / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "forecast_created",
                "rows": len(forecasts),
                "municipalities": audit["forecast_municipalities"],
                "output_directory": str(out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
