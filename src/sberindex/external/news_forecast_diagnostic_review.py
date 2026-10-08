"""Fixed low-complexity rules with origin-only availability, not trained news gain."""

import numpy as np
import pandas as pd

from sberindex.detection.event_conditioned_review import registry_events
from sberindex.paths import ROOT

OUT = ROOT / "reports/news_event_extraction_review"


def known_at_origin(events, tid, origin, lag=0):
    asof = (pd.Period(origin, "M") + 1).to_timestamp() + pd.DateOffset(months=lag)
    start = (pd.Period(origin, "M") - 1).to_timestamp()
    return any(
        e["territory_id"] == tid and start <= pd.Timestamp(e["available_from"]) < asof
        for e in events
    )


def date_balanced_mae(errors, targets):
    frame = pd.DataFrame(
        {"error": np.asarray(errors, dtype=float), "target": list(targets)}
    )
    return float(frame.groupby("target").error.mean().mean())


def run():
    events = registry_events()
    p = pd.read_parquet(ROOT / "reports/growth_bridge_review/predictions.parquet")
    p = p[p.model.eq("hierarchy05_growth_bridge") & p.horizon.eq(1)].copy()
    p["known_news"] = [
        known_at_origin(events, int(r.territory_id), r.origin) for r in p.itertuples()
    ]
    rows = []
    for rule, coefficient in [
        ("baseline", 0.0),
        ("fixed_negative_3pct", -0.03),
        ("fixed_positive_3pct", 0.03),
    ]:
        prediction = p.predicted * np.exp(coefficient * p.known_news.to_numpy())
        ae = abs(p.actual - prediction)
        for scope, mask in [
            ("entire_panel", np.ones(len(p), bool)),
            ("available_news_subset", p.known_news.to_numpy()),
        ]:
            g = p.loc[mask]
            rows.append(
                {
                    "rule": rule,
                    "scope": scope,
                    "n": int(mask.sum()),
                    "MAE": date_balanced_mae(ae[mask], g.target)
                    if mask.any()
                    else None,
                    "MAE_pooled_rows": float(ae[mask].mean()) if mask.any() else None,
                    "evaluated_pairs": int(mask.sum()),
                    "evaluated_dates": int(g.target.nunique()),
                    "WAPE": float(ae[mask].sum() / g.actual.sum())
                    if mask.any()
                    else None,
                    "trained_on_2023_news": False,
                }
            )
    p[["territory_id", "origin", "target", "known_news"]].to_csv(
        OUT / "forecast_news_availability.csv", index=False
    )
    pd.DataFrame(rows).to_csv(OUT / "fixed_news_forecast_diagnostic.csv", index=False)
    (OUT / "FORECAST_DIAGNOSTIC.md").write_text(
        "# Новостная поправка: фиксированная диагностика\n\n"
        + pd.DataFrame(rows).to_csv(index=False)
        + "\nMAE вычисляется сначала внутри каждого target-месяца, затем равными весами по месяцам; pooled row MAE сохранена отдельно для аудита. Число оценённых пар и дат указано явно. Фиксированные ±0.03 в логарифме заданы без выбора по результатам; это чувствительность, не обученная модель или улучшение. Новости разрешены только до первого дня после origin; география — явно названные МО. Исходный прогноз использует расходы, доступные по принятому в проекте сценарию. Сравнение обоих знаков не является выбором лучшего знака.\n\nКорпус 2023 года выборочный, отсутствие новости неизвестно. На нём нельзя честно обучить общий предиктор новостных шоков/отсутствия событий. Наличие источника не доказывает шок расходов.\n"
    )
    return rows


if __name__ == "__main__":
    print(run())
