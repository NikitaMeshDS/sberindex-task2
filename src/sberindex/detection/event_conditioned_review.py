"""Fixed event-available EWMA threshold sensitivity, not a real FAR estimate."""

import json

import numpy as np
import pandas as pd

from sberindex.detection.peer_residual_review import alert_episodes, release_date
from sberindex.paths import ROOT

OUT = ROOT / "reports/event_conditioned_review"


def event_available(events, tid, month, release_lag=0, window=2):
    release = release_date(month, release_lag)
    for e in events:
        if int(e["territory_id"]) != int(tid):
            continue
        available = pd.Timestamp(e["available_from"])
        start = pd.Period(available, freq="M")
        target = pd.Period(month, freq="M")
        # Never retroactively lower thresholds for targets before publication month.
        if available <= release and start <= target < start + window:
            return True
    return False


def registry_events():
    r = pd.read_csv(ROOT / "data/external/regional_news_review/registry.csv").fillna("")
    events = []
    for row in r[r.source_url.str.contains(r"mchs\.gov\.ru", regex=True)].itertuples():
        for tid in json.loads(row.territory_ids or "[]"):
            events.append(
                {
                    "territory_id": int(tid),
                    "available_from": row.available_from,
                    "publication_date": row.published_date,
                    "event_start": row.event_start,
                    "event_role": row.event_role,
                    "episode_id": row.episode_id,
                    "source_url": row.source_url,
                    "snapshot_path": row.snapshot_path,
                }
            )
    return events


def run():
    cfg = json.loads((ROOT / "configs/event_conditioned_review.json").read_text())
    panel = pd.read_parquet(
        ROOT / "reports/peer_residual_review/detector_panel.parquet"
    )
    panel = panel[panel.method.eq("ewma") & panel.signal.eq("own")].copy()
    events = registry_events()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_csv(OUT / "official_exposure_registry.csv", index=False)
    records = []
    details = []
    for lag in cfg["release_lags_months"]:
        availability = np.array(
            [
                event_available(
                    events, r.territory_id, r.target, lag, cfg["event_window_months"]
                )
                for r in panel.itertuples()
            ]
        )
        for policy in ["baseline", "event_conditioned", "global_0.75"]:
            factor = np.ones(len(panel))
            if policy == "event_conditioned":
                factor[availability] = cfg["threshold_factor"]
            if policy == "global_0.75":
                factor[:] = 0.75
            scored = panel.copy()
            scored["event_available"] = availability
            scored["threshold_factor"] = factor
            scored["active_alert"] = scored.observed & (
                scored.score > scored.threshold * factor
            )
            scored["alert_episode"] = False
            for _, group in scored.groupby("territory_id", sort=False):
                sorted_group = group.sort_values("target")
                flags = alert_episodes(sorted_group.active_alert.to_numpy()[None, :])[0]
                scored.loc[sorted_group.index, "alert_episode"] = flags
            n = int(scored.observed.sum())
            unlabeled = ~scored.event_available
            records.append(
                {
                    "policy": policy,
                    "release_lag_months": lag,
                    "observed_months": n,
                    "event_available_months": int(
                        (scored.observed & availability).sum()
                    ),
                    "active_alert_months": int(scored.active_alert.sum()),
                    "episodes": int(scored.alert_episode.sum()),
                    "active_per100": 100 * scored.active_alert.sum() / n,
                    "unlabeled_active_months": int(
                        (scored.active_alert & unlabeled).sum()
                    ),
                    "real_false_alarm_rate": None,
                }
            )
            scored["policy"] = policy
            scored["release_lag_months"] = lag
            details.append(scored)
    result = pd.concat(details, ignore_index=True)
    result.to_parquet(OUT / "detector_panel.parquet", index=False)
    burden = pd.DataFrame(records)
    burden.to_csv(OUT / "monitoring_burden.csv", index=False)
    # All named flood exposures across three regions, not an optimized Orsk subset.
    flood = {
        e["territory_id"]
        for e in events
        if "flood" in e["episode_id"] or "orsk" in e["episode_id"]
    }
    case = result[result.territory_id.isin(flood)]
    case.to_csv(OUT / "flood_cases.csv", index=False)
    summary = (
        case.groupby(["territory_id", "policy", "release_lag_months"])
        .agg(
            active_months=("active_alert", "sum"),
            event_available_months=("event_available", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "flood_summary.csv", index=False)
    audit = {
        "status": "complete",
        "threshold_factor": 0.75,
        "threshold_frozen_from": "short_history_review",
        "all_panel_global_factor": 0.75,
        "real_false_alarm_rate": None,
        "absence_label": "unlabeled",
        "causal_effect_proven": False,
        "news_full_corpus": False,
        "event_registry_role": "official exposure context, not spending changepoint",
        "registry_rows": len(events),
        "flood_ids": sorted(flood),
        "BOCPD": "not modified: differing probability threshold and hazard cannot fairly reuse EWMA rule",
        "forecast_news_ablation": "fixed availability diagnostic only; sparse selected 2023 corpus does not support a learned complete-panel news model",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = burden[burden.release_lag_months.eq(0)]
    plt.bar(b.policy, b.active_per100)
    plt.ylabel("Alert months per 100 observed")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "burden.png")
    plt.close()
    (OUT / "REPORT.md").write_text(
        "# Детектор с новостным контекстом\n\n"
        + "```csv\n"
        + burden.to_csv(index=False)
        + "```"
        + "\n\nПорог EWMA заранее умножен на 0.75 только для явно указанных МО и двух месяцев после доступности новости. Публикация + сутки — сценарий. Сравнение: замороженный исходный порог и глобальное снижение до 0.75 на всей панели. Проверены лаги релиза 0/1/2 месяца; старые периоды до публикации не пересчитываются. Орск и остальные паводковые случаи представлены без настройки по их результату.\n\nАктивные месяцы вне выборочного реестра — неразмеченная нагрузка мониторинга, не ложные тревоги. Реестр показывает экспозицию события, а не истинный разрыв расходов. Обученную новостную поправку прогноза нельзя честно оценить на неполном событийно выбранном корпусе 2023 года. BOCPD hazard не изменён: сопоставимый механизм с фиксированным множителем потребовал бы отдельного протокола.\n"
    )
    return audit


if __name__ == "__main__":
    print(run())
