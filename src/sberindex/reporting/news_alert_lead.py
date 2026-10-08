"""Scenario lead times to expense alerts, with missing coverage and censoring explicit."""

import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.detection.short_history_review import (
    predict_online,
    config as detector_config,
)

OUT = ROOT / "reports/news_alert_lead"


def release_date(month, lag):
    """First day after observation month, plus nonnegative full release-lag months."""
    if int(lag) != lag or lag < 0:
        raise ValueError("Release lag must be a nonnegative integer")
    return (pd.Period(month, "M") + 1 + int(lag)).start_time


def alert_lead(available, event_start, alarm_months, lag, last_observed):
    news = pd.Timestamp(available)
    event = (
        pd.Timestamp(event_start)
        if event_start is not None and pd.notna(event_start)
        else None
    )
    last_release = release_date(last_observed, lag)
    result = dict(
        status="right_censored_no_subsequent_alert",
        news_to_alert_days=None,
        alert_release_date=None,
        alarm_target=None,
        event_to_news_days=(news - event).days if event is not None else None,
        observation_end=last_release.strftime("%Y-%m-%d"),
    )
    if news > last_release:
        result["status"] = "news_outside_observation_window"
        return result
    future = [
        (release_date(month, lag), month)
        for month in alarm_months
        if release_date(month, lag) >= news
    ]
    if future:
        date, month = min(future)
        result.update(
            status="subsequent_alert",
            news_to_alert_days=(date - news).days,
            alert_release_date=date.strftime("%Y-%m-%d"),
            alarm_target=month,
        )
    return result


def coverage_status(city, forecast_ids, expense_ids):
    if city in forecast_ids:
        return "forecast_available"
    return (
        "no_eligible_h1_forecast_for_this_territory"
        if city in expense_ids
        else "no_comparable_expense_series"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_parquet(
        ROOT / "reports/growth_bridge_review/predictions.parquet"
    )
    selected = json.loads(
        (ROOT / "reports/growth_bridge_review/selection.json").read_text()
    )
    # The model name is a fixed study choice; never pick by case errors.
    forecasts = forecasts[
        (forecasts.model == "hierarchy05_growth_bridge") & (forecasts.horizon == 1)
    ]
    residuals = forecasts.assign(
        residual_log=np.log(forecasts.actual / forecasts.predicted)
    ).pivot(index="territory_id", columns="target", values="residual_log")
    months = pd.period_range("2024-01", "2024-12", freq="M").astype(str).tolist()
    residuals = residuals.reindex(columns=months)
    raw_ids = pd.read_parquet(
        ROOT / "data/consumption.parquet", columns=["territory_id", "category"]
    )
    expense_ids = set(raw_ids.loc[raw_ids.category == "Все категории", "territory_id"])

    protocol = json.loads(
        (ROOT / "reports/short_history_review/audit.json").read_text()
    )
    method = protocol["choice"]["selected_online_method"]
    if method is None:
        raise ValueError("No selected detector; do not select on real news cases")
    registry = pd.read_csv(ROOT / "data/external/regional_news_review/registry.csv")
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    lookup = lookup[lookup.year == 2024]
    rows, alarm_rows = [], []
    predictions = {}
    for candidate in ("spike", "cusum", "cusum_spike", method):
        if candidate in predictions:
            continue
        alarms = predict_online(
            residuals.to_numpy(float),
            candidate,
            protocol["parameters"],
            detector_config(),
        )
        predictions[candidate] = dict(zip(residuals.index, alarms))
        for city, times in zip(residuals.index, alarms):
            for time in times:
                alarm_rows.append(
                    dict(
                        territory_id=int(city),
                        method=candidate,
                        target=months[time],
                        residual_log=residuals.loc[city].iloc[time],
                    )
                )
    pd.DataFrame(alarm_rows).to_csv(OUT / "real_alarms.csv", index=False)
    observed_count = int(np.isfinite(residuals.to_numpy(float)).sum())
    burden = pd.DataFrame(
        [
            dict(
                method=candidate,
                observed_mo_months=observed_count,
                alarm_episodes=sum(map(len, predictions[candidate].values())),
                alarm_episodes_per100_observed_mo_months=100
                * sum(map(len, predictions[candidate].values()))
                / observed_count,
                false_alarm_rate_identified=False,
            )
            for candidate in predictions
        ]
    )
    burden.to_csv(OUT / "monitoring_burden.csv", index=False)
    for claim in registry.itertuples():
        ids = json.loads(claim.territory_ids)
        expanded = not bool(ids)
        if expanded:
            ids = (
                lookup.loc[lookup.region_code == claim.region_code, "territory_id"]
                .drop_duplicates()
                .tolist()
            )
        if not ids:
            ids = [None]
        for city in ids:
            for candidate in predictions:
                for lag in (0, 1, 2):
                    common = dict(
                        claim_id=claim.claim_id,
                        episode_id=claim.episode_id,
                        territory_id=city,
                        regional_exposure=expanded,
                        method=candidate,
                        release_lag_months=lag,
                        published_date=claim.published_date,
                        available_from=claim.available_from,
                        event_start=claim.event_start,
                        source_url=claim.source_url,
                        causal_attribution=False,
                        historical_vintage_verified=False,
                    )
                    if city not in residuals.index:
                        rows.append(
                            dict(
                                **common,
                                status=coverage_status(
                                    city, residuals.index, expense_ids
                                ),
                            )
                        )
                        continue
                    if pd.Timestamp(claim.available_from).year < 2024:
                        rows.append(
                            dict(
                                **common,
                                status="event_before_forecast_evaluation_window",
                            )
                        )
                        continue
                    event = claim.event_start if pd.notna(claim.event_start) else None
                    anchor = (
                        max(pd.Period(claim.available_from, "M"), pd.Period(event, "M"))
                        if event
                        else pd.Period(claim.available_from, "M")
                    )
                    alarm_months = [
                        months[t]
                        for t in predictions[candidate][city]
                        if pd.Period(months[t], "M") >= anchor
                    ]
                    observed = residuals.loc[city].dropna()
                    if observed.empty:
                        rows.append(
                            dict(**common, status="no_observed_forecast_residual")
                        )
                        continue
                    eligible_observations = [
                        m for m in observed.index if pd.Period(m, "M") >= anchor
                    ]
                    data_lead = alert_lead(
                        claim.available_from,
                        event,
                        eligible_observations,
                        lag,
                        observed.index[-1],
                    )
                    common["first_eligible_expense_release_date"] = data_lead[
                        "alert_release_date"
                    ]
                    common["news_to_expense_release_days"] = data_lead[
                        "news_to_alert_days"
                    ]
                    rows.append(
                        dict(
                            **common,
                            **alert_lead(
                                claim.available_from,
                                event,
                                alarm_months,
                                lag,
                                observed.index[-1],
                            ),
                        )
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "claim_lead_times.csv", index=False)
    counts = (
        frame.groupby(["method", "release_lag_months", "status"])
        .size()
        .rename("claim_territory_rows")
        .reset_index()
    )
    counts.to_csv(OUT / "coverage.csv", index=False)
    observed = frame[(frame.status == "subsequent_alert") & (frame.method == method)]
    summary = (
        observed.groupby("release_lag_months")
        .agg(
            rows=("news_to_alert_days", "size"),
            median_days=("news_to_alert_days", "median"),
            min_days=("news_to_alert_days", "min"),
            max_days=("news_to_alert_days", "max"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "conditional_summary.csv", index=False)
    cases = frame[
        (frame.territory_id.isin([1673, 1665]))
        & (frame.episode_id.isin(["orsk_apr2024", "orenburg_flood_apr2024"]))
    ]
    cases.to_csv(OUT / "orsk_orenburg.csv", index=False)
    body = "# Новости: опережение доступности расходной тревоги\n\n"
    body += f"Детектор **{method}** выбран на короткой синтетике, а затем применён к лог-ошибкам h1 модели с годовой поправкой; пороги на реальных случаях не подбирались. CUSUM, spike и их объединение показаны как альтернативы. Такой перенос — проверяемый прототип: эмпирическая FAR расходов не известна.\n\n"
    body += "Время новости — available_from из реестра (обычно публикация + 1 день, сценарий). Выпуск расходов — первый день после наблюдаемого месяца + 0/1/2 полных месяца задержки. Исторические даты выпуска неизвестны. Берётся первая последующая тревога для месяца не раньше месяца сообщения/начала события.\n\n"
    body += "| Кейс / метод | Лаг выпуска, мес. | Статус | Месяц тревоги | Опережение, дни |\n|---|---:|---|---|---:|\n"
    compact = cases[
        (cases.method == method)
        & (cases.claim_id.isin(registry.drop_duplicates("episode_id").claim_id))
    ]
    for row in compact.itertuples():
        value = (
            f"{row.news_to_alert_days:.0f}" if pd.notna(row.news_to_alert_days) else "—"
        )
        body += f"| {int(row.territory_id)} / {row.method} | {row.release_lag_months} | {row.status} | {row.alarm_target if pd.notna(row.alarm_target) else '—'} | {value} |\n"
    body += "\nДля каждого claim и территории сохранены все три сценария, в том числе отсутствие рядов, события 2023 вне окна прогноза и цензурированные случаи без тревоги. Региональная экспозиция не равна доказанному ущербу каждому МО. Строки одного эпизода и региона зависимы; их нельзя считать независимыми событиями.\n\n"
    body += "Положительное опережение показывает доступность контекста раньше некоторой последующей тревоги. Оно не доказывает, что тревога вызвана именно этой новостью или что шок предсказан до события. Для этого нужна независимая разметка расходных сдвигов. Нельзя заменять отсутствие тревоги выдуманной задержкой и усреднять только обнаруженные случаи без раскрытия цензурирования.\n\n"
    body += "Полные исходные числа: claim_lead_times.csv; реальные тревоги: real_alarms.csv; статусы покрытия: coverage.csv. Воспроизведение: `PYTHONPATH=src python -m sberindex.reporting.news_alert_lead`.\n"
    body += "\nСценарии задержки сдвигают доступность наблюдений/тревог; h1 прогнозы остаются из опыта лаг 0, новые прогнозные backtest с задержанным origin не выполнялись. Реальный объём тревог — monitoring_burden.csv; он не равен FAR без разметки.\n"
    body += "\nДля случая без тревоги отдельно сохранены first_eligible_expense_release_date и news_to_expense_release_days: опережение информации о расходах, не обнаружения шока. Для новости Орска, доступной 7 апреля, расходы апреля доступны по сценариям 1 мая / 1 июня / 1 июля: 24 / 55 / 85 дней. Наличие этого информационного интервала не означает, что в расходах возник сдвиг.\n"
    (OUT / "REPORT.md").write_text(body)
    paths = [
        "src/sberindex/reporting/news_alert_lead.py",
        "reports/growth_bridge_review/predictions.parquet",
        "reports/short_history_review/audit.json",
        "data/external/regional_news_review/registry.csv",
        "results/municipal_lookup.csv",
        "data/consumption.parquet",
    ]
    audit = dict(
        status="passed",
        selected_detector=method,
        news_claims=len(registry),
        rows=len(frame),
        spending_release_vintage_verified=False,
        independent_real_holdout=False,
        source_selection=selected,
        input_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
        },
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        cases[cases.method == method][
            [
                "episode_id",
                "territory_id",
                "release_lag_months",
                "status",
                "alarm_target",
                "news_to_alert_days",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
