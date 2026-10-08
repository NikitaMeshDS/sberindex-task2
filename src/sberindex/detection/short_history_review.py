"""Twelve training months, independent synthetic selection, onset and boundary metrics."""

import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.detection import event_metric_review as legacy

OUT = ROOT / "reports/short_history_review"
CONFIG = ROOT / "configs/short_history_review.json"
ONLINE = legacy.ONLINE + ("cusum_spike",)


def config():
    return json.loads(CONFIG.read_text())


def make_panel(seed, per_shape, cfg=None):
    cfg = config() if cfg is None else cfg
    rng = np.random.default_rng(seed)
    train, horizon = cfg["train_months"], cfg["evaluation_months"]
    t = np.arange(train + horizon)
    panel = []
    for shape in ("no_change", "step", "pulse"):
        for i in range(per_shape):
            onset = int(rng.integers(cfg["onset_min"], cfg["onset_max"] + 1))
            duration = int(rng.integers(cfg["duration_min"], cfg["duration_max"] + 1))
            amplitude = cfg["amplitudes_log"][i % 3] * (-1 if i % 2 else 1)
            x = rng.uniform(7, 9) + rng.uniform(0.08, 0.22) * np.sin(
                2 * np.pi * t / 12 + rng.uniform(0, 2 * np.pi)
            )
            truth = []
            if shape != "no_change":
                x[train + onset :] += amplitude
                truth = [onset]
                if shape == "pulse":
                    x[train + onset + duration :] -= amplitude
                    truth.append(onset + duration)
            x += rng.normal(0, cfg["noise_sigma"], len(t))
            panel.append(
                dict(
                    series_id=f"{seed}:{shape}:{i:04}",
                    seed=seed,
                    shape=shape,
                    amplitude_log=abs(amplitude) if truth else 0.0,
                    truth=truth,
                    log_values=x,
                    residual=legacy.forecast_residual(x, train),
                )
            )
    return panel


def calibrate(null_signal, cfg=None):
    cfg = config() if cfg is None else cfg
    parameters = {}
    scores = {}
    for method in legacy.ONLINE:
        scores[method] = legacy.online_scores(null_signal, method, cfg["noise_sigma"])
        parameters[method] = float(
            np.quantile(np.nanmax(scores[method], axis=1), cfg["threshold_quantile"])
        )
    family = np.maximum(
        scores["cusum"] / parameters["cusum"], scores["spike"] / parameters["spike"]
    )
    parameters["cusum_spike"] = float(
        np.quantile(np.nanmax(family, axis=1), cfg["threshold_quantile"])
    )
    return parameters


def predict_online(signal, method, parameters, cfg=None):
    cfg = config() if cfg is None else cfg
    if method == "cusum_spike":
        score = np.maximum(
            legacy.online_scores(signal, "cusum", cfg["noise_sigma"])
            / parameters["cusum"],
            legacy.online_scores(signal, "spike", cfg["noise_sigma"])
            / parameters["spike"],
        )
    else:
        score = legacy.online_scores(signal, method, cfg["noise_sigma"])
    return [legacy.alarm_events(row > parameters[method]) for row in score]


def onset_counts(item, predictions, tolerance=1):
    truth = item["truth"]
    onset = truth[:1]
    matches = legacy.match_events(onset, predictions, tolerance)
    used = {b for _, b in matches}
    rest = [p for p in predictions if p not in used]
    neutral = legacy.match_events(truth[1:], rest, tolerance)
    return len(matches), len(rest) - len(neutral), len(onset) - len(matches)


def choose(summary, cfg=None):
    cfg = config() if cfg is None else cfg
    eligible = summary[
        (summary.split == "selection")
        & (summary["shape"] == "all")
        & summary.method.isin(ONLINE)
        & (
            summary.null_false_alarms_per100_mo_month
            <= cfg["null_false_alarms_budget_per100_mo_month"]
        )
    ]
    if eligible.empty:
        return None
    return str(
        eligible.sort_values(["onset_f1", "method"], ascending=[False, True])
        .iloc[0]
        .method
    )


def run():
    cfg = config()
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = [
        r
        for r in make_panel(cfg["calibration_seed"], cfg["calibration_null_n"], cfg)
        if r["shape"] == "no_change"
    ]
    parameters = calibrate(np.stack([r["residual"] for r in calibration]), cfg)
    penalty_rows = []
    for method in legacy.OFFLINE:
        for penalty in cfg[f"{method}_penalties"]:
            events = [
                legacy.offline_events(
                    r["residual"], method, penalty, cfg["noise_sigma"], cfg
                )
                for r in calibration
            ]
            rate = (
                100 * sum(map(len, events)) / (len(events) * cfg["evaluation_months"])
            )
            penalty_rows.append(dict(method=method, penalty=penalty, null_rate=rate))
        passing = [
            r
            for r in penalty_rows
            if r["method"] == method
            and r["null_rate"] <= cfg["null_false_alarms_budget_per100_mo_month"]
        ]
        if not passing:
            raise ValueError(f"{method}: no prespecified penalty passes")
        parameters[method] = min(r["penalty"] for r in passing)
    rows, panel_rows = [], []
    selected = None
    for split, seeds in [
        ("calibration", [cfg["calibration_seed"]]),
        ("selection", [cfg["selection_seed"]]),
        ("evaluation", cfg["evaluation_seeds"]),
    ]:
        for seed in seeds:
            panel = (
                calibration
                if split == "calibration"
                else make_panel(seed, cfg["series_per_shape"], cfg)
            )
            signal = np.stack([r["residual"] for r in panel])
            for item in panel:
                for t, residual in enumerate(item["residual"]):
                    panel_rows.append(
                        dict(
                            split=split,
                            series_id=item["series_id"],
                            shape=item["shape"],
                            month=t,
                            residual_log=residual,
                            onset=t in item["truth"][:1],
                            boundary=t in item["truth"],
                        )
                    )
            for method in ONLINE + legacy.OFFLINE:
                predictions = (
                    predict_online(signal, method, parameters, cfg)
                    if method in ONLINE
                    else [
                        legacy.offline_events(
                            r["residual"],
                            method,
                            parameters[method],
                            cfg["noise_sigma"],
                            cfg,
                        )
                        for r in panel
                    ]
                )
                measured = legacy.measure(
                    panel,
                    predictions,
                    "cusum" if method == "cusum_spike" else method,
                    split,
                    cfg,
                )
                for item, pred, row in zip(panel, predictions, measured):
                    row["method"] = method
                    row["onset_tp"], row["onset_fp"], row["onset_fn"] = onset_counts(
                        item, pred, cfg["tolerance_months"]
                    )
                    matches = legacy.match_events(
                        item["truth"][:1], pred, cfg["tolerance_months"]
                    )
                    row["onset_delay"] = (
                        [max(0, b - a) for a, b in matches] if method in ONLINE else []
                    )
                rows.extend(measured)
        if split == "selection":
            selected = choose(summarize(rows, cfg), cfg)
    summary = summarize(rows, cfg)
    serial = pd.DataFrame(rows)
    for col in (
        "signed_errors",
        "detection_delays",
        "availability_delays",
        "onset_delay",
    ):
        serial[col] = serial[col].map(json.dumps)
    serial.to_csv(OUT / "series_metrics.csv", index=False)
    summary.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(panel_rows).to_parquet(
        OUT / "synthetic_residuals.parquet", index=False
    )
    pd.DataFrame(penalty_rows).to_csv(OUT / "penalty_calibration.csv", index=False)
    choice = dict(
        selected_online_method=selected,
        objective="maximum onset F1 among selection null FAR <=0.2; ties alphabetic",
        fixed_before_evaluation=True,
        real_false_alarm_guarantee=False,
        historical_default_replaced=False,
    )
    (OUT / "choice.json").write_text(
        json.dumps(choice, ensure_ascii=False, indent=2) + "\n"
    )
    paths = [
        "configs/short_history_review.json",
        "docs/protocols/SHORT_HISTORY_REVIEW.md",
        "src/sberindex/detection/short_history_review.py",
        "src/sberindex/detection/event_metric_review.py",
        "src/sberindex/detection/asof_detector_audit.py",
        "src/sberindex/detection/bocpd_audit.py",
        "configs/detectors.json",
    ]
    audit = dict(
        status="passed",
        config=cfg,
        parameters=parameters,
        choice=choice,
        input_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
        },
        evaluation_series=3 * cfg["series_per_shape"] * len(cfg["evaluation_seeds"]),
        independent_real_time_holdout=False,
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    table = summary[(summary.split == "evaluation") & (summary["shape"] == "all")]
    print(
        table[
            [
                "method",
                "f1",
                "onset_f1",
                "onset_recall",
                "null_false_alarms_per100_mo_month",
                "mean_onset_delay_months",
            ]
        ].to_string(index=False)
    )
    print(json.dumps(choice))
    body = "# Детекторы: 12 месяцев истории и 12 месяцев оценки\n\n"
    body += f"Выбранный онлайн метод: **{selected}**, по отбору до чтения оценочных наборов. Все восемь методов показаны, включая совместный CUSUM + spike.\n\n"
    body += "| Метод | F1 всех границ | F1 начала | Recall начала | Null FAR / 100 МО-мес. | Задержка начала, мес. |\n|---|---:|---:|---:|---:|---:|\n"
    for row in table.itertuples():
        body += f"| {row.method} | {row.f1:.3f} | {row.onset_f1:.3f} | {row.onset_recall:.3f} | {row.null_false_alarms_per100_mo_month:.3f} | {row.mean_onset_delay_months:.3f} |\n"
    body += "\nОтдельные файлы содержат результаты отбора, null/step/pulse и каждый seed. Возврат pulse нейтрален только в метрике начала; в метрике всех границ он остаётся обязательной границей. ±1 месяц — локализация, положительная задержка условна на обнаруженных событиях. Пропуски не исчезают из recall. Офлайн ответы доступны лишь после конца окна, их задержку получения показывает comparison.csv.\n\n"
    body += "Порог recall 0,5 старого профиля не был математически невозможным: для детектора только начала максимальная pooled полнота составляла 2/3. Новый профиль отвечает отдельной задаче начала шока; он не переписывает старый неудачный отбор. Совместный канал имеет общую калибровку. Синтетика использует фиксированный сезонный прогноз с годом обучения, а не эмпирический null расходов; будущая реальная FAR и значимость превосходства не доказаны.\n\n"
    body += "Команда: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src python -m sberindex.detection.short_history_review`. Seeds, параметры и SHA256 — в audit.json. Новые случайные наборы независимы; реальный 2024 уже исследован.\n"
    (OUT / "REPORT.md").write_text(body)


def summarize(rows, cfg):
    summary = legacy.summarize(rows, cfg["evaluation_months"])
    frame = pd.DataFrame(rows)
    for index, row in summary.iterrows():
        part = frame[(frame.split == row["split"]) & (frame.method == row["method"])]
        if row["shape"] != "all":
            part = part[part["shape"] == row["shape"]]
        tp, fp, fn = [int(part[c].sum()) for c in ("onset_tp", "onset_fp", "onset_fn")]
        summary.loc[index, "onset_precision"] = tp / (tp + fp) if tp + fp else np.nan
        summary.loc[index, "onset_recall"] = tp / (tp + fn) if tp + fn else np.nan
        summary.loc[index, "onset_f1"] = (
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        )
        delays = sum(part.onset_delay.tolist(), [])
        summary.loc[index, "mean_onset_delay_months"] = (
            np.mean(delays) if delays else np.nan
        )
    return summary


if __name__ == "__main__":
    run()
