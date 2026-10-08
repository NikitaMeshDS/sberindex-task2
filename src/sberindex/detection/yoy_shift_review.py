"""Retrospective Jan-Mar panel calibration for sustained regional-relative YoY shifts."""

import json

import numpy as np
import pandas as pd

from sberindex.detection.asof_detector_audit import causal_scores
from sberindex.detection.detector_settings import load_settings
from sberindex.detection.event_conditioned_review import registry_events
from sberindex.paths import ROOT

OUT = ROOT / "reports/yoy_shift_review"


def regional_growth(values, regions, min_peers=5):
    growth = np.log(values[:, 12:] / values[:, :12])
    growth[~np.isfinite(growth)] = np.nan
    peer = np.full_like(growth, np.nan)
    for region in np.unique(regions[np.isfinite(regions)]):
        idx = np.flatnonzero(regions == region)
        for i in idx:
            other = idx[idx != i]
            for t in range(12):
                finite = growth[other, t]
                finite = finite[np.isfinite(finite)]
                if len(finite) >= min_peers:
                    peer[i, t] = np.median(finite)
    return growth - peer


def sustained_scores(signal, drift=0.01):
    center = np.nanmedian(signal[:, :3], axis=1, keepdims=True)
    centered = signal - center
    settings = load_settings()
    settings["cusum_drift"] = drift
    return (
        centered,
        causal_scores(centered, "cusum", settings),
        causal_scores(centered, "ewma", settings),
    )


def synthetic_summary(shape, method, score, threshold):
    """Only injected changes have a postchange detection window."""
    active = score[:, 3:] > threshold
    window_alert = float((score[:, 5:7] > threshold).any(axis=1).mean())
    return {
        "shape": shape,
        "method": method,
        "active_per100": float(100 * active.mean()),
        "any_alert": float(active.any(axis=1).mean()),
        "reference_window_any_alert": window_alert,
        "detected_by_second_postchange": window_alert
        if shape in {"spike", "sustained"}
        else None,
    }


def run():
    cfg = json.loads((ROOT / "configs/yoy_shift_review.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    raw = raw[raw.category.eq(cfg["category"])]
    months = [str(p) for p in pd.period_range("2023-01", "2024-12", freq="M")]
    pivot = (
        raw.pivot(index="territory_id", columns="date", values="value")
        .reindex(columns=months)
        .sort_index()
    )
    ids = pivot.index.to_numpy()
    geo = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    geo = geo[geo.year.eq(2024)].set_index("territory_id")
    regions = geo.reindex(ids).region_code.to_numpy(float)
    values = pivot.to_numpy(float).copy()
    values[values <= 0] = np.nan
    signal = regional_growth(values, regions, cfg["min_region_peers"])
    signal[~np.isfinite(signal[:, :3]).all(axis=1), :] = np.nan
    centered, cusum, ewma = sustained_scores(signal, cfg["cusum_drift"])
    thresholds = {
        m: float(np.nanquantile(score[:, :3], cfg["threshold_quantile"]))
        for m, score in [("cusum", cusum), ("ewma", ewma)]
    }
    rows = []
    burdens = []
    for method, score in [("cusum", cusum), ("ewma", ewma)]:
        active = np.isfinite(score) & (score > thresholds[method])
        observed = np.isfinite(score[:, 3:])
        burdens.append(
            {
                "method": method,
                "threshold": thresholds[method],
                "monitoring_months": int(observed.sum()),
                "active_months": int(active[:, 3:].sum()),
                "active_per100": 100 * active[:, 3:].sum() / observed.sum(),
                "municipalities_with_alert": int(active[:, 3:].any(axis=1).sum()),
                "real_false_alarm_rate": None,
            }
        )
        for i, tid in enumerate(ids):
            for t, target in enumerate(months[12:]):
                rows.append(
                    {
                        "territory_id": int(tid),
                        "region": regions[i],
                        "target": target,
                        "phase": "calibration" if t < 3 else "monitoring",
                        "method": method,
                        "relative_yoy_log": signal[i, t],
                        "centered_signal": centered[i, t],
                        "score": score[i, t],
                        "threshold": thresholds[method],
                        "active_alert": bool(active[i, t]),
                        "observed": bool(np.isfinite(score[i, t])),
                    }
                )
    panel = pd.DataFrame(rows)
    panel.to_parquet(OUT / "detector_panel.parquet", index=False)
    burden = pd.DataFrame(burdens)
    burden.to_csv(OUT / "monitoring_burden.csv", index=False)
    events = registry_events()
    flood = {
        e["territory_id"]
        for e in events
        if "flood" in e["episode_id"] or "orsk" in e["episode_id"]
    }
    reforms = geo[
        geo.municipal_district_name_short.isin(["Киевский", "Кокошкино"])
    ].index.tolist()
    cases = panel[panel.territory_id.isin(flood | set(reforms))]
    cases.to_csv(OUT / "cases.csv", index=False)
    # Synthetic shapes are context under the same empirical thresholds, not a new tuning loop.
    rng = np.random.default_rng(cfg["seed"])
    base = rng.normal(0, cfg["synthetic_sigma"], (cfg["synthetic_n"], 12))
    synthetic = []
    for shape in ["null", "spike", "sustained", "seasonal"]:
        x = base.copy()
        if shape == "spike":
            x[:, 5] += 0.12
        if shape == "sustained":
            x[:, 5:] += 0.12
        if shape == "seasonal":
            x += 0.12 * np.sin(2 * np.pi * np.arange(12) / 12)[None, :]
        _, c, e = sustained_scores(x, cfg["cusum_drift"])
        for method, score in [("cusum", c), ("ewma", e)]:
            synthetic.append(
                synthetic_summary(shape, method, score, thresholds[method])
            )
    pd.DataFrame(synthetic).to_csv(OUT / "synthetic_context.csv", index=False)
    audit = {
        "status": "complete",
        "calibration_period": "2024-01..2024-03",
        "monitoring_period": "2024-04..2024-12",
        "pre2024_yoy_available": False,
        "calibration_independent": False,
        "calibration_description": "retrospective monitoring-budget calibration after cases were previously seen; quantile computed on entire panel without named-case fitting",
        "thresholds": thresholds,
        "regions": int(pd.Series(regions).nunique()),
        "eligible_municipalities": int(np.isfinite(signal).any(axis=1).sum()),
        "absence_label": "unlabeled",
        "real_false_alarm_rate": None,
        "real_shock_prediction_proven": False,
        "reform_ids": reforms,
        "flood_ids": sorted(flood),
        "geography": "2024 snapshot, IDs preserved; no equivalence across reforms assumed",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for tid in [1673, 1665, *reforms]:
        g = cases[cases.territory_id.eq(tid) & cases.method.eq("cusum")]
        ax.plot(g.target, g.centered_signal, label=str(tid))
    ax.axvspan(-0.5, 2.5, alpha=0.1, color="gray")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Region-relative centered YoY log")
    fig.tight_layout()
    fig.savefig(OUT / "case_signals.png")
    plt.close(fig)
    (OUT / "REPORT.md").write_text(
        "# Устойчивые изменения темпа год к году\n\n"
        + "```csv\n"
        + burden.to_csv(index=False)
        + "```"
        + "\n\nСигнал: логарифм расходов 2024/2023 минус медиана остальных МО того же региона в том же месяце. До 2024 года годовая база отсутствует. Центр по январю–марту 2024, CUSUM с фиксированным drift 0.01 и EWMA; 99-й процентиль калибровочных score всей панели фиксируется перед апрелем. МО без трёх калибровочных наблюдений не оцениваются. Будущие месяцы не участвуют в пороге/центре/истории состояния.\n\nЭто ретроспективная бюджетная калибровка: рассматриваемые случаи уже были известны команде; независимый замороженный эксперимент не заявляется. Синтетические null/spike/sustained/seasonal представлены при тех же порогах как контекст. Метрика обнаружения после изменения определена только для spike и sustained; null и сезонный контроль не имеют введённой границы. Для них сохраняется отдельная доля тревог в контрольном окне. Эмпирическая нагрузка на неразмеченные МО не является реальным FAR.\n\nПаводки означают событийную экспозицию, а не гарантированный разрыв расходов. Киевский/Кокошкино сохраняют исходные ID; юридическая реформа 2024 года, дата публикации справочника и скачок границ/данных — разные метки. Ни причинность, ни равенство территории до/после реформы не предполагаются.\n"
    )
    return audit


if __name__ == "__main__":
    print(run())
