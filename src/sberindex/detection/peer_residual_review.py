"""Frozen detector comparison on own and leave-one-out regional residuals."""

import hashlib
import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.detection.asof_detector_audit import causal_scores

OUT = ROOT / "reports/peer_residual_review"
CONFIG = ROOT / "configs/peer_residual_review.json"


def regional_peer_residuals(frame, min_peers=5):
    """Each row represents a unique focal ID/month; peers exclude that ID."""
    if frame.duplicated(["territory_id", "target"]).any():
        raise ValueError("duplicate focal municipality/month")
    out = frame.copy()
    out["peer_count"] = 0
    out["peer_median_log"] = np.nan
    for _, group in out.dropna(subset=["region_code"]).groupby(
        ["region_code", "target"]
    ):
        finite = group[np.isfinite(group.own_residual_log)]
        for idx, row in group.iterrows():
            values = finite.loc[
                finite.territory_id != row.territory_id, "own_residual_log"
            ]
            out.loc[idx, "peer_count"] = len(values)
            if len(values) >= min_peers:
                out.loc[idx, "peer_median_log"] = float(values.median())
    out["peer_residual_log"] = out.own_residual_log - out.peer_median_log
    return out


def detector_scores(signal, method, parameters, settings):
    signal = np.asarray(signal, dtype=float)
    if method == "cusum_spike":
        return np.maximum(
            causal_scores(signal, "cusum", settings) / parameters["cusum"],
            causal_scores(signal, "spike", settings) / parameters["spike"],
        )
    return causal_scores(signal, method, settings)


def alert_episodes(active):
    active = np.asarray(active, bool)
    previous = np.concatenate(
        [np.zeros((len(active), 1), bool), active[:, :-1]], axis=1
    )
    return active & ~previous


def release_date(target, lag):
    return (pd.Period(target, freq="M") + 1 + lag).to_timestamp()


def run():
    cfg = json.loads(CONFIG.read_text())
    audit_source = json.loads(
        (ROOT / "reports/short_history_review/audit.json").read_text()
    )
    parameters = audit_source["parameters"]
    settings = json.loads((ROOT / "configs/detectors.json").read_text())
    if audit_source["choice"]["selected_online_method"] != "ewma":
        raise ValueError("Expected frozen EWMA selection")
    pred = pd.read_parquet(ROOT / "reports/growth_bridge_review/predictions.parquet")
    pred = pred[(pred.model == cfg["model"]) & (pred.horizon == cfg["horizon"])].copy()
    geo = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    geo = geo[geo.year == cfg["geography_snapshot_year"]][
        ["territory_id", "region_code"]
    ]
    if geo.territory_id.duplicated().any():
        raise ValueError("Ambiguous geography snapshot")
    pred = pred.merge(geo, on="territory_id", how="left", validate="many_to_one")
    valid = (pred.actual > 0) & (pred.predicted > 0)
    pred["own_residual_log"] = np.nan
    pred.loc[valid, "own_residual_log"] = np.log(
        pred.loc[valid, "actual"] / pred.loc[valid, "predicted"]
    )
    panel = regional_peer_residuals(pred, cfg["min_peers"])
    ids = sorted(panel.territory_id.unique())
    months = [
        str(p)
        for p in pd.period_range(panel.target.min(), panel.target.max(), freq="M")
    ]
    details, burdens = [], []
    for signal_name in ["own", "peer"]:
        signal = (
            panel.pivot(
                index="territory_id",
                columns="target",
                values=signal_name + "_residual_log",
            )
            .reindex(index=ids, columns=months)
            .to_numpy(float)
        )
        observed = np.isfinite(signal)
        for method in cfg["methods"]:
            scores = detector_scores(signal, method, parameters, settings)
            active = observed & (scores > parameters[method])
            episode = alert_episodes(active)
            burdens.append(
                dict(
                    signal=signal_name,
                    method=method,
                    municipalities=len(ids),
                    observed_months=int(observed.sum()),
                    unavailable_months=int((~observed).sum()),
                    active_alert_months=int(active.sum()),
                    alert_episodes=int(episode.sum()),
                    municipalities_with_alert=int(active.any(axis=1).sum()),
                    active_per100_observed_months=100 * active.sum() / observed.sum(),
                    episodes_per100_observed_months=100
                    * episode.sum()
                    / observed.sum(),
                )
            )
            for i, tid in enumerate(ids):
                for j, month in enumerate(months):
                    details.append(
                        dict(
                            territory_id=tid,
                            target=month,
                            signal=signal_name,
                            method=method,
                            residual_log=signal[i, j],
                            score=scores[i, j],
                            threshold=parameters[method],
                            observed=observed[i, j],
                            active_alert=active[i, j],
                            alert_episode=episode[i, j],
                        )
                    )
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "residual_panel.parquet", index=False)
    detail = pd.DataFrame(details)
    detail.to_parquet(OUT / "detector_panel.parquet", index=False)
    burden = pd.DataFrame(burdens)
    burden.to_csv(OUT / "monitoring_burden.csv", index=False)
    yoy = (
        panel[panel.territory_id.isin(cfg["case_ids"])]
        .pivot(index="target", columns="territory_id", values="yoy_actual")
        .reindex(months)
    )
    yoy.columns = [
        "orenburg_yoy_pct" if i == 1665 else "orsk_yoy_pct" for i in yoy.columns
    ]
    yoy["orsk_minus_orenburg_pp"] = yoy.orsk_yoy_pct - yoy.orenburg_yoy_pct
    yoy.to_csv(OUT / "case_yoy.csv")
    case = detail[detail.territory_id.isin(cfg["case_ids"])]
    case.to_csv(OUT / "case_detector_months.csv", index=False)
    lead_rows = []
    news = pd.Timestamp(cfg["news_available_scenario"])
    for (tid, signal_name, method), group in case.groupby(
        ["territory_id", "signal", "method"]
    ):
        for lag in cfg["release_lags_months"]:
            eligible = group[
                (group.target >= cfg["event_onset"][:7]) & group.alert_episode
            ].copy()
            eligible["release"] = eligible.target.map(lambda x: release_date(x, lag))
            eligible = eligible[eligible.release > news].sort_values("release")
            end = release_date(group[group.observed].target.max(), lag)
            first = eligible.iloc[0] if len(eligible) else None
            lead_rows.append(
                dict(
                    territory_id=tid,
                    signal=signal_name,
                    method=method,
                    lag_months=lag,
                    news_published=cfg["news_published"],
                    news_available_scenario=cfg["news_available_scenario"],
                    event_onset=cfg["event_onset"],
                    first_postnews_alarm_month=first.target
                    if first is not None
                    else None,
                    alarm_release=first.release.date().isoformat()
                    if first is not None
                    else None,
                    publication_to_alarm_days=(
                        first.release - pd.Timestamp(cfg["news_published"])
                    ).days
                    if first is not None
                    else None,
                    availability_to_alarm_days=(first.release - news).days
                    if first is not None
                    else None,
                    right_censored=first is None,
                    observation_end_release=end.date().isoformat(),
                )
            )
    leads = pd.DataFrame(lead_rows)
    leads.to_csv(OUT / "news_lead_sensitivity.csv", index=False)
    plot(case, yoy)
    paths = [
        "configs/peer_residual_review.json",
        "docs/protocols/PEER_RESIDUAL_REVIEW.md",
        "src/sberindex/detection/peer_residual_review.py",
        "reports/growth_bridge_review/predictions.parquet",
        "reports/short_history_review/audit.json",
        "src/sberindex/detection/short_history_review.py",
        "src/sberindex/detection/asof_detector_audit.py",
        "configs/detectors.json",
        "results/municipal_lookup.csv",
        "data/external/regional_news_review/registry.csv",
        "data/external/regional_news_review/source_1.web.json",
        "data/external/regional_news_review/source_25.web.json",
        "reports/peer_residual_review/payment_context_source_metadata.json",
    ]
    audit = dict(
        status="complete",
        config=cfg,
        parameters={m: parameters[m] for m in cfg["methods"]},
        settings=settings,
        thresholds_tuned=False,
        selected_method="ewma",
        real_false_alarm_guarantee=False,
        geography_historical_vintage=False,
        monthly_release_dates_verified=False,
        news_available_assumed=True,
        missing_state_policy="reset on unavailable month, matching existing causal scorer",
        input_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
        },
        output_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.iterdir()
            if p.is_file() and p.name not in ["audit.json", "REPORT.md"]
        },
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    report(burden, yoy, leads)
    print(burden.to_string(index=False))
    print(yoy.to_string())
    print(
        leads[(leads.territory_id == 1673) & (leads.method == "ewma")].to_string(
            index=False
        )
    )


def plot(case, yoy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, signal_name in zip(axes[:2], ["own", "peer"]):
        g = case[
            (case.territory_id == 1673)
            & (case.method == "ewma")
            & (case.signal == signal_name)
        ].sort_values("target")
        dates = pd.to_datetime(g.target)
        ax.plot(dates, g.residual_log, marker="o", label=f"{signal_name} log residual")
        ax.plot(dates, g.score, label="absolute EWMA score")
        ax.axhline(
            g.threshold.iloc[0],
            color="red",
            linestyle="--",
            label="threshold for EWMA score",
        )
        a = g.active_alert
        ax.scatter(
            dates[a], g.score[a], marker="x", color="red", s=70, label="active alert"
        )
        ax.axvline(pd.Timestamp("2024-04-05"), color="gray", linestyle=":")
        ax.set_title(f"Orsk {signal_name}: no frozen EWMA alert in 2024", fontsize=10)
        ax.set_ylabel("Log units")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)
    dates = pd.to_datetime(yoy.index)
    axes[2].plot(dates, yoy.orsk_yoy_pct, marker="o", label="Orsk YoY (%)")
    axes[2].plot(dates, yoy.orenburg_yoy_pct, marker="o", label="Orenburg YoY (%)")
    axes[2].plot(
        dates, yoy.orsk_minus_orenburg_pp, marker="s", label="Orsk minus Orenburg (pp)"
    )
    axes[2].axhline(0, color="gray", linewidth=0.7)
    axes[2].axvline(pd.Timestamp("2024-04-05"), color="gray", linestyle=":")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    axes[2].set_ylabel("% / percentage points")
    fig.suptitle(
        "Orsk: frozen detector and actual YoY comparison\nRetrospective monthly data; descriptive association, no causal claim",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "orsk_case.png", dpi=180)
    fig.savefig(OUT / "orsk_case.svg")
    plt.close(fig)



def markdown(frame, index=False):
    if index:
        frame = frame.reset_index()
    columns = [str(c) for c in frame.columns]
    def cell(value):
        return "—" if pd.isna(value) else str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)

def report(burden, yoy, leads):
    selected = leads[(leads.method == "ewma") & (leads.lag_months == 0)]
    lines = [
        "# Peer residual detector review",
        "",
        "Forecast: hierarchy05_growth_bridge, h=1. Thresholds, EWMA alpha 0.45 and CUSUM drift 0.025 remain frozen. EWMA was selected on synthetic data; this retrospective real-data review is not independent temporal validation.",
        "",
        "## Entire-panel monitoring burden",
        "",
        markdown(burden),
        "",
        "Burden counts are not FAR: no real spending-shift ground truth labels are available. Denominators count finite residual months only. New episodes are first active month after an inactive/unavailable month. Same-month regional residuals exclude the focal ID and require five peers. Missing months reset online state. Geography uses a 2024 snapshot, not historical geography vintages.",
        "",
        "## Фактический годовой рост расходов, %; разрыв, п.п.",
        "",
        markdown(yoy, index=True),
        "",
        "May–July Orsk minus Orenburg gaps are 9.19, 6.23 and 6.11 pp (unweighted mean 7.18 pp). Thus approximately 7 pp describes that chosen three-month window; it is not a persistent full-year post-flood gap. Orenburg also experienced flooding and is not an untreated control. These data cannot establish flood causation. All four frozen methods produce no alert in either city in 2024. Orsk maximum own/peer EWMA scores are 0.03873/0.03449, below the 0.07428 threshold.",
        "",
        "Выплаты и восстановление остаются гипотезами объяснения, а не доказанными причинами. Муниципальный МФЦ Орска в публикации 06.05.2024 сообщил о доступности с 02.05.2024 поддержки на наём жилья: https://мфц-орск.рф/2024/05/06/stala-dostupna-novaya-mera-podderzhki-predostavlenie-denezhnoj-vyplaty-na-naem-zhilogo-pomeshheniya-v-svyazi-s-chs/ . Это подтверждает объявление меры, а не фактические даты выплат, суммы получателям или причинный эффект на расходы. Снимок сохранён в data/external/peer_residual_review/mfc_support_page.json. Payments and recovery spending are plausible contextual hypotheses for higher observed YoY, not established causes. Primary-site research did not verify actual payment start dates or municipality-specific amounts. The official regional flood portal indexes recovery/support news in May (https://pavodok.orb.ru/), but its full page returned HTTP 403; federal payment-page retrieval timed out (https://government.ru/docs/52698/). No claim that payments began in May, and no payment-to-consumption attribution, is supported. See payment_context_source_metadata.json for retrieval limitations.",
        "",
        "## Доступность новости и первая последующая тревога",
        "",
        markdown(selected),
        "",
        "Publication: 6 April; event onset: 5 April; news available: 7 April under publication-plus-one-day scenario. Source: https://56.mchs.gov.ru/deyatelnost/press-centr/novosti/5249015; independent onset evidence: https://21.mchs.gov.ru/deyatelnost/press-centr/vse_novosti/5260528. Alarm dates use assumed following-month-first-day releases; lag 1/2 adds calendar months. Both publication-to-alert and assumed availability-to-alert days appear in the sensitivity CSV. No alarm is right-censored, with no invented lead. First subsequent means an episode from April onward, not a forced May detection.",
        "",
        "Monthly residuals and contemporaneous regional peers only become observable on publication of monthly spending. A news lead, when observed, is a lead over statistical detection, not prediction before the flood. Actual historical release and ingestion timestamps remain unverified.",
        "",
        "![Orsk case](orsk_case.png)",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run()
