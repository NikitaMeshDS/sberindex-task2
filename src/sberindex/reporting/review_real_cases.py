"""Observable major-event cases and explicitly missing municipal series."""

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
from sberindex.forecasting.full_cohort_review import eligible_ids

OUT = ROOT / "reports/review_real_cases"


def seasonal_errors(values, profile):
    values = np.asarray(values, float)
    profile = np.asarray(profile, float)
    actual = values[12:24]
    origins = values[11:23]
    prediction = origins * profile[np.arange(12)] / profile[(np.arange(12) - 1) % 12]
    prediction[~np.isfinite(origins) | (origins <= 0)] = np.nan
    return actual, prediction


def main():
    config = json.loads((ROOT / "configs/review_real_cases.json").read_text())
    raw = pd.read_parquet(ROOT / "data/consumption.parquet")
    months = pd.period_range("2023-01", "2024-12", freq="M").astype(str)
    panel = (
        raw[raw.category == "Все категории"]
        .pivot(index="territory_id", columns="date", values="value")
        .reindex(columns=months)
    )
    train = panel.loc[eligible_ids(panel)].iloc[:, :12].sum().to_numpy(float)
    profile = train / train.mean()
    lookup = pd.read_csv(ROOT / "results/municipal_lookup.csv")
    geo = lookup[lookup.year == config["geography_year"]]
    registry = pd.read_csv(ROOT / "data/external/regional_news_review/registry.csv")
    rows, coverage = [], []
    for case in config["cases"]:
        ids = (
            [case["territory_id"]]
            if case["territory_id"] is not None
            else geo.loc[
                geo.region_code == case["region_code"], "territory_id"
            ].tolist()
        )
        present = [city for city in ids if city in panel.index]
        source = registry[registry.episode_id == case["episode"]]
        assert len(source) and case["publication"] in set(source.published_date)
        coverage.append(
            dict(
                case=case["name"],
                episode=case["episode"],
                mapped_ids=len(ids),
                source_present_ids=len(present),
                status="observed_series" if present else "no_comparable_expense_series",
                publication=case["publication"],
                official_source=source.loc[
                    source.published_date == case["publication"], "source_url"
                ].iloc[0],
                spending_shift_ground_truth=False,
            )
        )
        for city in present:
            actual, prediction = seasonal_errors(
                panel.loc[city].to_numpy(float), profile
            )
            for target, a, p in zip(months[12:], actual, prediction):
                rows.append(
                    dict(
                        case=case["name"],
                        territory_id=city,
                        target=target,
                        actual=a,
                        seasonal_prediction=p,
                        residual_pct=100 * (a / p - 1),
                    )
                )
    OUT.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "observed_series.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "sber-review-real-cases",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    dates = pd.to_datetime(months[12:])
    for column, name in enumerate(["Орск", "Оренбург"]):
        part = frame[frame.case == name]
        axes[0, column].plot(
            dates,
            part.actual / 1000,
            color="#0b6b55",
            marker="o",
            ms=4,
            label="Наблюдение",
        )
        axes[0, column].plot(
            dates,
            part.seasonal_prediction / 1000,
            color="#386cb0",
            linestyle="--",
            label="h1: профиль 2023",
        )
        axes[0, column].set_title(name)
        axes[0, column].set_ylabel("тыс. руб. / жителя")
        axes[0, column].legend(frameon=False, fontsize=9)
        axes[1, column].bar(dates, part.residual_pct, width=19, color="#386cb0")
        axes[1, column].axhline(0, color="#718096", linewidth=0.8)
        axes[1, column].set_ylabel("Отклонение от ориентира, %")
        for ax in axes[:, column]:
            ax.axvline(
                pd.Timestamp("2024-04-01"), color="#c06934", linewidth=1, linestyle=":"
            )
            ax.grid(axis="y", alpha=0.2)
        axes[0, column].text(
            0.04,
            0.92,
            "Апрель: публикации о паводке",
            transform=axes[0, column].transAxes,
            fontsize=9,
            color="#984a20",
        )
    ticks = pd.to_datetime(
        ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2024-12-01"]
    )
    for ax in axes[1]:
        ax.set_xticks(ticks, ["янв", "апр", "июл", "окт", "дек"])
        ax.set_xlabel("2024")
    fig.suptitle("Новости и месячные расходы: наблюдаемые кейсы", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Паводок не служит меткой расходного сдвига. Отклонение от сезонного ориентира не является причинным эффектом.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT / "cases.png", dpi=180)
    fig.savefig(OUT / "cases.svg", metadata={"Date": None})
    plt.close(fig)
    april = frame[frame.target == "2024-04"]
    body = "# Крупные события и доступность расходных кейсов\n\n"
    body += "Используются только официально сопоставленные ID. Снимок географии ретроспективный. Дата сообщения не подменяет начало события, а внешнее событие не создаёт метку изменения расходов.\n\n"
    body += "| Кейс | Сопоставленных ID | ID с расходами | Источник |\n|---|---:|---:|---|\n"
    for row in coverage:
        body += f"| {row['case']} | {row['mapped_ids']} | {row['source_present_ids']} | [Публикация {row['publication']}]({row['official_source']}) |\n"
    body += "\n![Наблюдаемые кейсы](cases.png)\n\n"
    for row in april.itertuples():
        body += f"{row.case}: апрельский расход {row.actual:.0f} руб., ориентир {row.seasonal_prediction:.0f} руб., отклонение {row.residual_pct:+.2f}%. "
    body += "\n\nОриентир использует только последний известный уровень и замороженный общемуниципальный профиль2023. Это описательная проверка, без оценки причинного ущерба, статистической значимости или гарантии обнаружения. Monthly aggregation может скрыть локальные/краткие последствия и изменение состава операций.\n\n"
    body += "Анапа и Белгород: выбранные официальные ID отсутствуют в расходной панели; МО Курской области по справочнику2024 тоже не имеют сопоставимых рядов. Другой ID или прокси региона не подставляется без документированного соответствия. Нельзя рассчитывать precision/recall/F1 расходных изменений на этих новостных событиях. Для Анапы также нет последующих2025месяцев для проверки устойчивого изменения.\n\n"
    body += "Новости Орска6апреля об уже произошедшем прорыве5апреля не являются предупреждением до шока. Под условием публикация+1день сообщение доступно7апреля, через2дня после начала. Фактическая дата выпуска расходов неизвестна: leadtime к ней не вычисляется.\n\n"
    body += "Воспроизведение: `PYTHONPATH=src python -m sberindex.reporting.review_real_cases`. Числа сохранены в observed_series.csv, отсутствие рядов в coverage.csv.\n"
    (OUT / "REPORT.md").write_text(body)
    inputs = [
        "configs/review_real_cases.json",
        "src/sberindex/reporting/review_real_cases.py",
        "src/sberindex/forecasting/full_cohort_review.py",
        "data/consumption.parquet",
        "results/municipal_lookup.csv",
        "data/external/regional_news_review/registry.csv",
    ]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit = dict(
        status="passed",
        independent_time_validation=False,
        causal_effect_estimated=False,
        spending_shift_ground_truth=False,
        input_sha256={name: sha(ROOT / name) for name in inputs},
        output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.name != "audit.json"},
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        pd.DataFrame(coverage)[["case", "source_present_ids", "status"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
