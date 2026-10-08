"""Russian scientific figures for the short-history and news-availability studies."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sberindex.paths import ROOT


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "sber-final-review",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    growth = ROOT / "reports/growth_bridge_review"
    scores = pd.read_csv(growth / "summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    positions = np.arange(4)
    for offset, method, label, color in [
        (-0.25, "seasonal_pooled", "Прежний сезонный", "#a5b4bd"),
        (0.0, "prophet_pooled_profile", "Prophet + профиль", "#386cb0"),
        (0.25, "hierarchy05_growth_bridge", "Региональный профиль + рост", "#087e65"),
    ]:
        values = (
            scores[scores.model == method]
            .set_index("horizon")
            .reindex([1, 3, 6, 12])
            .MAE
        )
        # Baseline spelling is checked against the saved complete comparison.
        if values.isna().any():
            raise ValueError(f"Missing forecast comparison for {method}")
        bars = ax.bar(positions + offset, values, width=0.24, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 45,
                f"{value:.0f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(
        positions,
        [
            "1 месяц\n12 дат",
            "3 месяца\n10 дат",
            "6 месяцев\n7 дат",
            "12 месяцев\n1 дата",
        ],
    )
    ax.set_ylim(0, 5500)
    ax.set_ylabel("MAE, руб. среднего расхода жителя")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_title(
        "Полная когорта: один прогноз на четыре горизонта\nОдинаковые пары; архив 2024 повторно использован",
        fontsize=13,
    )
    fig.savefig(growth / "comparison.png", dpi=180)
    fig.savefig(growth / "comparison.svg", metadata={"Date": None})
    plt.close(fig)
    out = ROOT / "reports/short_history_review"
    table = pd.read_csv(out / "comparison.csv")
    table = table[
        (table.split == "evaluation") & (table["shape"] == "all")
    ].sort_values("onset_f1", ascending=True)
    names = table.method.tolist()
    colors = [
        "#c67d32"
        if m in ("pelt", "kernelcpd")
        else "#087e65"
        if m == "ewma"
        else "#6a8fa8"
        for m in names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].barh(names, table.onset_f1, color=colors)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("F1 начала шока, ±1 месяц")
    for i, v in enumerate(table.onset_f1):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].barh(names, table.null_false_alarms_per100_mo_month, color=colors)
    axes[1].axvline(0.2, color="#b54646", linestyle="--", label="Бюджет отбора 0,2")
    axes[1].set_xlabel("Ложные тревоги / 100 null МО-месяцев")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Год истории: EWMA выбран на отдельном отборе\nНовые симуляции оценки; оранжевый — офлайн методы",
        fontsize=13,
    )
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.svg", metadata={"Date": None})
    plt.close(fig)
    out = ROOT / "reports/news_alert_lead"
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    dates = pd.to_datetime(
        [
            "2024-04-05",
            "2024-04-07",
            "2024-04-30",
            "2024-05-01",
            "2024-06-01",
            "2024-07-01",
        ]
    )
    ax.plot([dates[0], dates[-1]], [0, 0], color="#bbc6ca")
    marks = [
        ("Прорыв\n5 апреля", dates[0], 0, "#c67d32"),
        ("Новость доступна\n7 апреля (сценарий)", dates[1], 0.5, "#087e65"),
        ("Конец апреля", dates[2], 0, "#738695"),
        ("Выпуск: лаг 0\n24 дня после новости", dates[3], 0.5, "#386cb0"),
        ("Лаг 1\n55 дней", dates[4], 0.5, "#386cb0"),
        ("Лаг 2\n85 дней", dates[5], 0.5, "#386cb0"),
    ]
    for label, date, y, color in marks:
        ax.plot([date, date], [0, y], color=color)
        ax.scatter(date, y, s=45, color=color)
        ax.annotate(
            label,
            (date, y),
            xytext=(0, 12 if y else -25),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(-0.7, 1.4)
    ax.set_yticks([])
    ax.set_xticks(
        pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"]),
        ["апрель", "май", "июнь", "июль"],
    )
    ax.set_title(
        "Орск: новость раньше месячных данных\nПоследующей тревоги EWMA в доступном окне нет",
        fontsize=13,
    )
    ax.text(
        0.5,
        0.06,
        "24 / 55 / 85 дней — доступность информации, а не доказанный прогноз шока",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.savefig(out / "availability.png", dpi=180)
    fig.savefig(out / "availability.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
