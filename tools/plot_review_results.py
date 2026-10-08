"""Scientific plots for the full-cohort comparison and fixed direct-h3 trial."""

from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.hashsalt": "sber-review-results",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180)
    fig.savefig(directory / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    out = ROOT / "reports/full_cohort_review"
    table = pd.read_csv(out / "summary.csv")
    variants = [
        "prophet_disabled",
        "prophet_yearly3",
        "prophet_pooled_profile",
        "seasonal_pooled",
        "seasonal_naive",
    ]
    labels = [
        "Prophet\nбез годовой",
        "Prophet\nFourier3",
        "Prophet\nс профилем",
        "Сезонный\nпрофиль",
        "Сезонный\nnaive",
    ]
    colors = ["#718096", "#7057ad", "#386cb0", "#0b6b55", "#c48b47"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, h in zip(axes, [1, 3, 6, 12]):
        group = table[table.horizon == h].set_index("model").loc[variants]
        ax.bar(np.arange(len(variants)), group.MAE, color=colors, width=0.72)
        for i, value in enumerate(group.MAE):
            ax.text(i, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(
            np.arange(len(variants)), labels, rotation=28, ha="right", fontsize=9
        )
        ax.set_title(
            f"h{h}: {int(group.dates.iloc[0])} дат · {int(group.observations.iloc[0])} пар",
            fontsize=11,
        )
        ax.set_ylabel("MAE, руб. / жителя")
        ax.set_ylim(0, float(group.MAE.max()) * 1.2)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Все МО, пригодные по данным 2023: фиксированные варианты Prophet и сезонные ориентиры",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "2075 ID до наблюдения 2024; одинаковые пары моделей внутри горизонта. h12: одна дата. Исследованный архив, не независимый тест.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    save(fig, out, "comparison")

    out = ROOT / "reports/direct_horizon_review"
    table = pd.read_csv(out / "monthly.csv")
    pair = table[
        (table.horizon == 3) & table.model.isin(["seasonal_pooled", "direct_lags"])
    ].pivot(index="target", columns="model", values="MAE")
    assert pair.notna().all().all() and len(pair) == 10
    labels = ["мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for model, label, color in [
        ("seasonal_pooled", "Сезонный профиль", "#386cb0"),
        ("direct_lags", "Прямой HGB h3", "#0b6b55"),
    ]:
        axes[0].plot(
            np.arange(10), pair[model], marker="o", label=label, color=color, ms=5
        )
    axes[0].set_ylabel("MAE по МО, руб. / жителя")
    axes[0].legend(frameon=False)
    gains = pair.seasonal_pooled - pair.direct_lags
    axes[1].bar(np.arange(10), gains, color=np.where(gains >= 0, "#0b6b55", "#b96547"))
    axes[1].axhline(0, color="#718096", linewidth=0.8)
    axes[1].set_ylabel("Выигрыш HGB, руб.")
    axes[1].set_xticks(np.arange(10), labels)
    axes[1].set_xlabel("Целевой месяц 2024")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Прямой трёхмесячный прогноз: эффект по каждой дате", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Одни 251 МО, 2510 пар. Rolling обучение только на наблюдённых целях; 2024 уже исследован. Основной метод не менялся.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])
    save(fig, out, "h3_dates")


if __name__ == "__main__":
    main()
