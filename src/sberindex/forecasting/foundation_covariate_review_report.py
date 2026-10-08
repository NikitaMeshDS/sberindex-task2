"""Render isolated foundation appendix from measured, hash-checked predictions."""

import json
import numpy as np
import pandas as pd
from sberindex.paths import ROOT
from sberindex.forecasting.foundation_covariate_review import OUT, KEYS, sha, verify


def render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checks = verify()
    audit = json.loads((OUT / "audit.json").read_text())
    summary = pd.read_csv(OUT / "summary.csv")
    monthly = pd.read_csv(OUT / "monthly.csv")
    runtime = pd.read_csv(OUT / "runtime.csv")
    frame = pd.read_parquet(OUT / "predictions.parquet")
    overall = summary[summary.category.eq("Все категории")].copy()
    model_order = [
        "seasonal_naive",
        "seasonal_pooled",
        "chronos2_univariate",
        "chronos2_profile",
        "chronos2_mo_group",
        "chronos2_mo_group_profile",
        "bolt_base_univariate",
    ]
    labels = [
        "Сезонный naive",
        "Сезонный профиль",
        "C2: один ряд",
        "C2 + профиль",
        "C2: категории МО",
        "C2: группа + профили",
        "Bolt base",
    ]
    colors = [
        "#aeb8c5",
        "#334155",
        "#5776a3",
        "#cc7c41",
        "#438375",
        "#855d93",
        "#bd5065",
    ]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "svg.hashsalt": "sber-foundation-review",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    x = np.arange(4)
    width = 0.11
    for i, (model, label, color) in enumerate(zip(model_order, labels, colors)):
        g = overall[overall.model.eq(model)].set_index("horizon").reindex([1, 3, 6, 12])
        axes[0].bar(x + (i - 3) * width, g.MAE, width, label=label, color=color)
    axes[0].set_xticks(x, ["h1\n12 дат", "h3\n10 дат", "h6\n7 дат", "h12\n1 дата"])
    axes[0].set_ylabel("MAE по датам, руб. / жителя")
    axes[0].set_title("Все категории: одинаковые прогнозные пары")
    axes[0].grid(axis="y", alpha=0.15)
    axes[0].set_axisbelow(True)
    for model, label, color in zip(model_order[2:], labels[2:], colors[2:]):
        g = monthly[
            (monthly.category.eq("Все категории"))
            & monthly.horizon.eq(1)
            & monthly.model.eq(model)
        ].sort_values("target")
        baseline = (
            monthly[
                (monthly.category.eq("Все категории"))
                & monthly.horizon.eq(1)
                & monthly.model.eq("chronos2_univariate")
            ]
            .set_index("target")
            .MAE
        )
        axes[1].plot(
            g.target.str[5:],
            g.MAE - g.target.map(baseline),
            marker="o",
            markersize=3,
            label=label,
            color=color,
        )
    axes[1].axhline(0, color="#334155", linewidth=0.8)
    axes[1].set_title("h1: monthly MAE change versus C2: один ряд")
    axes[1].set_xlabel("Целевой месяц 2024")
    axes[1].set_ylabel("Разница MAE, руб. (минус — улучшение)")
    axes[1].grid(axis="y", alpha=0.15)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle("Календарные профили и совместный прогноз категорий МО", fontsize=15)
    fig.subplots_adjust(bottom=0.21, top=0.83, wspace=0.3)
    fig.savefig(OUT / "comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "comparison.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    # Existing tiny and stronger Prophet references are added only on exact cached overlaps;
    # their reduced calendar coverage is shown separately from the new full rolling experiment.
    sources = {}
    overlap = []
    for path, models in [
        ("results/asof_foundation_predictions.parquet", ["chronos_bolt_tiny"]),
        (
            "reports/prophet_seasonality/predictions.parquet",
            ["prophet_yearly3", "prophet_pooled_profile"],
        ),
    ]:
        cached = pd.read_parquet(ROOT / path)
        sources[path] = sha(ROOT / path)
        for model in models:
            original = cached[cached.model.eq(model)].copy()
            new = frame[
                frame.category.eq("Все категории")
                & frame.model.eq("bolt_base_univariate")
            ].copy()
            keys = [k for k in KEYS if k != "category"]
            pair = new.merge(
                original[keys + ["actual", "predicted"]],
                on=keys,
                suffixes=("_new", "_ref"),
                validate="one_to_one",
            )
            np.testing.assert_array_equal(pair.actual_new, pair.actual_ref)
            for h, g in pair.groupby("horizon"):
                for name, col in [
                    ("bolt_base_univariate", "predicted_new"),
                    (model, "predicted_ref"),
                ]:
                    ae = abs(g.actual_new - g[col])
                    overlap.append(
                        dict(
                            reference=model,
                            horizon=int(h),
                            model=name,
                            MAE=float(ae.groupby(g.target).mean().mean()),
                            dates=int(g.target.nunique()),
                            municipalities=int(g.territory_id.nunique()),
                            observations=len(g),
                        )
                    )
    overlap = pd.DataFrame(overlap)
    overlap.to_csv(OUT / "cached_reference_overlap.csv", index=False)

    def table(rows, columns):
        text = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for r in rows:
            text.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
        return "\n".join(text)

    main = []
    for h in [1, 3, 6, 12]:
        for model in model_order:
            r = overall[(overall.horizon.eq(h)) & overall.model.eq(model)].iloc[0]
            main.append(
                dict(
                    h=h,
                    model=model,
                    dates=int(r.dates),
                    MO=int(r.municipalities),
                    n=int(r.observations),
                    MAE=f"{r.MAE:.3f}",
                    YoY_MAE_pp=f"{r.YoY_MAE_pp:.3f}",
                    skill_profile=f"{r.skill_vs_seasonal_pooled:.3f}",
                )
            )
    comparisons = []
    for h in [1, 3, 6, 12]:
        g = overall[overall.horizon.eq(h)].set_index("model")
        for model, reference in [
            ("chronos2_profile", "chronos2_univariate"),
            ("chronos2_mo_group", "chronos2_univariate"),
            ("chronos2_mo_group_profile", "chronos2_mo_group"),
            ("bolt_base_univariate", "chronos2_univariate"),
        ]:
            delta = g.loc[model, "MAE"] - g.loc[reference, "MAE"]
            fraction = delta / g.loc[reference, "MAE"] * 100
            pairs = monthly[
                monthly.category.eq("Все категории") & monthly.horizon.eq(h)
            ].pivot(index="target", columns="model", values="MAE")
            comparisons.append(
                dict(
                    h=h,
                    contrast=model + " vs " + reference,
                    delta_MAE=f"{delta:.3f}",
                    delta_pct=f"{fraction:.1f}%",
                    lower_error_dates=f"{int((pairs[model] < pairs[reference]).sum())}/{len(pairs)}",
                )
            )
    rt = runtime.groupby("model").seconds.sum()
    total = runtime.seconds.sum()
    counts = frame.groupby(["category", "horizon", "model"]).size()
    lines = [
        "# Foundation: явные будущие ковариаты и группы категорий",
        "",
        "Новые измеренные расчёты на фиксированной когорте 256 ID: четыре варианта Chronos-2 и Chronos-Bolt-base. Параметры и протокол зафиксированы до расчёта ошибок. Нормированные Chronos из прежнего опыта не повторялись. Ниже расходы «Все категории»; полные результаты всех шести категорий сохранены в summary.csv/monthly.csv.",
        "",
        "На расходах «Все категории» собственный профиль снижает MAE Chronos-2 на 11.0/5.8/11.4/5.6% для h1/3/6/12; группы с шестью профилями — на 27.0/15.6/18.8/13.6%. Одна группировка почти не меняет результат и на h6 слегка ухудшает MAE. Сезонный pooled-контроль имеет меньшую MAE, чем каждый foundation-вариант, на всех четырёх горизонтах. Bolt-base хуже нового univariate C2 на всех горизонтах, хотя лучше прежнего tiny на ограниченных точных пересечениях. Эти числа описывают конкретную таблицу, не общий рейтинг моделей. По месяцам знак неоднороден, особенно в групповом варианте; выигрыш агрегата нельзя выдавать за устойчивое улучшение каждой даты.",
        "",
        "Группировка задана явно: шесть категорий одного МО — шесть target-variates одного задания. У каждого МО отдельный group_id; межмуниципального обмена нет. Проверены фактические group_ids установленного датасета: [0×6,1×6], с профилями [0×12,1×12]. cross_learning=False сохраняет эти границы; True объединил бы задания батча. Это измеренный опыт группового внимания внутри МО, а не утверждение, что обычный batch predict_df включает cross-learning.",
        "",
        "Ковариаты: только шесть pooled-профилей 2023, повторяемых по месяцу календаря. В одиночном варианте каждой категории дан её профиль; в групповом — все шесть профилей. Будущие факты расходов, включая будущие категории, не передавались. Расходы на origin входят в исторические targets; параметры модели не дообучались. Ошибки нового опыта не использовались для настройки или выбора варианта.",
        "",
        "Из 256 выбранных МО все имеют полную историю 2023; с origin 2024-01 полный набор наблюдаемых категорий сохраняется у 251. Это правило по прошлому строже одиночной категории и одинаково для всех вариантов. Будущий факт отбирается только при оценивании; отсутствующие цели не восстанавливаются. Dates12/10/7/1 для h1/3/6/12; h12 — единственная дата 2024-12.",
        "",
        "![Сопоставление](comparison.png)",
        "",
        "## Основные одинаковые пары",
        "",
        table(
            main,
            ["h", "model", "dates", "MO", "n", "MAE", "YoY_MAE_pp", "skill_profile"],
        ),
        "",
        "MAE — среднее сначала по МО, затем с одинаковым весом целевых месяцев. skill_profile=1−MAE/MAE сезонного профиля. Положительное значение означает меньшую ошибку. PooledR² и within-MO R² есть в CSV; на h12 within-MO R² неопределён, поскольку у каждого МО один факт.",
        "",
        "## Фиксированные контрасты",
        "",
        table(
            comparisons,
            ["h", "contrast", "delta_MAE", "delta_pct", "lower_error_dates"],
        ),
        "",
        "Отрицательная delta означает меньшую ошибку. По месяцам видно, устойчив ли знак, а не только агрегат. Эти различия описательные; муниципалитеты и overlapping origins не независимы, формальная значимость и победитель не объявляются. Крупнее не гарантирует лучше: Bolt-base отдельно сопоставлен с Chronos-2 и сезонными контролями.",
        "",
        "## Прежние tiny / сильные Prophet на точных пересечениях",
        "",
        "cached_reference_overlap.csv содержит только одинаковые territory/origin/target/horizon и проверенные факты. Прежнийtiny не покрывает все ранние h3/h6; его сокращённые dates нельзя смешивать с основной новой таблицей. Сильные Prophet взяты из отдельного фиксированного опыта; ни одна прежняя прогнозная таблица не заменена.",
        "",
        table(
            [
                dict(
                    reference=r.reference,
                    h=int(r.horizon),
                    model=r.model,
                    dates=int(r.dates),
                    n=int(r.observations),
                    MAE=f"{r.MAE:.3f}",
                )
                for r in overlap.itertuples()
            ],
            ["reference", "h", "model", "dates", "n", "MAE"],
        ),
        "",
        "## Ресурсы и происхождение",
        "",
        f"Final complete run inference+load: {total:.1f}с; полное исполнение с записью/хешированием {audit['seconds']:.1f}с. Peak process RSS {audit['peak_rss_GiB']:.2f}GiB (<8GiB). CPU, 2 torch threads, batch_size96 series(C2),32(Bolt), модели загружались последовательно. Runtime по каждому origin/варианту сохранён. Новых fits:0. Все пять модельных вариантов посчитаны на полной выбранной когорте 256 ID, без уменьшения выборки ради ресурсов.",
        "",
        f"chronos-forecasting {audit['chronos_version']}; torch {audit['torch_version']}. Chronos-2 revision `{audit['model_revisions']['chronos2_revision']}`; Bolt-base revision `{audit['model_revisions']['bolt_revision']}`. В audit.json — SHA256 исходных файлов, checkpoint/config артефактов, установленного pipeline/dataset/preprocess и результатов. frozen_inputs.json фиксирует входы до inference. Само имя ревизии не является доказательством отсутствия training leakage.",
        "",
        "Первый полный проход Chronos-2 остановился перед Bolt из-за неполного snapshot metadata (.gitattributes). После исправления конверсии RSS для переносимости macOS/Linux выполнен полный повтор с теми же параметрами. first_pass содержит прежние прогнозы/тайминги; repeat_verification.json подтверждает точное совпадение первого прохода C2/контролей с завершённым повтором. Указанные выше тайминги относятся к последнему полному запуску; setup/download и предыдущие попытки в них не входят.",
        "",
        "## Проверки и границы вывода",
        "",
        "Три теста утечки и отдельный тест конверсии RSS для macOS/Linux прошли после ожидаемых сбоев отсутствующей реализации. На реальных входах для всех origin проверено: изменение будущих расходов не меняет history eligibility, historical targets или future profile covariates. Профили считаются только по первым12месяцам; тест замены 2024 подтверждает их инвариантность. verify пересчитывает метрики, проверяет полное совпадение модельных пар и исходных actual/year-ago и SHA256 файлов проекта/сохранённых чисел. Хеши checkpoint и установленного API записаны для происхождения inference; проверка сохранённых метрик не требует заново загружать модель. "
        + str(checks),
        "",
        "Архив 2024 уже многократно изучен. [Chronos-2 опубликован20 октября 2025, Bolt 26 ноября 2024](https://github.com/amazon-science/chronos-forecasting#-news). Новые pinned checkpoints и установленноеПО не восстанавливают реально доступные модели 2024; точный training cutoff и возможное пересечение с архивом не доказаны. Это past-only входы в ретроспективном опыте, а не независимая временная валидация или deployable 2024 vintage. Uniform zero lag не моделирует настоящие публикационные задержки/ревизии.256-ID sample не равен полной когорте 2075 ID. h12 имеет одну дату. Главная модель не изменена.",
        "",
        "Официальный [pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py) подтверждает multivariate target и future_covariates. [Ответ maintainer о группировке](https://github.com/amazon-science/chronos-forecasting/discussions/464), [Bolt-base card](https://huggingface.co/amazon/chronos-bolt-base). Источники API проверены до inference; точная установленная реализация защищена хешами.",
        "",
        "Воспроизведение: `PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_covariate_review`; проверка без загрузки моделей: та же команда с `--verify`; отчёт: `PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_covariate_review_report`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    report_audit = dict(
        checks=checks,
        extra_source_sha256=sources,
        report_source_sha256=sha(
            ROOT / "src/sberindex/forecasting/foundation_covariate_review_report.py"
        ),
        outputs={
            p: sha(OUT / p)
            for p in [
                "REPORT.md",
                "comparison.png",
                "comparison.svg",
                "cached_reference_overlap.csv",
                "repeat_verification.json",
            ]
        },
    )
    (OUT / "report_audit.json").write_text(
        json.dumps(report_audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        overall[
            ["horizon", "model", "MAE", "dates", "municipalities", "observations"]
        ].to_string(index=False)
    )
    print("Saved REPORT.md, comparison.png/svg and exact cached overlaps.")


if __name__ == "__main__":
    render()
