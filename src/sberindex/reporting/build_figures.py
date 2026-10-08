"""Generate publication-size research figures from saved, audited results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sberindex.paths import ROOT
OUT = ROOT / "figures"
GREEN, GRAY, BLUE, ORANGE = "#138B54", "#819089", "#5463AD", "#C37626"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelcolor": "#273C32",
                     "text.color": "#173629", "figure.facecolor": "white",
                     "savefig.facecolor": "white", "svg.fonttype": "none",
                     "svg.hashsalt": "sberindex-task2-2026"})


def save(fig, name, note):
    fig.text(.06, .035, note, fontsize=9, color="#52675C", va="bottom")
    fig.tight_layout(rect=(.025, .11, .98, .96))
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": "2026-10-04"})
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    metrics = pd.read_csv(ROOT / "results/forecast_comparison.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    chosen, prophet = [], []
    for h in (1, 3, 6, 12):
        sub = metrics[metrics.horizon == h].set_index("model")
        chosen.append(sub.loc["global_hgb" if h == 12 else "blend_75", "MAE"])
        prophet.append(sub.loc["prophet", "MAE"])
    x = np.arange(4)
    for offset, vals, label, color in [(-.19, prophet, "Prophet", GRAY),
                                      (.19, chosen, "Смесь 75/25; на 12 мес. — HGB*", GREEN)]:
        bars = ax.bar(x + offset, vals, width=.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:,.0f}".replace(",", " ") for v in vals], padding=4, fontsize=11)
    ax.set_xticks(x, ["1 месяц\n6 дат", "3 месяца\n4 даты", "6 месяцев\n1 дата", "12 месяцев*\n1 дата"])
    ax.set_ylabel("MAE, руб. на муниципалитет")
    ax.set_title("Сравнение после даты настройки смеси")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(prophet) * 1.26)
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    save(fig, "forecast_quality", "256 одинаковых МО. Для 1–6 мес. происхождение прогноза ≥ июнь 2024.\n*12 мес.: модель выбрана в поисковом сравнении по единственной дате; независимая проверка отсутствует.")

    growth = pd.read_parquet(ROOT / "results/real_growth_interpretation.parquet")
    months = sorted(growth.date.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, category in zip(axes, ["Все категории", "Продовольствие"]):
        monthly = growth[growth.category == category].groupby("date")[["nominal_yoy_pct", "real_yoy_pct"]].median().reindex(months)
        ax.plot(range(12), monthly.nominal_yoy_pct, color=GRAY, marker="o", label="Номинальный рост")
        ax.plot(range(12), monthly.real_yoy_pct, color=GREEN, marker="o", label="После поправки на ИПЦ")
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set_xticks(range(0, 12, 2), ["янв", "мар", "май", "июл", "сен", "ноя"])
        ax.set_title(category); ax.set_xlabel("2024 год"); ax.grid(alpha=.15)
    axes[0].set_ylabel("Медианный рост к тому же месяцу 2023, %")
    axes[1].legend(frameon=False, fontsize=10)
    save(fig, "nominal_real", "2 016 полных МО; невзвешенные муниципальные медианы, не общероссийский объём расходов.\nРегиональный ИПЦ Росстата: общий / продовольствие. Винтаж сентября 2026, ретроспективная интерпретация.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    city_base = growth[(growth.region_code == 77) & (growth.category == "Все категории")].groupby("date").real_yoy_pct.median().reindex(months)
    for ax, city_id in zip(axes, [2593, 2595]):
        city = growth[(growth.territory_id == city_id) & (growth.category == "Все категории")].set_index("date").reindex(months)
        ax.axvspan(5.5, 11.5, color=GREEN, alpha=.06)
        ax.plot(range(12), city.nominal_yoy_pct, color=GRAY, linestyle="--", label="Номинальный рост")
        ax.plot(range(12), city.real_yoy_pct, color=GREEN, marker="o", label="После поправки на ИПЦ")
        ax.plot(range(12), city_base, color=BLUE, linewidth=1.6, label="Медиана МО Москвы, после ИПЦ")
        ax.axvline(4.25, color=ORANGE, linestyle=":", linewidth=1.5)
        ax.text(4.38, 52, "Закон о реформе\n8 мая", color=ORANGE, fontsize=9)
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set_xticks(range(0, 12, 2), ["янв", "мар", "май", "июл", "сен", "ноя"])
        ax.set_ylim(-30, 62); ax.set_xlim(-.4, 11.4)
        ax.set_title(f"{city.municipal_district_name_short.iloc[0]} · ID {city_id}")
        ax.set_xlabel("2024 год")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Рост расходов к тому же месяцу 2023, %")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "named_case", "Светлая область — период после первой тревоги (июль); рост начинается раньше. Сигнал сохраняется после ИПЦ.\nРеформа — гипотеза о сопоставимости учёта, причинная связь не доказана. Названия на начало 2024; ИПЦ ретроспективный.")

    cm = pd.read_csv(ROOT / "results/category_forecast_comparison.csv")
    wide = cm.pivot(index=["category", "horizon"], columns="model", values="MAE")
    gain = (100 * (wide.prophet - wide.seasonal_pooled) / wide.prophet).unstack("horizon")
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(gain.to_numpy(), cmap="RdYlGn", vmin=-30, vmax=70, aspect="auto")
    ax.set_xticks(range(3), ["1 месяц", "3 месяца", "6 месяцев"])
    ax.set_yticks(range(6), gain.index)
    for i in range(6):
        for j in range(3):
            ax.text(j, i, f"{gain.iloc[i,j]:+.0f}%", ha="center", va="center", color="#132B21", fontsize=15, weight="bold")
    ax.set_title("Сезонная модель против Prophet: где уменьшается MAE")
    fig.colorbar(image, ax=ax, label="Снижение MAE, %")
    save(fig, "category_quality", "128 фиксированных МО; цели июль–декабрь 2024; по 6 дат на горизонт.\nЭто диагностическое сравнение двух заданных моделей; отрицательное число означает преимущество Prophet.")

    det = pd.read_csv(ROOT / "results/change_robustness.csv")
    det = det[det.category == "Все категории"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharex=True, sharey=True)
    labels = {"spike": "Разовый", "rolling_3m": "Среднее 3м", "ewma": "EWMA", "cusum": "CUSUM"}
    colors = {"spike": BLUE, "rolling_3m": GREEN, "ewma": ORANGE, "cusum": GRAY}
    for ax, shape, title in zip(axes, ["permanent_20pct", "ramp_0_to_20pct", "one_month_20pct"],
                                ["Постоянный +20%", "Плавно от 0 до +20%", "Всплеск +20% на месяц"]):
        for row in det[det.shock_shape == shape].itertuples():
            x, y = 100 * row.false_alarm_rate, 100 * row.detection_rate
            ax.scatter(x, y, color=colors[row.method], s=70)
            offset = -14 if (y > 95 or (shape == "ramp_0_to_20pct" and row.method == "rolling_3m")) else 6
            ax.annotate(labels[row.method], (x, y), xytext=(4, offset), textcoords="offset points", fontsize=8)
        ax.set_title(title); ax.set_xlim(0, 37); ax.set_ylim(0, 108)
        ax.set_xlabel("Тревоги, % контрольных МО"); ax.grid(alpha=.15)
    axes[0].set_ylabel("Обнаружено к декабрю, % затронутых МО")
    save(fig, "detector_tradeoff", "Все расходы; синтетические изменения у 25% МО с июля; пороги по январю–июню.\nВ контроле возможны естественные изменения: реальная частота ложных тревог не установлена.")
    paired = pd.read_csv(ROOT / "results/event_attribution_summary.csv")
    paired = paired[(paired.category == "Все категории") & (paired.shift_pct == 20)]
    order = ["spike", "rolling_3m", "ewma", "cusum"]
    names = ["Разовый", "Среднее 3м", "EWMA", "CUSUM"]
    pulse = paired[paired['shape'] == 'pulse'].set_index('method').loc[order]
    step = paired[paired['shape'] == 'step'].set_index('method').loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    for ax, vals, title, color in [
        (axes[0], pulse.new_at_onset * 100, "Импульс: тревога в июле", BLUE),
        (axes[1], step.new_by_third_active_month * 100, "Постоянный сдвиг: к сентябрю", GREEN),
        (axes[2], step.new_any_control * 100, "Новые тревоги у незатронутых", ORANGE),
    ]:
        bars = ax.barh(np.arange(4), vals, color=color)
        ax.bar_label(bars, labels=[f"{x:.1f}%" for x in vals], padding=3, fontsize=9)
        ax.set_yticks(np.arange(4), names)
        ax.set_xlim(0, 108 if ax != axes[2] else 28)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    save(fig, "detector_event_timing", "Все расходы, сдвиг +20% у 25% МО; среднее 10 синтетических распределений. Тревога засчитывается,\nесли её нет в тот же месяц на исходном ряду. Правый график — перенос эффекта через общий ориентир, не оценка реальных ложных тревог.")
    scope = pd.read_csv(ROOT / "results/change_scope_summary.csv")
    scope = scope[(scope.category == "Все категории") & (scope.shift_pct == 20) & (scope.method == "rolling_3m")]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = scope.coverage.to_numpy() * 100
    for key, label, color in [("treated_alarm_rate", "Затронутые МО", GREEN),
                              ("control_alarm_rate", "Незатронутые МО", ORANGE),
                              ("new_alarm_rate_all_treated", "Новые тревоги у затронутых", BLUE)]:
        ax.plot(x, scope[key + "_mean"] * 100, marker="o", label=label, color=color)
        ax.fill_between(x, scope[key + "_min"] * 100, scope[key + "_max"] * 100, color=color, alpha=.12)
    ax.set(xlabel="Доля МО с искусственным сдвигом +20%, %", ylabel="Доля МО с тревогой, %",
           title="Массовый шок меняет ориентир локального детектора", ylim=(0, 102))
    ax.set_xticks(x); ax.legend(frameon=False); ax.grid(alpha=.15)
    save(fig, "detector_scope", "Все расходы, 2 016 МО; среднее за 3 месяца; фиксированный порог до июля.\nЛинии — среднее 10 распределений, полосы — минимум/максимум, не доверительный интервал.")
    hierarchy = pd.read_csv(ROOT / "results/hierarchical_summary.csv")
    hierarchy = hierarchy[(hierarchy.category == "Все категории") & (hierarchy.shift_pct == 20) & (hierarchy["shape"] == "permanent")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    order = ["spike", "rolling_3m", "ewma", "cusum"]
    for ax, scope, title in zip(axes, ["national", "regional", "local"],
                               ["Общая панель: 1 сценарий", "Регион: 3 сценария", "МО: 3 распределения"]):
        part = hierarchy[hierarchy.scope == scope].set_index("method").loc[order]
        x = np.arange(4)
        ax.bar(x-.2, part.target_baseline_alarm_rate*100, .38, color=GRAY, label="На исходных данных")
        ax.bar(x+.2, part.target_new_alarm_rate*100, .38, color=GREEN, label="Новые после шока")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, ["Разовый", "3 месяца", "EWMA", "CUSUM"], rotation=30)
        ax.set_ylim(0, 108); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel("Доля целевых единиц с тревогой, %")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(-.1, 1.24), ncol=2)
    save(fig, "hierarchical_audit", "Все расходы, постоянный +20%; оценка уровня январь–март, калибровка апрель–июнь, проверка июль–декабрь.\nЕдиницы: панель / регион / МО. Много исходных тревог: эксперимент не обосновывает замену основного детектора.")
    trajectory = pd.read_csv(ROOT / "results/forecast_residual_trajectories.csv")
    trajectory = trajectory[(trajectory.category == "Все категории") & (trajectory.shift_pct == 20)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    for ax, shape, title in zip(axes, ["permanent", "one_month", "ramp"],
                               ["Постоянный +20%", "Всплеск на месяц", "Плавно до +20%"]):
        for model, label, color in [("adaptive_1m", "Обновлять каждый месяц", GREEN),
                                    ("frozen_6m", "Сохранить июньский прогноз", BLUE)]:
            part = trajectory[(trajectory["shape"] == shape) & (trajectory.model == model)].sort_values("month_after")
            ax.plot(part.month_after, part.induced_median_log_error * 100, marker="o", color=color, label=label)
        ax.set_title(title); ax.set_xticks(range(6), ["июл", "авг", "сен", "окт", "ноя", "дек"])
        ax.axhline(0, color=GRAY, lw=.8); ax.grid(alpha=.15)
    axes[0].set_ylabel("Добавленная ошибка, 100 × логарифм отношения")
    axes[1].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, 1.25), ncol=2)
    save(fig, "forecast_residual_memory", "Парная разность ошибок изменённой и исходной панели; общий шок, все расходы.\nЭто механизм реакции, а не частота обнаружения. Фиксированный прогноз сохраняет и собственные ошибки модели.")
    group = pd.read_csv(ROOT / "results/forecast_audit_groups.csv")
    group = group[group.dimension == "expense_quartile"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for ax, h in zip(axes, [1, 3]):
        for model, label, color in [("prophet", "Prophet", GRAY), ("blend_75", "Смесь 75/25", GREEN)]:
            part = group[(group.horizon == h) & (group.model == model)].sort_values("group")
            ax.plot(part["group"], part.MAE, marker="o", color=color, label=label)
        ax.set_title(f"Горизонт {h} мес."); ax.set_xlabel("Группа по средним расходам 2023 года")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("MAE, руб."); axes[1].legend(frameon=False)
    save(fig, "forecast_expense_groups", "Границы квартилей определены по 2023 году, на всех 2 016 полных МО; прогнозы — для исходных 256 МО.\nQ1 — наименьшие расходы, Q4 — наибольшие. Диагностика на использованном архиве, без перенастройки модели.")
    coverage = pd.read_csv(ROOT / "results/online_interval_monthly.csv")
    coverage = coverage[(coverage.model == "blend_75") & (coverage.nominal_coverage == .95)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for strategy, label, color in [("fixed_jul_aug", "Июль–август", GRAY),
                                    ("expanding", "Вся доступная история ошибок", BLUE),
                                    ("rolling_3m", "Последние 3 месяца", GREEN)]:
        part = coverage[coverage.strategy == strategy].sort_values("target")
        axes[0].plot(range(4), part.coverage*100, marker="o", label=label, color=color)
        axes[1].plot(range(4), part.width_rub, marker="o", label=label, color=color)
    axes[0].axhline(95, color=ORANGE, ls="--", label="Номинальные 95%")
    axes[0].set(title="Фактическое покрытие", ylabel="Доля значений внутри интервала, %", ylim=(65,100))
    axes[1].set(title="Цена расширения интервала", ylabel="Средняя ширина, руб.")
    for ax in axes: ax.set_xticks(range(4), ["сен", "окт", "ноя", "дек"]); ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "interval_reliability", "Смесь75/25, 256МО, горизонт1м. Для каждого прогноза используются только ранее доступные ошибки.\nПредполагается отсутствие задержки публикации. Зависимая короткая история не даёт гарантии покрытия95%.")
    adaptive = pd.read_csv(ROOT / "results/adaptive_forecast_monthly.csv")
    adaptive = adaptive[(adaptive.horizon == 1) & adaptive.model.isin(["blend_75", "adaptive_rolling_3m"])]
    monthly = adaptive.pivot(index="target", columns="model", values="ae").sort_index()
    gain = monthly.blend_75-monthly.adaptive_rolling_3m
    fig, ax = plt.subplots(figsize=(12, 5.8))
    bars=ax.bar(range(len(gain)), gain, color=[GREEN if x>=0 else ORANGE for x in gain])
    ax.bar_label(bars, fmt="%.0f", padding=4)
    ax.set_xticks(range(6), ["июл", "авг", "сен", "окт", "ноя", "дек"])
    ax.axhline(0,color=GRAY,lw=.8);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    ax.set(title="Адаптация улучшает среднюю ошибку, но не каждый месяц", ylabel="MAE прежней смеси − MAE адаптивной модели, руб.")
    save(fig, "adaptive_forecast_months", "256МО, горизонт1месяц; выбор из25смесей по трём ранее наблюдавшимся месяцам.\nПоложительное значение — улучшение. Основной выигрыш приходится на декабрь; независимого теста нет.")
    health = pd.read_csv(ROOT / "results/robust_anchor_monthly_gains.csv")
    health = health[(health.category == "Здоровье") & (health.cohort == "all") & (health.horizon == 1)].sort_values("target")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(range(6), health.baseline_MAE, marker="o", color=GRAY, label="Последний месяц")
    ax.plot(range(6), health.MAE, marker="o", color=GREEN, label="Взвешенный уровень трёх месяцев")
    ax.set_xticks(range(6), ["июл", "авг", "сен", "окт", "ноя", "дек"])
    ax.set(title="Здоровье: сглаживание помогает в четырёх из шести месяцев", ylabel="MAE, руб.")
    ax.legend(frameon=False);ax.grid(alpha=.15)
    save(fig, "health_anchor_comparison", "2 016МО, горизонт1месяц; одинаковая региональная сезонность, веса0,2/0,3/0,5 выбраны из заданных вариантов.\nВыбор — по февралю–июню. Октябрь/декабрь ухудшаются; оценка на уже использованной истории.")
    news = pd.read_csv(ROOT / 'results/news_corpus_monthly.csv').pivot(index='target', columns='model', values='MAE')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for name, label, color in [('without_news','Без новостей',GRAY),('hikes_only','Только повышения',BLUE),('all_decisions','Все решения',GREEN)]:
        axes[0].plot(range(11),news[name],marker='o',label=label,color=color)
        if name != 'without_news':
            axes[1].plot(range(11),news[name]-news.without_news,marker='o',label=label,color=color)
    for ax in axes:
        ax.set_xticks(range(11),['фев','мар','апр','май','июн','июл','авг','сен','окт','ноя','дек'],rotation=45)
        ax.axvline(4.5,color=GRAY,ls=':',lw=1);ax.grid(alpha=.15)
    axes[0].set(title='Ошибка трёх вариантов HGB',ylabel='MAE, руб.')
    axes[1].set(title='Добавление новостей: выигрыш нестабилен',ylabel='MAE с новостями − MAE без новостей, руб.')
    axes[1].axhline(0,color=GRAY,lw=.8);axes[0].legend(frameon=False,fontsize=8)
    news_coverage = pd.read_csv(ROOT/'results/news_corpus_coverage.csv')
    save(fig,'news_corpus_comparison',f'Когорта: 2 075 МО по 2023 году; оценено {news_coverage.scored_ids.min()}–{news_coverage.scored_ids.max()} МО на дату. Горизонт 1 месяц.\nДевять обучающих переходов; одинаковые настройки. Справа отрицательные значения — улучшение. 2024 год уже изучался.')
    weather=pd.read_csv(ROOT/'results/weather_news_articles.csv')
    coverage=pd.read_csv(ROOT/'results/weather_news_coverage.csv')
    monthly=weather.groupby(weather.published_date.str[:7]).size()
    top=coverage[coverage.in_sber_panel].sort_values('articles').tail(10)
    fig,axes=plt.subplots(1,2,figsize=(12,6.4),gridspec_kw={'width_ratios':[1,1.2]})
    axes[0].plot(range(24),monthly,marker='o',color=BLUE)
    axes[0].set_xticks([0,5,11,17,23],['янв 23','июн 23','дек 23','июн 24','дек 24'],rotation=35)
    axes[0].set(title='Полнота годовых индексов',ylabel='Записи в месяц')
    axes[1].barh(top.region_name,top.articles,color=GREEN)
    axes[1].set(title='География заголовков неоднородна',xlabel='Число явных упоминаний; 10 лидеров')
    axes[0].grid(alpha=.15);axes[1].grid(axis='x',alpha=.15);axes[1].set_axisbelow(True)
    save(fig,'weather_news_coverage','Источник: Росгидромет, годовые индексы 2023–2024; 2 441 запись. Регион найден в 569 заголовках.\n29 из 77 субъектов исходной панели. Это охват публикаций, а не частота опасных явлений; тела статей не разобраны.')
    geo=pd.read_csv(ROOT/'results/weather_body_geography_comparison.csv').groupby('method')[['TP','FP','FN']].sum()
    methods=['title','body_direct','body_coordinated']
    found=geo.loc[methods,'TP'].to_numpy();missed=geo.loc[methods,'FN'].to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(12,6.2))
    axes[0].bar(range(3),found,color=GREEN,label='Найдены')
    axes[0].bar(range(3),missed,bottom=found,color='#DCE4DF',label='Пропущены')
    for i,n in enumerate(found):axes[0].text(i,n+1,str(n),ha='center')
    axes[0].set_xticks(range(3),['Заголовок','Полный текст','Текст +\nперечисления'])
    axes[0].set(title='Региональные упоминания: 5 предупреждений',ylabel='Из 56 размеченных пар статья–регион',ylim=(0,65))
    axes[0].legend(frameon=False,fontsize=9)
    review=pd.read_csv(ROOT/'results/weather_body_review.csv')
    flagged=review.title_predicts_weather
    matrix=np.array([[(flagged&review.weather_prediction_in_body).sum(),(flagged&~review.weather_prediction_in_body).sum()],
                     [(~flagged&review.weather_prediction_in_body).sum(),(~flagged&~review.weather_prediction_in_body).sum()]])
    axes[1].imshow(matrix,cmap='Greens',vmin=0,vmax=15)
    for (i,j),n in np.ndenumerate(matrix):axes[1].text(j,i,str(n),ha='center',va='center',fontsize=20)
    axes[1].set_xticks([0,1],['Есть прогноз\nв теле статьи','Нет прогноза\nв теле статьи'])
    axes[1].set_yticks([0,1],['Заголовок: прогноз\nили предупреждение','Другой заголовок'])
    axes[1].set_title('Содержание: 34 проверенных текста')
    save(fig,'weather_body_validation','Один разметчик — ассистент; выборка стратифицирована. Правило перечислений разработано на этих текстах.\nЭто диагностика, не независимая точность. Четыре прогноза в PDF не считались прогнозами в теле страницы.')
    dv=pd.read_csv(ROOT/'results/deseasonal_hgb_validation.csv')
    dl=pd.read_csv(ROOT/'results/deseasonal_hgb_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    for ax,values,title in [(axes[0],dv,'Ранняя проверка: февраль–июнь'),(axes[1],dl[dl.horizon==1],'Поздняя оценка: июль–декабрь')]:
        ax.plot(values.hgb_weight,values.MAE,marker='o',color=BLUE)
        ax.set(title=title,xlabel='Доля HGB после удаления сезонности',ylabel='MAE, руб.')
        ax.set_xticks([0,.25,.5,.75,1]);ax.grid(alpha=.15)
    save(fig,'deseasonal_hgb_selection','256 МО, горизонт 1 месяц. Ранняя проверка выбрала нулевой вес HGB; основной прогноз сохранён.\nПозднее улучшение отдельных вариантов не используется для выбора задним числом. Независимого теста нет.')
    event = pd.read_csv(ROOT / 'results/real_event_orsk.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, category in zip(axes, ['Все категории', 'Продовольствие']):
        values = event[event.category == category].sort_values('month')
        ax.axvspan(.65, 1.35, color=ORANGE, alpha=.12)
        ax.plot(range(4), values.yoy_pct, marker='o', color=GREEN, label='Орск')
        ax.plot(range(4), values.panel_median_yoy_pct, marker='o', color=GRAY,
                label='Медиана МО с наблюдаемой парой')
        ax.set_xticks(range(4), ['мар', 'апр', 'май', 'июн'])
        ax.set_title(category)
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Рост расходов к тому же месяцу 2023, %')
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    save(fig, 'orsk_event_context', 'МЧС: прорыв дамбы 5 апреля 2024, сообщение 6 апреля. Затемнение — месяц события.\n'
         'Средние расходы жителей МО; событие отобрано ретроспективно. График не доказывает причинный эффект или точность детектора.')
    lag = pd.read_csv(ROOT / 'results/asof_reporting_delay_summary.csv')
    fig, ax = plt.subplots(figsize=(10, 5.7))
    for model, label, color in [('prophet', 'Prophet', GRAY),
                                ('seasonal_pooled', 'Сезонная модель', GREEN),
                                ('blend_75', 'Смесь 75/25', BLUE)]:
        part = lag[lag.model == model].sort_values('reporting_delay_months')
        ax.plot(part.reporting_delay_months, part.MAE_date_balanced,
                marker='o', linewidth=2, color=color, label=label)
        for x, y in zip(part.reporting_delay_months, part.MAE_date_balanced):
            offset = (-17 if x != 1 else 8) if model == 'seasonal_pooled' else (-17 if x == 1 and model == 'blend_75' else 7)
            ax.annotate(f'{y:.0f}', (x, y), xytext=(0, offset), textcoords='offset points',
                        ha='center', color=color, fontsize=9)
    ax.set_xticks([0, 1, 2], ['0 мес.', '1 мес.', '2 мес.'])
    ax.set(xlabel='Гипотетическая задержка данных', ylabel='MAE, руб. на МО',
           title='Цена задержки в доступности расходов')
    ax.set_ylim(700, 2500)
    ax.grid(alpha=.15); ax.legend(frameon=False, loc='center left', bbox_to_anchor=(.02, .53))
    save(fig, 'asof_reporting_delay', 'Одна когорта, заданная по 2023 году; одинаковые наблюдаемые пары МО–цель, сентябрь–декабрь 2024.\n'
         'Выпуск в конце t, цель t+1. Исторических дат публикации нет; это сценарий, не измеренная задержка. Период уже изучался.')
    regional = pd.read_csv(ROOT/'results/asof_regional_comparison.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.8))
    models = ['seasonal_pooled','blend_selected','regional_asof_selected','prophet']
    labels = ['Сезонный\nориентир','Общая\nсмесь','Региональная\nсмесь','Prophet']
    for ax, horizon in zip(axes, [1,3,6]):
        group = regional[regional.horizon == horizon].set_index('model')
        vals = group.loc[models, 'MAE_date_balanced']
        bars = ax.bar(range(4), vals, color=[GRAY,BLUE,GREEN,ORANGE])
        ax.bar_label(bars, labels=[f'{v:.0f}' for v in vals], padding=4, fontsize=10)
        dates = int(group.dates.iloc[0])
        ax.set(title=f'{horizon} мес. · {dates} '+('дата' if dates == 1 else 'дат'))
        ax.set_xticks(range(4),labels,rotation=35,ha='right',fontsize=9)
        ax.set_ylim(0,float(vals.max())*1.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('MAE, руб. на МО; равный вес дат')
    save(fig,'asof_regional_comparison','251 одинаковый МО на дату; когорта и сезонные профили определены по 2023 году.\n'
         'Региональная смесь выбрана из 25 вариантов по февралю–июню. На 6 мес. одна дата; 2024 год уже изучался.')
    print(f"Saved 20 figures as PNG and SVG to {OUT}")


if __name__ == "__main__":
    main()
