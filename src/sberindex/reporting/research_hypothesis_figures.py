"""Scientific plots for frozen calendar, normalization and detector hypotheses."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sberindex.paths import ROOT
from sberindex.reporting.build_figures import GREEN,GRAY,BLUE,ORANGE,save


def main():
    foundation=pd.read_csv(ROOT/'results/foundation_seasonal_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,name,label in zip(axes,['chronos_bolt_tiny','chronos_2'],['Chronos-Bolt Tiny','Chronos-2']):
        for offset,model,text,color in [(-.25,name,'Исходный ряд',GRAY),(0,name+'_seasonal','Без сезонности',GREEN),(.25,'seasonal_pooled','Сезонный ориентир',BLUE)]:
            part=foundation[foundation.model == model].set_index('horizon').loc[[1,3,6,12]]
            bars=ax.bar(np.arange(4)+offset,part.MAE_date_balanced,width=.24,label=text,color=color)
            ax.bar_label(bars,labels=[f'{x:.0f}' for x in part.MAE_date_balanced],padding=3,fontsize=8,rotation=45)
        ax.set(title=label,ylabel='MAE, руб. на МО');ax.set_xticks(range(4),['1 мес.','3 мес.','6 мес.*','12 мес.*'])
        ax.set_ylim(0,13000);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9)
    save(fig,'foundation_seasonal_comparison','Одни наблюдаемые пары и закреплённые веса; профиль только по 2023 году. Десезонирование помогает, но ориентир лучше.\n*6/12 месяцев: по одной дате. Период 2024 уже исследован; историческая доступность предобучающих данных не доказана.')
    detection=pd.read_csv(ROOT/'results/asof_detector_summary.csv')
    detection=detection[(detection.category == 'Все категории')&(detection.cohort == 'asof_june')&(detection.scope == 'all_eligible')&(detection.shift_pct == 20)]
    methods=['spike','rolling_3m','ewma','cusum']
    fig,axes=plt.subplots(1,3,figsize=(12,6))
    for ax,shape,column,title,factor in zip(axes,['pulse','step','step'],['new_by_onset','new_by_third','new_control_per100_observed_months'],['Импульс: новая тревога\nв первый месяц, %','Ступень: новая тревога\nк третьему месяцу, %','Ступень: новые тревоги\nна 100 контрольных МО-месяцев'],[100,100,1]):
        for offset,scaling,label,color in [(-.18,'raw','Исходный сигнал',GRAY),(.18,'regularized_noise','Масштабирование шума',GREEN)]:
            values=detection[(detection.scaling == scaling)&(detection['shape'] == shape)].set_index('method').loc[methods,column]*factor
            bars=ax.bar(np.arange(4)+offset,values,width=.35,label=label,color=color)
            ax.bar_label(bars,labels=[f'{x:.1f}' for x in values],padding=3,fontsize=8)
        ax.set(title=title);ax.set_xticks(range(4),['Разовый','Среднее 3м','EWMA','CUSUM'],rotation=35,ha='right',fontsize=9)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if factor == 100:ax.set_ylim(0,108)
        else:ax.set_ylim(0,5.4)
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    save(fig,'detector_noise_tradeoff','Когорта 2 024 МО по доступности к июню; пороги по январю–июню, сопоставимая калибровочная нагрузка.\nИскусственный +20% у примерно четверти МО, 10 назначений. Контрольные тревоги не являются размеченными ложными тревогами.')
    calendar=pd.read_csv(ROOT/'results/calendar_summary.csv')
    # A single final target at 12m has its own period label in the source.
    fig,ax=plt.subplots(figsize=(10,5.8))
    for offset,model,label,color in [(-.18,'hgb_frozen','HGB без календаря',GRAY),(.18,'hgb_frozen_calendar','HGB с календарём',GREEN)]:
        part=calendar[(calendar.model == model)&(calendar.period != 'early')].set_index('horizon').loc[[1,3,6,12]]
        bars=ax.bar(np.arange(4)+offset,part.MAE,width=.35,label=label,color=color)
        ax.bar_label(bars,labels=[f'{x:.0f}' for x in part.MAE],padding=4,fontsize=10)
    ax.set(title='Календарные признаки не улучшили поздний прогноз',ylabel='MAE, руб. на МО')
    ax.set_xticks(range(4),['1 месяц','3 месяца','6 месяцев*','12 месяцев*']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_ylim(0,3400);ax.legend(frameon=False)
    save(fig,'calendar_ablation','Одинаковое обучение HGB только на 2023 году; одинаковые наблюдаемые пары. Добавлены дни месяца и числа дней недели.\nПраздники не включены. *По одной целевой дате; 2024 год уже исследован. Февраль отсутствует среди обучающих целей HGB.')

    groups = pd.read_csv(ROOT/'results/asof_distribution_groups.csv')
    groups = groups[groups.horizon.eq(1) & groups.dimension.eq('expense_quartile')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for offset, model, label, color in [(-.24,'seasonal_pooled','Сезонная',BLUE),
            (0,'hgb_frozen','HGB без календаря',GRAY),
            (.24,'chronos_2_seasonal','Chronos-2 без сезонности',GREEN)]:
        values = groups[groups.model.eq(model)].set_index('group').loc[['Q1','Q2','Q3','Q4'],'NMAE_2023_pct']
        bars = axes[0].bar(np.arange(4)+offset, values, width=.23, label=label, color=color)
        axes[0].bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=8)
    axes[0].set(title='1 месяц: ошибки по уровню расходов', ylabel='NMAE к средним расходам 2023, %')
    axes[0].set_xticks(range(4), ['Q1','Q2','Q3','Q4'])
    axes[0].set_ylim(0,5.2);axes[0].legend(frameon=False,fontsize=8)
    monthly = pd.read_csv(ROOT/'results/foundation_seasonal_paired_monthly.csv')
    monthly = monthly[monthly.horizon.eq(3) & monthly.model.eq('chronos_2_seasonal') & monthly.reference.eq('chronos_2')].sort_values('target')
    values = monthly.MAE_difference
    bars = axes[1].bar(np.arange(len(values)),values,color=[GREEN if v < 0 else ORANGE for v in values])
    axes[1].bar_label(bars,labels=[f'{v:+.0f}' for v in values],padding=4,fontsize=10)
    axes[1].axhline(0,color=GRAY,linewidth=1)
    axes[1].set(title='3 месяца: выигрыш зависит от декабря',ylabel='MAE нормированного − raw Chronos-2, руб.')
    axes[1].set_xticks(range(4),['Сентябрь','Октябрь','Ноябрь','Декабрь'])
    axes[1].set_ylim(-5900,1700)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'asof_distribution_diagnostics','Q1–Q4 заданы по всем полным рядам 2023; оценка на одних 251 МО. NMAE использует масштаб 2023, не текущий факт.\nСправа отрицательная разность означает улучшение. Без декабря средняя разность +535 руб. Архив 2024 уже исследован.')

    delay = pd.read_csv(ROOT/'results/foundation_delay_summary.csv')
    fig, ax = plt.subplots(figsize=(10,6))
    for model, label, color, style in [('seasonal_pooled','Сезонная',BLUE,'-'),
        ('blend_75','Смесь 75/25',ORANGE,'--'),('hgb_frozen','HGB без календаря',GRAY,'-'),
        ('chronos_2_seasonal','Chronos-2 с профилем 2023',GREEN,'-')]:
        group = delay[delay.model.eq(model)].sort_values('reporting_delay_months')
        ax.plot(group.reporting_delay_months,group.MAE_date_balanced,marker='o',linestyle=style,color=color,label=label)
    ax.set(title='Прогноз следующего месяца при задержке расходов',xlabel='Гипотетическая задержка публикации, месяцев',ylabel='MAE, руб. на МО')
    ax.set_xticks([0,1,2],['0: эффективный h1','1: эффективный h2','2: эффективный h3'])
    ax.set_ylim(650,1550);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=10)
    save(fig,'foundation_reporting_delay','Одни 251 МО, цели сентября–декабря 2024; 8 моделей раскрыты в таблице. Ревизии и параметры фиксированы.\nБизнес-горизонт всегда 1 месяц; при задержке 1 месяц нужны 2 шага. Даты публикации неизвестны, период уже изучен.')

    early = pd.read_csv(ROOT/'results/operational_early_monthly.csv')
    fig, axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
    for ax, period, title in zip(axes,['early','late'],['Февраль–июнь: ранняя проверка','Сентябрь–декабрь: поздняя диагностика']):
        for model,label,color,style in [('hgb_frozen','HGB без календаря',GRAY,'-'),
                ('blend_75','Смесь 75/25',ORANGE,'--'),('seasonal_pooled','Сезонная',BLUE,'-'),
                ('prophet','Prophet',GREEN,'-')]:
            part=early[early.period.eq(period)&early.model.eq(model)].sort_values('target')
            ax.plot(part.target.str[5:],part.MAE,marker='o',color=color,linestyle=style,label=label)
        ax.set(title=title,xlabel='Целевой месяц 2024 года');ax.grid(alpha=.15);ax.set_ylim(0,9500)
    axes[0].set_ylabel('MAE, руб. на МО');axes[1].legend(frameon=False,fontsize=9)
    save(fig,'operational_early_late','Бизнес-горизонт 1 месяц, гипотетическая задержка 1 месяц, эффективный h2; по 251 наблюдаемому МО на дату.\nHGB обучен на переходах апреля–декабря 2023; вес смеси перенесён без настройки. Архив 2024 уже изучен, периоды различаются сезонами.')

    delayed=pd.read_csv(ROOT/'results/detector_delay_summary.csv')
    delayed=delayed[delayed.category.eq('Все категории')&delayed.shift_pct.eq(20)&delayed.scaling.eq('regularized_noise')]
    fig,axes=plt.subplots(1,3,figsize=(12,6),sharey=True)
    for ax,shape,title in zip(axes,['step','pulse','ramp'],['Ступень +20%','Импульс +20%','Плавный рост до +20%']):
        for method,label,color,style in [('spike','Разовый остаток',BLUE,'-'),
                ('rolling_3m','Среднее 3м',GRAY,'--'),('ewma','EWMA',GREEN,'-'),('cusum','CUSUM',ORANGE,':')]:
            part=delayed[delayed['shape'].eq(shape)&delayed.method.eq(method)].sort_values('reporting_delay_months')
            ax.plot(part.reporting_delay_months,100*part.new_by_third_calendar_month,marker='o',color=color,linestyle=style,label=label)
        ax.set(title=title,xlabel='Задержка публикации, месяцев');ax.set_xticks([0,1,2]);ax.set_ylim(0,100);ax.grid(alpha=.15)
    axes[0].set_ylabel('Новая тревога за первые 3 календарных месяца, %')
    axes[2].legend(frameon=False,fontsize=9)
    save(fig,'detector_calendar_delay','Пороги и масштаб шума фиксированы по январю–июню. Общий знаменатель: назначенные МО с первыми 3 наблюдаемыми месяцами.\nСреднее десяти назначений; синтетические сдвиги и гипотетические лаги. Начало: июль для ступени/импульса, август для плавного роста.')


    matched=pd.read_csv(ROOT/'results/bocpd_summary.csv')
    matched=matched[matched.category.eq('Все категории')&matched.scaling.eq('regularized_noise')&matched.shift_pct.eq(20)&matched.reporting_delay_months.eq(0)]
    methods=['spike','rolling_3m','ewma','cusum','bocpd'];labels=['Разовый','Среднее 3м','EWMA','CUSUM','BOCPD']
    fig,axes=plt.subplots(1,3,figsize=(13,6))
    for ax,shape,col,title,mult in zip(axes,['step','ramp','step'],['new_by_third_calendar_month','new_by_third_calendar_month','original_control_per100_delivered_months'],['Ступень +20%: первые 3 месяца','Плавный рост: первые 3 месяца','Исходные контрольные тревоги'],[100,100,1]):
        values=matched[matched['shape'].eq(shape)].set_index('method').loc[methods,col]*mult
        bars=ax.bar(range(5),values,color=[BLUE,GRAY,GREEN,ORANGE,'#7652a3'])
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=9)
        ax.set(title=title,ylabel='Доля новых обнаружений, %' if mult==100 else 'На 100 доставленных МО-месяцев')
        ax.set_xticks(range(5),labels,rotation=35,ha='right');ax.set_ylim(0,105 if mult==100 else 15);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'bocpd_matched_comparison','Общие пороги: максимумы февраля–июня, q99; hazard BOCPD=1/12, остальные параметры прежние. Среднее 10 назначений.\nВсе категории, масштабирование шума, без задержки. Контрольная нагрузка — наблюдаемые тревоги, а не доказанные ложные сигналы.')

    cases=pd.read_csv(ROOT/'results/real_registry_cases.csv');cases=cases[cases.category.eq('Все категории')]
    geo=pd.read_csv(ROOT/'results/real_registry_geography.csv')
    fig,axes=plt.subplots(2,4,figsize=(14,8),sharex=True,sharey=True)
    for ax,link in zip(axes.flat,geo.itertuples()):
        part=cases[cases.event_id.eq(link.event_id)&cases.territory_id.eq(link.territory_id)].sort_values('target')
        ax.plot(np.arange(1,13),part.yoy_pct,color=BLUE,marker='o',markersize=3)
        month=int(part.loc[part.relative_month.eq(0),'target'].iloc[0][-2:]);ax.axvspan(month-.45,month+.45,color=ORANGE,alpha=.22)
        ax.set(title=link.municipality,xlabel='Месяц 2024');ax.set_xticks([1,4,7,10,12]);ax.grid(alpha=.15);ax.axhline(0,color=GRAY,linewidth=.8)
        if not part.source_observed.any():ax.text(.5,.5,'Нет расходного ряда\nс этим официальным ID',transform=ax.transAxes,ha='center',va='center',fontsize=10)
    axes[0,0].set_ylabel('Расходы: изменение год к году, %');axes[1,0].set_ylabel('Расходы: изменение год к году, %')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.8,'6 событий, 7 связей с МО\n4 расходных ряда доступны\nДля 3 ID нет расходного ряда\n\nСобытие не означает\nструктурный сдвиг расходов',va='top',fontsize=12)
    save(fig,'official_event_cases','Все связи замороженного реестра МЧС: оранжевая полоса — месяц документированного события, не период воздействия.\nГодовое изменение номинальных расходов; нет причинной оценки. Один пожар связан с двумя МО; неизвестные окончания не заполнены.')


if __name__ == '__main__':
    main()
