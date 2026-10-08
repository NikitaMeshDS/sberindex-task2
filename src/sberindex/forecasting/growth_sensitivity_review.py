"""Fixed growth sensitivity on saved regional forecasts; no model selection."""
import hashlib
import json

import numpy as np
import pandas as pd

from sberindex.forecasting.growth_bridge_review import year_bridges
from sberindex.paths import ROOT

OUT = ROOT / 'reports/growth_sensitivity_review'
KEYS = ['territory_id', 'origin', 'target', 'horizon']


def apply_growth(predicted, origins, horizons, rate):
    """Transform frozen predictions without accepting target observations."""
    if not np.isfinite(rate) or rate <= -1:
        raise ValueError('Growth must be finite and greater than -1')
    k = np.array([year_bridges(o, h) for o, h in zip(origins, horizons)])
    return np.asarray(predicted, dtype=float) * (1 + rate) ** k


def available_at_origin(origin, source):
    return pd.Period(origin, 'M').end_time.normalize() >= pd.Timestamp(source['available_from'])


def metric_rows(frame, scenario, rate, kind):
    rows = []
    for subset, selected in [('all', frame), ('crossing_only', frame[frame.year_bridges > 0])]:
        for horizon, group in selected.groupby('horizon'):
            rows.append(dict(scenario=scenario, growth_rate=rate, scenario_kind=kind,
                             subset=subset, horizon=int(horizon), pairs=len(group),
                             origins=group.origin.nunique(), dates=group.target.nunique(),
                             municipalities=group.territory_id.nunique(),
                             MAE=float(group.groupby('target').ae.mean().mean())))
    return rows


def plot_sensitivity(summary, comparator):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    colors = ['#00798c', '#c65d1b', '#5362a2', '#7f416b']
    grid = summary[summary.scenario_kind == 'fixed_diagnostic_grid']
    for ax, subset, title in zip(axes, ['all', 'crossing_only'],
                                 ['All common pairs (date-balanced)', 'Calendar-year crossing only (one origin)']):
        for h, color in zip([1, 3, 6, 12], colors):
            s = grid[(grid.subset == subset) & (grid.horizon == h)]
            ax.plot(s.growth_rate * 100, s.MAE, marker='o', color=color, label=f'h={h}')
        ax.axvline(7.4, color='#666666', linestyle=':', linewidth=1)
        ax.axvline(17.2, color='#666666', linestyle='--', linewidth=1)
        ax.set(xlabel='Fixed growth scenario g (%)', ylabel='MAE (nominal rubles)', title=title)
        ax.grid(alpha=.15)
        ax.legend(ncol=2, frameon=False)
    fig.suptitle('Regional weight 0.5: growth sensitivity, no rate selection', fontsize=14)
    fig.text(.01, -.02, '7.4%: December 2023 CPI, ex-post only (released Jan 2024). 17.2%: October wage YoY, released Dec 2023.', fontsize=9)
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'sensitivity.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    cfg = json.loads((ROOT / 'configs/growth_sensitivity_review.json').read_text())
    sources = json.loads((ROOT / cfg['source_record']).read_text())
    saved = pd.read_parquet(ROOT / cfg['base_predictions'])
    base = saved[saved.model == cfg['base_model']].copy().reset_index(drop=True)
    if len(base) != 60700 or base.duplicated(KEYS).any():
        raise ValueError('Expected exactly 60,700 unique common pairs')
    base['year_bridges'] = [year_bridges(o, h) for o, h in zip(base.origin, base.horizon)]
    grid = cfg['fixed_diagnostic_grid']
    frames, rows = [], []
    scenarios = [(g, f'g{100*g:g}', 'fixed_diagnostic_grid') for g in grid]
    scenarios.append((cfg['supplemental_asof_cpi_rate'], 'cpi_november_asof', 'supplemental_asof'))
    for rate, scenario, kind in scenarios:
        frame = base.copy()
        frame['predicted'] = apply_growth(base.predicted, base.origin, base.horizon, rate)
        np.testing.assert_array_equal(frame.loc[base.year_bridges == 0, 'predicted'],
                                      base.loc[base.year_bridges == 0, 'predicted'])
        frame['ae'] = abs(frame.actual - frame.predicted)
        frame['scenario'] = scenario
        frame['growth_rate'] = rate
        rows.extend(metric_rows(frame, scenario, rate, kind))
        frames.append(frame[KEYS + ['actual', 'predicted', 'ae', 'year_bridges', 'scenario', 'growth_rate']])
    summary = pd.DataFrame(rows)
    baseline = summary[summary.growth_rate == 0].set_index(['subset', 'horizon']).MAE
    summary['gain_vs_g0_pct'] = [100 * (1 - r.MAE / baseline.loc[(r.subset, r.horizon)])
                                   for r in summary.itertuples()]
    previous = pd.read_csv(ROOT / 'reports/growth_bridge_review/summary.csv')
    comparator = previous[previous.model.str.startswith('prophet_')].copy()
    OUT.mkdir(exist_ok=True, parents=True)
    summary.to_csv(OUT / 'summary.csv', index=False)
    comparator.to_csv(OUT / 'prophet_comparators.csv', index=False)
    pd.concat(frames, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
    plot_sensitivity(summary, comparator)
    paths = ['configs/growth_sensitivity_review.json', cfg['source_record'],
             cfg['base_predictions'], 'reports/growth_bridge_review/summary.csv',
             'data/external/growth_bridge_review/growth_source.json',
             'src/sberindex/forecasting/growth_sensitivity_review.py',
             'docs/protocols/GROWTH_SENSITIVITY_REVIEW.md']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = dict(status='passed', pairs_per_scenario=len(base), scenarios=len(scenarios),
                 identical_pairs=True, k0_unchanged=True, target_actual_used_for_prediction=False,
                 regional_weight=cfg['region_weight'], diagnostic_grid=grid,
                 grid_fixed_before_this_computation=True, grid_defined_after_review_of_2024=True,
                 chosen_rate=None, model_selection=False, primary_model_replaced=False,
                 independent_time_validation=False, crossing_origins=sorted(base.loc[base.year_bridges > 0, 'origin'].unique().tolist()),
                 cpi_december_available_at_dec2023_origin=available_at_origin('2023-12', sources['cpi_december']),
                 cpi_november_available_at_dec2023_origin=available_at_origin('2023-12', sources['cpi_november']),
                 sources=sources, input_sha256={p: sha(ROOT / p) for p in paths},
                 output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.suffix in ['csv', 'parquet', 'png', 'svg']})
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Чувствительность MAE к фиксированной поправке роста', '',
             'Региональный вес 0,5; неизменные 60 700 пар. MAE в номинальных рублях: сначала среднее по МО внутри target, затем равное среднее по датам. Сетка фиксирована до данного расчёта, после просмотра 2024; это описательная диагностика, без выбора g и без замены модели.', '',
             '| g, % | h1 | h3 | h6 | h12 |', '|---|---:|---:|---:|---:|']
    for rate in grid:
        s = summary[(summary.growth_rate == rate) & (summary.subset == 'all')].set_index('horizon')
        lines.append('| ' + f'{100*rate:g}' + ' | ' + ' | '.join(f'{s.loc[h,"MAE"]:.2f}' for h in cfg['horizons']) + ' |')
    lines += ['', 'Все ненулевые ставки сетки улучшают h3/6/12 относительно g=0; h1 улучшается лишь до 17,2% включительно, а 20% и 25% ухудшают его. При 17,2% выигрыш h1 лишь 0,51%; результаты существенно зависят от ставки. Сетка не обосновывает единственную оптимальную ставку.', '',
              '| g, %: только k>0 | h1 | h3 | h6 | h12 |', '|---|---:|---:|---:|---:|']
    for rate in [0, .074, .172]:
        s = summary[(summary.growth_rate == rate) & (summary.subset == 'crossing_only')].set_index('horizon')
        lines.append('| ' + f'{100*rate:g}' + ' | ' + ' | '.join(f'{s.loc[h,"MAE"]:.2f}' for h in cfg['horizons']) + ' |')
    lines += ['', 'Для k>0 число пар h1/3/6/12: 2031/2031/2025/2023. Во всех случаях единственный origin — декабрь 2023. Общая MAE включает 12/10/7/1 дат и 24282/20233/14162/2023 пары. Количество МО не создаёт независимых временных повторений; независимость и статистическая значимость не заявляются.', '',
              '## Источники и доступность', '',
              '[Рост номинальной начисленной зарплаты октября 2023:17,2%](https://rosstat.gov.ru/storage/mediabank/osn-11-2023.pdf), доклад опубликован 27 декабря 2023; доступен декабрьскому origin при прежнем лаге 0. Это исходный фиксированный wage proxy, не рост реальных доходов.', '',
              '[Декабрьский ИПЦ опубликован 12 января 2024](https://rosstat.gov.ru/central-news?page=52&per_page=10&print=1). [Ретроспективная таблица Росстата](https://rosstat.gov.ru/storage/mediabank/1_15-01-2025.html) подтверждает 7,42% декабрь/декабрь и 5,87% в среднем за 2023. Запрошенные 7,4% — округлённый декабрьский ИПЦ: только ex-post диагностика, использование как известного в декабре 2023 параметра создаёт look-ahead.', '',
              '[Ноябрьский ИПЦ 7,48%](https://rosstat.gov.ru/storage/mediabank/194_08-12-2023.pdf) опубликован 8 декабря 2023 и доступен декабрьскому origin. Его отдельный сценарий не меняет запрошенную сетку:']
    s = summary[(summary.scenario == 'cpi_november_asof') & (summary.subset == 'all')].set_index('horizon')
    lines.append('MAE h1/3/6/12 = ' + '/'.join(f'{s.loc[h,"MAE"]:.2f}' for h in cfg['horizons']) + ' руб.')
    lines += ['', 'ИПЦ описывает цены фиксированной потребительской корзины, зарплата — начисленное вознаграждение работников. Номинальные расходы также зависят от объёмов, состава покупок, сбережений и покрытия SberIndex. Перенос любой ставки в расходы с эластичностью 1 — предположение, не экономическое тождество и не причинная оценка. Диагностические 5/10/15/20/25% не являются отдельными опубликованными макропоказателями.', '',
              '## Проверенные Prophet варианты', '', '| h | disabled | pooled_profile | yearly3 |', '|---|---:|---:|---:|']
    p = comparator.pivot(index='horizon', columns='model', values='MAE')
    for h in cfg['horizons']:
        lines.append(f'| {h} | ' + ' | '.join(f'{p.loc[h,m]:.2f}' for m in ['prophet_disabled','prophet_pooled_profile','prophet_yearly3']) + ' |')
    lines += ['', 'Сравнения перенесены из существующей summary; нового переобучения Prophet и выбора модели нет. Для h1/3/6 вся сетка 0–25% сохраняет меньшую общую MAE, чем каждый проверенный Prophet вариант. Для h12 это верно для сценариев 7,4–20%; при 0/5/25% преимущество перед лучшим проверенным Prophet отсутствует. Этот факт относится к имеющейся выборке и не гарантирует перенос на новые годы.', '',
              '![Фиксированная чувствительность](sensitivity.png)', '',
              'Источники зафиксированы в sources.json; audit.json содержит доступность, ограничения и SHA256. Прямое открытие архива Росстата завершилось timeout; подтверждение дат получено из индексированного официального архива, историческая неизменность файлов не доказана.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(summary[summary.subset == 'all'][['scenario','horizon','MAE','gain_vs_g0_pct']].to_string(index=False))


if __name__ == '__main__':
    main()
