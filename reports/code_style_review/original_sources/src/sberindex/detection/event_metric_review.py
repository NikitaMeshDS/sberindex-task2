"""Independent labelled synthetic event benchmark; never modifies legacy artifacts.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
../venv/bin/python -m sberindex.detection.event_metric_review
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.optimize import linear_sum_assignment

from sberindex.paths import ROOT
from sberindex.detection.detector_settings import load_settings
from sberindex.detection.asof_detector_audit import causal_scores
from sberindex.detection.bocpd_audit import bocpd_scores

ONLINE = ('spike', 'rolling_3m', 'ewma', 'cusum', 'bocpd')
OFFLINE = ('pelt', 'kernelcpd')
CONFIG_PATH = ROOT / 'configs/event_metric_review.json'
OUT = ROOT / 'reports/event_metric_review'


def match_events(truth, predicted, tolerance=1):
    """Maximum-cardinality one-to-one match, then minimum total absolute error.

    Both dates are zero-based first changed/alert sample. Unmatched estimates
    count as FP, unmatched boundaries as FN. No terminal endpoint is an event.
    """
    truth, predicted = sorted(set(truth)), sorted(set(predicted))
    if not truth or not predicted:
        return []
    distance = np.abs(np.subtract.outer(truth, predicted))
    # A forbidden edge outweighs all admissible distances together.
    cost = np.where(distance <= tolerance, distance, (len(truth)+len(predicted)+1)*(tolerance+1))
    rows, cols = linear_sum_assignment(cost)
    return [(truth[i], predicted[j]) for i, j in zip(rows, cols) if distance[i, j] <= tolerance]


def alarm_events(alarm):
    """One event per contiguous threshold-exceedance episode; no lookahead."""
    alarm = np.asarray(alarm, bool)
    return np.flatnonzero(alarm & ~np.r_[False, alarm[:-1]]).tolist()


def forecast_residual(log_values, train_months=36):
    """Frozen seasonal month means fit only on the first 36 observations."""
    x = np.asarray(log_values, float)
    profile = np.array([x[:train_months][np.arange(train_months) % 12 == m].mean() for m in range(12)])
    return x[train_months:] - profile[np.arange(train_months, len(x)) % 12]


def synthetic_panel(seed, per_shape, config=None):
    config = json.loads(CONFIG_PATH.read_text()) if config is None else config
    rng = np.random.default_rng(seed)
    train, horizon = config['train_months'], config['evaluation_months']
    t = np.arange(train+horizon)
    panel = []
    for shape in ('no_change', 'step', 'pulse'):
        for i in range(per_shape):
            onset = int(rng.integers(6, 13))
            duration = int(rng.integers(4, 7))
            amp = float(config['amplitudes_log'][i % len(config['amplitudes_log'])]) * (-1 if i % 2 else 1)
            seasonal = rng.uniform(.08, .22)*np.sin(2*np.pi*t/12+rng.uniform(0, 2*np.pi))
            latent = np.full(len(t), rng.uniform(7, 9)) + seasonal
            truth = []
            if shape != 'no_change':
                latent[train+onset:] += amp
                truth = [onset]
                if shape == 'pulse':
                    latent[train+onset+duration:] -= amp
                    truth.append(onset+duration)
            x = latent+rng.normal(0, config['noise_sigma'], len(t))
            panel.append(dict(series_id=f'{seed}:{shape}:{i:04}', seed=seed, shape=shape,
                              amplitude_log=abs(amp) if shape != 'no_change' else 0.,
                              truth=truth, log_values=x, residual=forecast_residual(x, train)))
    return panel


def online_scores(signal, method, sigma):
    return bocpd_scores(signal, sigma, settings=load_settings()) if method == 'bocpd' else causal_scores(signal, method, load_settings())


def offline_events(signal, method, penalty, sigma, config=None):
    config = json.loads(CONFIG_PATH.read_text()) if config is None else config
    z = np.asarray(signal, float).reshape(-1, 1)/sigma
    if method == 'pelt':
        algo = rpt.Pelt(model='l2', min_size=config['offline_min_size'], jump=config['offline_jump'])
    elif method == 'kernelcpd':
        algo = rpt.KernelCPD(kernel='rbf', min_size=config['offline_min_size'], jump=config['offline_jump'],
                            params={'gamma': config['kernel_gamma']})
    else:
        raise ValueError(method)
    # ruptures endpoint k means [previous:k), next starts at zero-based k.
    return [int(k) for k in algo.fit(z).predict(pen=penalty) if k < len(z)]


def measure(panel, predictions, method, split, config):
    rows = []
    for series, pred in zip(panel, predictions):
        truth = series['truth']
        matches = match_events(truth, pred, config['tolerance_months'])
        matched_pred = {b for _, b in matches}
        tp, fp, fn = len(matches), len(pred)-len(matches), len(truth)-len(matches)
        # Offline result only available after observing the last evaluation sample.
        availability = [b if method in ONLINE else config['evaluation_months']-1 for _, b in matches]
        rows.append(dict(series_id=series['series_id'], seed=series['seed'], shape=series['shape'],
                         amplitude_log=series['amplitude_log'], method=method, split=split,
                         truth_count=len(truth), predicted_count=len(pred), tp=tp, fp=fp, fn=fn,
                         null_false_alarms=len(pred) if series['shape']=='no_change' else 0,
                         signed_errors=[b-a for a, b in matches],
                         detection_delays=[max(0, b-a) for a, b in matches] if method in ONLINE else [],
                         availability_delays=[b+config['reporting_delay_months']-a for (a, _), b in zip(matches, availability)],
                         truth=json.dumps(truth), predictions=json.dumps(pred), matches=json.dumps(matches),
                         unmatched_predictions=json.dumps([p for p in pred if p not in matched_pred])))
    return rows


def summarize(rows, horizon):
    frame = pd.DataFrame(rows)
    output = []
    for (split, method, shape), part in frame.groupby(['split', 'method', 'shape']):
        tp, fp, fn = (int(part[col].sum()) for col in ('tp', 'fp', 'fn'))
        errors = [x for values in part.signed_errors for x in values]
        delays = [x for values in part.detection_delays for x in values]
        availability = [x for values in part.availability_delays for x in values]
        null_n = int((part['shape']=='no_change').sum())
        output.append(dict(split=split, method=method, shape=shape, series_n=len(part),
                           mo_months=len(part)*horizon, true_events=tp+fn, tp=tp, fp=fp, fn=fn,
                           precision=tp/(tp+fp) if tp+fp else np.nan,
                           recall=tp/(tp+fn) if tp+fn else np.nan,
                           f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                           unmatched_predictions_per100_mo_month=100*fp/(len(part)*horizon),
                           null_false_alarms_per100_mo_month=100*part.null_false_alarms.sum()/(null_n*horizon) if null_n else np.nan,
                           mean_signed_localization_error_months=np.mean(errors) if errors else np.nan,
                           mean_absolute_localization_error_months=np.mean(np.abs(errors)) if errors else np.nan,
                           mean_detection_delay_months=np.mean(delays) if delays else np.nan,
                           mean_report_availability_delay_months=np.mean(availability) if availability else np.nan))
    # Pool all shapes with the same matching decisions; null denominator remains null only.
    for (split, method), part in frame.groupby(['split', 'method']):
        tp, fp, fn = (int(part[col].sum()) for col in ('tp', 'fp', 'fn'))
        errors = sum(part.signed_errors.tolist(), [])
        delays = sum(part.detection_delays.tolist(), [])
        availability = sum(part.availability_delays.tolist(), [])
        null = part[part['shape']=='no_change']
        output.append(dict(split=split, method=method, shape='all', series_n=len(part), mo_months=len(part)*horizon,
                           true_events=tp+fn, tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else np.nan,
                           recall=tp/(tp+fn) if tp+fn else np.nan, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                           unmatched_predictions_per100_mo_month=100*fp/(len(part)*horizon),
                           null_false_alarms_per100_mo_month=100*null.null_false_alarms.sum()/(len(null)*horizon) if len(null) else np.nan,
                           mean_signed_localization_error_months=np.mean(errors) if errors else np.nan,
                           mean_absolute_localization_error_months=np.mean(np.abs(errors)) if errors else np.nan,
                           mean_detection_delay_months=np.mean(delays) if delays else np.nan,
                           mean_report_availability_delay_months=np.mean(availability) if availability else np.nan))
    return pd.DataFrame(output)


def main():
    config = json.loads(CONFIG_PATH.read_text()); OUT.mkdir(parents=True, exist_ok=True)
    sigma, horizon = config['noise_sigma'], config['evaluation_months']
    calibration = [r for r in synthetic_panel(config['calibration_seed'], config['calibration_null_n'], config) if r['shape']=='no_change']
    cal_signal = np.stack([r['residual'] for r in calibration])
    parameters, calibration_rows = {}, []
    for method in ONLINE:
        scores = online_scores(cal_signal, method, sigma)
        threshold = float(np.quantile(np.nanmax(scores, axis=1), load_settings()['threshold_quantile']))
        parameters[method] = threshold
        events = [alarm_events(s > threshold) for s in scores]
        calibration_rows += measure(calibration, events, method, 'calibration', config)
    penalty_rows = []
    for method in OFFLINE:
        for penalty in config[f'{method}_penalties']:
            events = [offline_events(r['residual'], method, penalty, sigma, config) for r in calibration]
            rate = 100*sum(map(len, events))/(len(calibration)*horizon)
            penalty_rows.append(dict(method=method, penalty=penalty, null_false_alarms_per100_mo_month=rate,
                                     passes_budget=rate <= config['null_false_alarms_budget_per100_mo_month']))
        passing = [p for p in penalty_rows if p['method']==method and p['passes_budget']]
        if not passing:
            raise RuntimeError(f'{method}: no penalty meets prespecified null budget')
        parameters[method] = min(p['penalty'] for p in passing)
        events = [offline_events(r['residual'], method, parameters[method], sigma, config) for r in calibration]
        calibration_rows += measure(calibration, events, method, 'calibration', config)
    all_rows = list(calibration_rows); panel_rows = []; examples = None
    for split, seeds in [('selection', [config['selection_seed']]), ('evaluation', config['evaluation_seeds'])]:
        for seed in seeds:
            panel = synthetic_panel(seed, config['series_per_shape'], config)
            if examples is None and split == 'evaluation': examples = panel
            signal = np.stack([r['residual'] for r in panel])
            for r in panel:
                for t, residual in enumerate(r['residual']):
                    panel_rows.append(dict(split=split, series_id=r['series_id'], shape=r['shape'], amplitude_log=r['amplitude_log'],
                                           evaluation_month=t, residual_log=residual, true_boundary=t in r['truth']))
            for method in ONLINE+OFFLINE:
                events = ([alarm_events(s > parameters[method]) for s in online_scores(signal, method, sigma)] if method in ONLINE
                          else [offline_events(r['residual'], method, parameters[method], sigma, config) for r in panel])
                all_rows += measure(panel, events, method, split, config)
    summary = summarize(all_rows, horizon)
    selection = summary[(summary.split=='selection') & (summary['shape']=='all') & summary.method.isin(ONLINE)]
    eligible = selection[(selection.null_false_alarms_per100_mo_month <= config['null_false_alarms_budget_per100_mo_month']) &
                         (selection.recall >= config['minimum_selection_recall'])]
    selected = None if eligible.empty else eligible.sort_values(['f1', 'method'], ascending=[False, True]).iloc[0].method
    choice = dict(synthetic_profile_selected_online_method=selected, selection_rule='null FA <= budget, pooled ±1-month recall >= minimum, then highest F1; ties alphabetic',
                  engineering_default='spike', engineering_default_changed=False,
                  reason='Separate artificial event-localization profile is not a validation of real municipal operating performance.')
    serial = pd.DataFrame(all_rows)
    for col in ('signed_errors', 'detection_delays', 'availability_delays'): serial[col] = serial[col].map(json.dumps)
    serial.to_csv(OUT/'series_metrics.csv', index=False)
    summary.to_csv(OUT/'comparison.csv', index=False)
    pd.DataFrame(panel_rows).to_parquet(OUT/'synthetic_residuals.parquet', index=False)
    pd.DataFrame(penalty_rows).to_csv(OUT/'offline_penalty_calibration.csv', index=False)
    sources = ['configs/event_metric_review.json', 'configs/detectors.json', 'src/sberindex/detection/event_metric_review.py',
               'src/sberindex/detection/asof_detector_audit.py', 'src/sberindex/detection/bocpd_audit.py',
               'src/sberindex/detection/detector_settings.py', 'tests/test_event_metric_review.py', 'requirements-lock.txt']
    protocol = dict(config=config, parameters=parameters, detector_settings=load_settings(), choice=choice,
                    matching='max cardinality one-to-one, minimum total absolute error, ±1 month inclusive; zero-based first changed sample',
                    false_alarms='All predictions on genuine synthetic null series. FP on changed series also reported separately as unmatched predictions.',
                    online_availability='source sample t + 1 assumed reporting month',
                    offline_availability='full evaluation sequence available at sample 23 + 1 assumed reporting month; localization is distinct',
                    forecast='Frozen 12 month-of-year means trained on 36 pre-evaluation log observations; no trend, no future fit, iid noise .03',
                    labels='null: none; step: first shifted month; pulse: onset and first reverted month. Only simulated signal boundaries.',
                    versions={p: importlib.metadata.version(p) for p in ['ruptures', 'numpy', 'scipy', 'pandas', 'matplotlib']},
                    input_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                    output_counts=dict(series_metric_rows=len(all_rows), residual_rows=len(panel_rows), comparison_rows=len(summary)))
    (OUT/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    (OUT/'choice.json').write_text(json.dumps(choice, ensure_ascii=False, indent=2)+'\n')
    draw(summary, examples, parameters, config)
    write_report(summary, protocol)
    print(summary[(summary.split=='evaluation') & (summary['shape']=='all')].to_string(index=False))
    print(json.dumps(choice, ensure_ascii=False))


def draw(summary, examples, parameters, config):
    tab = summary[(summary.split=='evaluation') & (summary['shape']=='all')].set_index('method').loc[list(ONLINE+OFFLINE)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    colors = ['#236b8e']*len(ONLINE)+['#c67d32']*len(OFFLINE)
    for m, row, c in zip(tab.index, tab.itertuples(), colors):
        axes[0].scatter(row.null_false_alarms_per100_mo_month, row.recall, color=c, s=65)
        offsets = {'cusum': (-36, 15), 'spike': (6, -17), 'ewma': (8, 12)}
        axes[0].annotate(m, (row.null_false_alarms_per100_mo_month, row.recall), xytext=offsets.get(m, (4, 5)), textcoords='offset points', fontsize=9)
    axes[0].axvline(config['null_false_alarms_budget_per100_mo_month'], linestyle=':', color='gray')
    axes[0].axhline(config['minimum_selection_recall'], linestyle=':', color='gray')
    axes[0].set(xlabel='False alarms / 100 null MO-months', ylabel='Boundary recall (±1 month)', title='Held-out synthetic event tradeoff', ylim=(-.03, 1.08))
    axes[1].barh(tab.index, tab.mean_report_availability_delay_months, color=colors)
    axes[1].set(xlabel='Months after matched true boundary', title='Report availability: conditional on matches')
    pulse = next(r for r in examples if r['shape']=='pulse' and r['amplitude_log']==.12)
    axes[2].plot(pulse['residual'], color='#35434c', label='Seasonal forecast residual')
    for i, t in enumerate(pulse['truth']): axes[2].axvline(t, color='#ad3232', linestyle='--', label='True boundary' if i==0 else None)
    for i, method in enumerate(OFFLINE):
        ev = offline_events(pulse['residual'], method, parameters[method], config['noise_sigma'], config)
        axes[2].scatter(ev, [(.18 if i==0 else -.15)]*len(ev), marker='v' if i==0 else '^', label=method, color=colors[-2+i])
    axes[2].set(xlabel='Evaluation month (zero based)', ylabel='Log residual', title='Fixed first pulse, amplitude 0.12')
    axes[2].legend(fontsize=8)
    fig.suptitle('Labelled seasonal synthetic benchmark • online blue, retrospective orange', fontsize=13)
    fig.savefig(OUT/'event_comparison.png', dpi=170)
    fig.savefig(OUT/'event_comparison.svg')
    plt.close(fig)


def write_report(summary, protocol):
    config = protocol['config']
    evaluation = summary[(summary.split=='evaluation') & (summary['shape']=='all')].set_index('method').loc[list(ONLINE+OFFLINE)]
    table = ['| Метод | TP/FP/FN | Precision | Recall | F1 | Ложные /100 нулевых МО-месяцев | Задержка обнаружения | Доступность отчёта |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, r in evaluation.iterrows():
        delay = f"{r.mean_detection_delay_months:.3f}" if method in ONLINE else '—'
        table.append(f"| {method} | {int(r.tp)}/{int(r.fp)}/{int(r.fn)} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.null_false_alarms_per100_mo_month:.3f} | {delay} | {r.mean_report_availability_delay_months:.3f} |")
    text = """# Единое сравнение событий: онлайн и ретроспективные ориентиры

Дата: 2026-10-06. Это независимый размеченный синтетический опыт. Он не переименовывает прежние контрольные тревоги на реальных данных в ложные. Реальные границы изменения расходов независимо не размечены, поэтому реальные precision/recall/F1 здесь не заявлены.

## Протокол до оценки

Конфигурация: `configs/event_metric_review.json`. Детекторы spike, rolling_3m, EWMA, CUSUM и BOCPD вызывают прежние реализации с неизменными параметрами `configs/detectors.json`. Их новые пороги рассчитаны на независимых 200 нулевых рядах: 0,99-квантиль максимума оценки за 24 месяца. Эти пороги относятся только к этому опыту и не заменяют рабочую калибровку.

PELT действительно вызывает `ruptures.Pelt(model='l2', min_size=2, jump=1)`, KernelCPD — `ruptures.KernelCPD(kernel='rbf', min_size=2, jump=1, params={'gamma':0.5})`. На том же остатке, делённом на известную синтетическую sigma=0,03, выбирается минимальный штраф из фиксированной сетки, который даёт ≤0,2 ложных событий на 100 нулевых МО-месяцев калибровки. Количество истинных границ офлайн алгоритмам не сообщается. Выбранные штрафы и полная сетка записаны в protocol.json и offline_penalty_calibration.csv. Ретроспективный ориентир не является математической верхней границей качества онлайн алгоритмов.

В каждом ряду 36 обучающих и 24 оцениваемых месяца. Генератор задаёт индивидуальный уровень, годовую синусоидальную сезонность и iid Gaussian шум sigma=0,03 в логарифме. Замороженный прогноз — среднее каждого календарного месяца по трём обучающим годам. Остаток — факт минус этот прогноз; ни профиль, ни онлайн оценки, ни порог не используют будущие оцениваемые факты. Выборка полностью синтетическая, без подстановки реальной панели. Неизвестная sigma, тренд, коррелированный шум и пропуски не проверялись.

Три класса равного размера: null без границ, step с одной границей, pulse с двумя (начало и первый месяц возврата). Начало равномерно от месяца 6 до 12 включительно; pulse длится 4–6 месяцев. Сдвиги ±0,06/0,12/0,18 в логарифме распределены поровну (положительные примерно +6,2/+12,7/+19,7%). Перед всеми сдвигами есть спокойный участок; синтетические метки известны по генератору, а не по внешнему событию.

Отдельные seed: калибровка 2026100601, выбор 2026100602 (360 рядов), оценка 2026100603–05 (1080 рядов). Это независимые шумовые реализации одного генератора, а не независимые реальные события/временные периоды. На финальной оценке 360 null, 360 step, 360 pulse; 1080 истинных границ, 25920 МО-месяцев, из них 8640 нулевых. На каждого из семи методов приходится одна и та же выборка.

## Правило метрик

Месяцы нумеруются с нуля. Граница ruptures k — первый элемент нового сегмента k; обязательный конечный endpoint 24 удаляется. Онлайн событие — первый месяц непрерывного эпизода превышения порога; новый эпизод возможен после снижения ниже порога. Никакого взгляда вперёд для формирования онлайн события нет.

Сопоставление в пределах ±1 месяца включительно: максимум числа однозначных пар, затем минимум суммарного абсолютного расстояния. Одна граница и одна тревога участвуют максимум в одной паре. TP — пары; FP — все несопоставленные предсказания, FN — несопоставленные истинные границы. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Нулевая неопределённая дробь записывается NaN. Дополнительные тревоги на устойчивом сдвиге считаются FP для этого задания локализации границ; это не равнозначно ненужности операционного предупреждения о сохраняющейся аномалии.

«Ложные /100» в таблице — все предсказания на заведомо нулевых рядах, делённые на их 8640 МО-месяцев и умноженные на 100. Несопоставленные предсказания на изменённых рядах также входят в FP; их общая плотность отдельно хранится в comparison.csv. Так знаменатель ложных тревог не смешивается с числом событий.

Задержка обнаружения = среднее max(0, дата онлайн тревоги − истинная дата) только по совпавшим событиям; ранняя тревога в пределах допуска имеет задержку 0. Знаковая ошибка локализации и абсолютная ошибка представлены отдельно в CSV. Допуск ±1 ограничивает эту условную задержку; пропуски не получают искусственно малую задержку и остаются FN. Для офлайн метода задержка обнаружения не определена и указана «—».

Доступность отчёта — среднее по тем же совпавшим событиям с условной задержкой публикации данных 1 месяц. Онлайн: месяц тревоги +1 минус истинная дата; офлайн: конец полной последовательности (23)+1 минус истинная дата. Хорошая офлайн локализация не означает своевременное предупреждение. Сравнение задержек условно по разным множествам совпавших событий, что надо учитывать вместе с recall.

## Итоги на финальной оценке

""" + '\n'.join(table) + """

![Сравнение и доступность](event_comparison.png)

Полная таблица comparison.csv также разделяет no_change/step/pulse. Машинное имя no_change сохраняет нулевой класс при обычном чтении CSV в pandas. series_metrics.csv содержит каждую истинную/предсказанную дату, пары и несопоставленные предсказания; synthetic_residuals.parquet — остатки и границы selection/evaluation. Для проверки отсутствия будущих данных используются тесты изменения будущих значений и инвариантности прошлых оценок, включая BOCPD.

## Выбор и границы вывода

Заранее заданный профиль онлайн выбора: ≤0,2 ложных события/100 нулевых МО-месяцев, pooled recall ≥0,50 при ±1 месяце; среди выполнивших — максимальный F1 на selection, без использования evaluation. Ни один онлайн метод не прошёл профиль: результат выбора `null`, ограничения несовместимы с проверенными фиксированными методами на этом генераторе. Нельзя задним числом ослаблять полноту для объявления победителя. На финальной оценке лучший F1 среди онлайн имеет CUSUM; это описательный результат, а не выбранный по финальным данным рабочий детектор.

Рабочий инженерный spike сохранён. Его прежний профиль (обнаружение сдвига за 3 месяца и контрольный объём тревог) отличается от текущего профиля локализации каждой границы ±1 месяц, включая возврат pulse. Слабый BOCPD здесь отражает сохранённые приоры/калибровку; это не доказательство слабости всех вариантов BOCPD. Результат KernelCPD зависит от фиксированных gamma и штрафной сетки. Новая настройка на финальной оценке не проводилась.

Нужны независимо размеченные реальные границы расходов, настоящие винтажи доступности и проверка на иных процессах шума/сезонности перед переносом чисел в эксплуатацию. Метрики новостей как предвестников этим опытом не закрыты.

## Воспроизведение и происхождение

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m sberindex.detection.event_metric_review
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src ../venv/bin/python -m unittest discover -s tests -p test_event_metric_review.py
```

ruptures 1.1.10 уже закреплён в requirements-lock.txt. protocol.json сохраняет параметры, seed, версии и SHA256 исходных файлов; CPU потоки ограничены двумя, запуск последовательный. Обновляются только новые файлы reports/event_metric_review.

Официальные описания: [PELT](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/) и [KernelCPD](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/kernelcpd/). Эти страницы использованы для проверки настоящих API и смысла офлайн сегментации; численные результаты получены локально.
"""
    (OUT/'REPORT.md').write_text(text)

if __name__ == '__main__':
    main()
