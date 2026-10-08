# Конфигурация и гиперпараметры

[Главная](../README.md) · [Запуск](GETTING_STARTED.md)

## Исполняемый config.json

[config.json](../config.json) читается модулями прогнозирования и исходного синтетического сравнения. Значения ниже относятся к текущему файлу.

| Ключ | Значение | Назначение |
|---|---|---|
| `random_seed` | 20260930 | Воспроизводимый отбор и модели |
| `prophet_chronos_sample_size` | 256 | Размер исходной выборки прогнозирования |
| `forecast_horizons_months` | `[1,3,6,12]` | Горизонты исходного benchmark |
| `chronos_model`, `chronos2_model` | Chronos-Bolt Tiny / Chronos-2 | Идентификаторы весов |
| `chronos_revision`, `chronos2_revision` | Закреплённые SHA в JSON | Версии весов |
| `hgb_max_iter` | 250 | Итерации HGB |
| `hgb_max_leaf_nodes` | 31 | Листья дерева |
| `hgb_learning_rate` | 0.05 | Скорость обучения |
| `hgb_min_samples_leaf` | 60 | Минимальный размер листа |
| `hgb_l2_regularization` | 2 | Регуляризация |
| `blend_candidate_seasonal_weights` | `[0,.25,.5,.75,1]` | Кандидаты раннего выбора смеси |
| `shift_sizes_pct` | `[10,20]` | Исходные размеры синтетических изменений |
| `alarm_calibration_quantiles` | `[.95,.99]` | Исходные пороги детекторов |
| `synthetic_shift_start_month` | `2024-07` | Начало исходного синтетического воздействия |

Изменение JSON влияет на читающие его этапы. Уже сохранённые прогнозы автоматически не становятся результатом новой конфигурации: provenance-проверки могут отклонить несовместимые входы. Новое исследование требует отдельного протокола, пересчёта соответствующих этапов и регистрации новых артефактов.

## Исполняемый configs/detectors.json

[configs/detectors.json](../configs/detectors.json) реально читается `asof_detector_audit`, `detector_delay_audit` и `bocpd_audit`. Загрузчик `detection/detector_settings.py` проверяет точный набор ключей и диапазоны; опечатки, bool, NaN и Infinity отвергаются.

| Ключ | Значение | Допустимо / роль |
|---|---:|---|
| `ewma_alpha` | 0.45 | (0,1), вес текущего наблюдения; предыдущий вес 1−alpha |
| `cusum_drift` | 0.025 | ≥0, вычитаемый дрейф CUSUM |
| `noise_individual_weight` | 0.5 | (0,1), доля индивидуального MAD; остальное pooled MAD |
| `noise_floor` | 0.0001 | >0, нижняя граница масштаба |
| `threshold_quantile` | 0.99 | (0,1), квантиль калибровочных максимумов МО |
| `bocpd_hazard` | 1/12 в JSON как число | (0,1), вероятность новой границы |
| `bocpd_mu` | 0 | Любое конечное число, prior mean |
| `bocpd_kappa` | 1 | >0, сила prior mean |
| `bocpd_alpha` | 2 | >1, форма NIG prior |
| `bocpd_sigma_floor` | 0.02 | >0, минимум эмпирического масштаба |
| `bocpd_mad_factor` | 1.4826 | >0, множитель медианного MAD |

BOCPD использует beta0=sigma²; sigma оценивается только по калибровке. Ключи в JSON плоские, дополнительные поля запрещены. Для стандартного запуска достаточно изменить этот файл и запустить нужный модуль; параметры записываются в JSON протокола расчёта вместе с SHA256 файла/loader. Настройки не изменяют сохранённые прогнозы тяжёлых моделей.

```bash
PYTHONPATH=src python -m sberindex.detection.asof_detector_audit
PYTHONPATH=src python -m sberindex.detection.detector_delay_audit
PYTHONPATH=src python -m sberindex.detection.bocpd_audit
```

Это отдельные новые опыты при изменённых параметрах. Прежние статические отчёты и рисунки относятся к зафиксированной исследовательской конфигурации; их нельзя автоматически подписывать новым результатом. Исторические `change_detection`/`event_attribution_audit` и кэш `change_robustness` используют прежнюю постановку и не перенастраиваются этим JSON. Полный refresh связывает ряд таблиц именно в эталонной постановке; для нового варианта необходимо обновить соответствующий протокол/интерпретацию. Месячные окна, synthetic seeds/shapes и rolling3 в этом переносе фиксированы.

## Параметры зафиксированных аудитов

Параметры алгоритмов трёх поздних detector-аудитов вынесены в отдельный исполняемый JSON. Не все остальные экспериментальные параметры централизованы в `config.json`. Ниже — фактические константы и их документы, без утверждения, что изменение JSON перенастроит эти опыты.

| Эксперимент | Фиксированные условия | Источник |
|---|---|---|
| As-of детекторы | 10 ID-стабильных назначений, ±20%, step/pulse/ramp | [DETECTOR_ASOF_EXPERIMENT](protocols/DETECTOR_ASOF_EXPERIMENT.md), `detection/asof_detector_audit.py` |
| Календарные задержки | Лаги 0/1/2, декабрьское цензурирование | [DETECTOR_DELAY_EXPERIMENT](protocols/DETECTOR_DELAY_EXPERIMENT.md), `detection/detector_delay_audit.py` |
| BOCPD | Hazard=1/12; NIG prior; общий q99 за Feb–Jun | [BOCPD_EXPERIMENT](protocols/BOCPD_EXPERIMENT.md), `detection/bocpd_audit.py` |
| Сезонное нормирование | Профиль только 2023; CPU, batch64, два torch-потока | [FOUNDATION_SEASONAL_EXPERIMENT](protocols/FOUNDATION_SEASONAL_EXPERIMENT.md), `forecasting/foundation_seasonal_audit.py` |
| Региональные новости | Минимум 3 разных публикации и 3 покрытых месяца в каждой части | [REGIONAL_NEWS_EXPERIMENT](protocols/REGIONAL_NEWS_EXPERIMENT.md), `external/regional_news_experiment.py` |

Пути модулей в таблице относятся к `src/sberindex/`. Полный состав условий каждого опыта хранится в протоколе Markdown, коде и соответствующем JSON в `results/`, защищённом SHA256.

## Манифесты

- [data_sources.json](../data_sources.json): URL, периоды, роль, дата получения, SHA256 и размер исходных снимков.
- [refresh_inputs.json](../refresh_inputs.json): 22 сохранённых модельных входа и их происхождение.
- [run_metadata.json](../reports/run_metadata.json): фактически исполненная версия кода и среда последнего вычислительного запуска.

Централизация остальных констант прогнозных/новостных аудитов и фиксированного дизайна экспериментов — отдельная задача инженерной части. Она должна сохранять текущие численные результаты и пройти повторную provenance-проверку.

Проверка переноса при исходных параметрах: [протокол](protocols/DETECTOR_CONFIGURATION_MIGRATION.md), [машинное сравнение](../reports/detector_configuration_migration.json). Все 194 CSV/Parquet и 58 PNG/SVG совпали с предыдущим коммитом побайтно; настройки не выбирались по результатам 2024 года.

## Исторический configs/operational_workflow.json (в исследовательском архиве)

Читается tools/operational_workflow.py: горизонты, лаги, конец обучения, вес смеси, демонстрационное решение; категория/масштаб/лаг/форма/срок детектора, бюджет, минимальная чувствительность и две непересекающиеся группы назначений. [Правило](protocols/OPERATIONAL_WORKFLOW.md). Это ранний эксперимент со смесью 75/25, а не итоговый алгоритм. Для новых прогнозов выбранной модели используйте `predict.py` и `configs/predict.json`. Гипотетические лаги не заменяют фактические даты публикации.

## Фиксированные расширения после внешнего отзыва, 6 октября 2026

| Конфигурация | Что управляет |
|---|---|
| `configs/prophet_seasonality.json` | Два усиленных варианта Prophet; неизменная 256-ID когорта |
| `configs/full_cohort_review.json` | Три Prophet-варианта на всей пригодной по 2023 когорте, четыре горизонта |
| `configs/foundation_covariate_review.json` | Ревизии Chronos-2/Bolt-base, явные группы, известные будущие профили, batch/потоки |
| `configs/event_metric_review.json` | Синтетические классы, отдельные seed, допуск, бюджеты, штрафы PELT/KernelCPD |
| `configs/hierarchical_review.json` | Фиксированные веса регионального профиля и fallback малых групп |
| `configs/direct_horizon_review.json` | Отдельные direct HGB, rolling training, запрет несуществующих h12-пар |
| `configs/spatial_review.json` | Ретроспективный граф, пять ближайших соседей, прошлые признаки и proxy-тревоги |
| `configs/review_real_cases.json` | Предварительно выбранные официальные ID/публикации; отсутствие ряда не заменяется прокси |

Новости: `data/external/regional_news_review/protocol.json` фиксирует доступность, окно, источники и модель до расчёта ошибок. Часть вариантов описана протоколами в соответствующих `reports/*/PREDECLARED_PROTOCOL.md`.

`run_review.py --mode verify` проверяет замороженные исходники, источники, параметры и численные результаты. `--mode recompute` запускает реальные обучения/инференс всех расширений. `--mode record` служит явной фиксацией проверенного снимка: неполные блоки или расходящиеся заявленные хэши отклоняются. Фиксация нового снимка не называется воспроизведением старых чисел.

Новый опыт: `configs/growth_bridge_review.json` — вес региона 0,5, замороженный опубликованный YoY номинальной зарплаты Росстата и число переходов года. [Протокол](protocols/GROWTH_BRIDGE_REVIEW.md).


## Год истории и сроки новостей

`configs/short_history_review.json`: 12 месяцев обучения / 12 оценки, отдельные seed калибровки/отбора/оценки, диапазоны начала и длительности, null бюджет 0,2 / 100 МО-месяцев и общая калибровка CUSUM + spike. `reports/short_history_review/choice.json` сохраняет выбранный EWMA и правило отбора.

`news_alert_lead.py` использует выбранный метод и закреплённые пороги, готовые h1 прогнозы `hierarchy05_growth_bridge`, сценарии задержки выпуска 0/1/2 месяца. Он не подбирает пороги по новостным событиям. Сетевой сбор RSS отделён: `python -m sberindex.external.mchs_rss --start 2023-01-01 --end 2024-12-31 --timeout 8`. Изменение параметров создаёт новый опыт и требует обновления протокола, а не подмены старого снимка.

## Последние проверки чувствительности и локального сигнала

- `configs/growth_sensitivity_review.json`: фиксированная диагностическая сетка ставок, ноябрьский ИПЦ 7,48%, те же пары и региональный вес 0,5; выбор новой ставки отключён.
- `configs/peer_residual_review.json`: h1, минимум пять других МО региона, четыре замороженных детектора, кейсы Орск/Оренбург, отдельные даты события/публикации/сценарной доступности и лаги выпуска 0/1/2.

Источники новых проверок: `data/external/growth_sensitivity_review/sources.json`, `data/external/peer_residual_review/mfc_support_page.json`; версии и SHA — в аудитах соответствующих папок `reports/`.

### Новостной интервал

[`news_interval_review.json`](../configs/news_interval_review.json): фиксированный радиус ×1,5 для доступного сообщения, прежний точечный прогноз и номинальные 80% границы. Это диагностическая гипотеза, не настройка выбранной модели. [Результаты](../reports/news_interval_review/REPORT.md).

## Конфигурация рабочего прогноза

[predict.json](../configs/predict.json) — входные таблицы, дата расчёта, год сезонного профиля, горизонты, региональный вес и ставка роста для новых данных. [Подробная инструкция](PREDICT_NEW_DATA.md).
