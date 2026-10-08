# Документация

[Главная страница](../README.md) · [Запуск](GETTING_STARTED.md)

## Для первого знакомства

Начните с [краткого отчёта](research/SUBMISSION_REPORT.md) и [комплекта для сдачи](SUBMISSION.md).

| Материал | Содержание |
|---|---|
| [Обзор исследования](research/PROJECT_OVERVIEW.md) | Вопрос, основные результаты и допустимые выводы |
| [Методология на русском](research/METHODOLOGY.md) | Данные, сравнения, метрики, интерпретация и ограничения |
| [Критерии конкурса](COMPETITION.md) | Семь требований, веса, доказательства и открытые задачи |
| [Архитектура](ARCHITECTURE.md) | Ответственность модулей и поток расчётов |
| [Запуск](GETTING_STARTED.md) | Установка, refresh, full, тесты и воспроизводимость |
| [Конфигурация](CONFIGURATION.md) | Параметры JSON и зафиксированные условия экспериментов |
| [Источники и использование данных](DATA_USAGE.md) | Роли данных, доступность и условия использования |

## Исследовательские отчёты

| Вопрос | Отчёт | Численные результаты |
|---|---|---|
| Сохраняется ли результат против сезонного Prophet? | [PROPHET_SEASONALITY_REPORT](research/PROPHET_SEASONALITY_REPORT.md) | `reports/prophet_seasonality/*` |
| Можно ли выровнять покрытие между группамиМО? | [GROUPED_INTERVAL_REPORT](research/GROUPED_INTERVAL_REPORT.md) | `reports/grouped_intervals/*` |
| Можно ли улучшить прогнозные интервалы? | [INTERVAL_CALIBRATION_REPORT](research/INTERVAL_CALIBRATION_REPORT.md) | `reports/interval_calibration/*` |
| Насколько убедительны результаты? | [STATISTICAL_EVIDENCE_REPORT](research/STATISTICAL_EVIDENCE_REPORT.md) | `reports/statistical_evidence/*` |
| Помогают ли поправки и новые признаки? | [RESIDUAL_FEATURE_REPORT](research/RESIDUAL_FEATURE_REPORT.md) | `reports/residual_features/*` |
| Какие новые официальные данные пригодны? | [NEW_SOURCES_AUDIT](research/NEW_SOURCES_AUDIT.md) | `reports/new_sources_audit.json` |
| Все критерии и доказательства | [CRITERIA_EVIDENCE](research/CRITERIA_EVIDENCE.md) | Ссылки на таблицы и протоколы внутри |
| Помогает ли сезонный профиль фундаментальным моделям? | [FOUNDATION_SEASONAL_REPORT](research/FOUNDATION_SEASONAL_REPORT.md) | `results/foundation_seasonal_*` |
| Где сосредоточены улучшения и ошибки? | [ASOF_DISTRIBUTION_REPORT](research/ASOF_DISTRIBUTION_REPORT.md) | `results/asof_distribution_*` |
| Что меняет задержка публикации для прогнозов? | [FOUNDATION_DELAY_REPORT](research/FOUNDATION_DELAY_REPORT.md) | `results/foundation_delay_*` |
| Сохраняется ли поздний выигрыш на ранних датах? | [OPERATIONAL_EARLY_REPORT](research/OPERATIONAL_EARLY_REPORT.md) | `results/operational_early_*` |
| Когда тревога реально может быть получена? | [DETECTOR_DELAY_REPORT](research/DETECTOR_DELAY_REPORT.md) | `results/detector_delay_*` |
| Как выбрать метод под срок и нагрузку? | [DETECTOR_DECISION_REPORT](research/DETECTOR_DECISION_REPORT.md) | `reports/detector_tradeoffs/*` |
| Что добавляет BOCPD? | [BOCPD_REPORT](research/BOCPD_REPORT.md) | `results/bocpd_*` |
| Какие официальные события сопоставлены с МО? | [REAL_EVENT_REGISTRY_REPORT](research/REAL_EVENT_REGISTRY_REPORT.md) | `results/real_registry_*` |

Пути `results/` относятся к корню репозитория. Научные рисунки — в [figures](../figures/), каждый в PNG и SVG.

## Протоколы и развитие

- [Следующий независимый временной тест](protocols/NEXT_HOLDOUT_PROTOCOL.md).
- [Гипотезы и выполненные циклы](protocols/RESEARCH_BACKLOG.md).
- [Протоколы экспериментов](protocols/): условия фиксируются до новых расчётов.
- [Правила разработки](../.github/CONTRIBUTING.md).

## Журнал и исторические материалы



## Рабочий выпуск

[Задержанные модели, выпуск прогноза и выбор детектора](research/OPERATIONAL_WORKFLOW_REPORT.md) · [Протокол](protocols/OPERATIONAL_WORKFLOW.md) · [Настройки](../configs/operational_workflow.json).



## Материалы для жюри

[Презентация PDF](../artifacts/presentation.pdf) · [Редактируемые слайды](../artifacts/presentation.pptx) · [Краткий отчёт](research/SUBMISSION_REPORT.md) · [Комплект](SUBMISSION.md).

История рабочих планов и внутренних отзывов сохранена в ветке `archive/research-history-20261006`. Научные протоколы, включая отрицательные результаты, остаются в основной ветке.
