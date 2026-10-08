# Установка и воспроизведение

[Главная](../README.md) · [Конфигурация](CONFIGURATION.md) · [Архитектура](ARCHITECTURE.md)

## Самый быстрый путь для жюри

Из корня распакованного **полного архива**:

```bash
python3.12 run_review.py --mode verify
```

Нужен только Python 3.12: этот режим использует стандартную библиотеку, не требует pip, GPU или весов. Обычно проверка занимает секунды. Успех — JSON `"status": "passed"`, `"mode": "verify"`, положительный `"files_checked"`. Проверяются сохранённые хеши; MAE не пересчитывается. Эталонные **932 / 1272 / 1371 / 1321** находятся в [README](../README.md#результаты-прогноза) и [CSV](../reports/growth_bridge_review/summary.csv).

Для просмотра ничего устанавливать не требуется: откройте [dashboard/index.html](../dashboard/index.html). Для тестов и вычислительных режимов установите среду ниже. Актуальная проверка чистого снимка — **189 тестов**; число защищённых файлов указано в аудите: [аудит](../reports/reviewer_final_packaging/clean_snapshot.json). Числа файлов меняются при обновлении упаковки. Приведённые далее 43/48/51/54 теста и полные временные замеры относятся к отмеченным историческим запускам.

## 1. Подготовить среду

Проверенная платформа — Python 3.12 на Apple Silicon. `requirements-lock.txt` фиксирует полную установленную среду; `requirements.txt` — основные прямые зависимости. Переносный refresh проверен в другой директории на той же машине. Новая чистая установка текущей версии на Linux/Windows отдельно не подтверждена.

```bash
git clone https://github.com/NikitaMeshDS/sberindex-task2.git
cd sberindex-task2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Публичный репозиторий клонируется без авторизации. Все следующие команды выполняются из корня клона с активной средой.

## 2. Восстановить текущие результаты

```bash
python run_pipeline.py --mode refresh
```

Последний подтверждённый вычислительный запуск:51 этап,44 основные проверки и проверки приложений. Единая команда также пересчитывает задержанные HGB/Prophet, прикладной выбор детектора и демонстрационный выпуск. Команда проверяет SHA256 исходных снимков и 22 сохранённых модельных входов, пересчитывает производные таблицы и графики. Основные Prophet/Chronos читаются из сохранённых прогнозов; некоторые небольшие HGB и Prophet для сценария задержки обучаются заново. Это воспроизведение свидетельств текущей версии из явно объявленных входов.

| Где смотреть | Что находится |
|---|---|
| [results](../results/) | Индивидуальные прогнозы, метрики, сигналы и протоколы |
| [figures](../figures/) | 29 научных графиков, PNG и SVG |
| [verification.json](../results/verification.json) | Итоги проверок реализации и хронологии |
| [run_metadata.json](../reports/run_metadata.json) | Этапы, коды выхода, среда, SHA256 исходников |
| `run_logs/` | Логи этапов; локальные `.log` не отслеживаются Git |

## 3. Проверить тесты и переносимость

```bash
PYTHONPATH=src python -m unittest discover -s tests
python reproduce_check.py
```

43 теста проверяют причинные инварианты, пропуски, парность фактов, календарное время и posterior BOCPD. Переносная команда создаёт временную копию в другом абсолютном пути, пересчитывает материалы, сравнивает результаты и проверяет сохранность модельных входов.

Последняя проверка в новой установленной среде:201/201 основных CSV/Parquet/JSON,31 файл приложений и 29 PNG/29 SVG совпали побайтно;22 входа не изменились. [Машинный отчёт](../reports/workflow_clean_reproduction.json), [64 проверенные версии среды](../reports/workflow_environment.json). Тесты и переносимость подтверждают реализацию; научную обобщаемость проверяет новый независимый период.

## 4. Полностью переобучить модели

```bash
python run_pipeline.py --mode full
```

Этот режим дополнительно запускает тяжёлые прогнозные этапы и обновляет манифест входов после успешного завершения. Потребуются веса Chronos с ревизиями из [config.json](../config.json); при первом запуске — загрузка из Hugging Face. Дополнительные сезонные/delay inference-эксперименты используют `local_files_only=True`, поэтому соответствующие веса должны быть в локальном кэше.

Полное обучение повторялось на предыдущей замороженной версии. Для текущего состояния подтверждены refresh и переносная пересборка; новый full не следует считать выполненным до появления соответствующих успешных логов и проверки. Время полного обучения зависит от оборудования и кэша весов; фиксированной оценки не заявляем.

## Отдельные команды

```bash
PYTHONPATH=src python -m sberindex.reporting.verify_artifacts
PYTHONPATH=src python -m sberindex.detection.bocpd_audit --verify
PYTHONPATH=src python -m sberindex.external.real_event_registry --verify
PYTHONPATH=src python -m sberindex.external.collect_warm_season --check-only
```

Если этап завершился с ошибкой, основной запуск укажет файл в `run_logs/`. Ошибка SHA256 означает, что файл отличается от зарегистрированного снимка: сначала выясните происхождение изменения. Инструкция по новым экспериментам — в [CONTRIBUTING](../.github/CONTRIBUTING.md).

Презентация и локальный интерфейс поставляются как готовые артефакты; обычные вычислительные команды не пересобирают PDF/PPTX.

## Повтор в новой среде

```bash
python3.12 -m venv ../clean-env
../clean-env/bin/python -m pip install -r requirements-lock.txt
python reproduce_check.py --python ../clean-env/bin/python
```

Проверка создаёт временную копию без производных результатов и сравнивает всю основную ветку и три приложения. Путь Python сохраняет виртуальную среду; её фактические prefix/base_prefix проверяются. Эта версия проверена на том же Apple Silicon Mac за 620,15 с; Linux/Windows, сетевой изолированный запуск и полное переобучение основных фундаментальных моделей отдельно не подтверждены.

[Команда выпуска прогнозов](research/OPERATIONAL_WORKFLOW_REPORT.md) принимает решение, задержку и новую сопоставимую историю.


## Дополнительная калибровка прогнозных интервалов

```bash
PYTHONPATH=src python -m sberindex.forecasting.interval_calibration
```

Отдельный фиксированный эксперимент, вне основного 51-этапногоrefresh. [Настройки](../configs/interval_calibration.json), [описание результатов и ограничений](research/INTERVAL_CALIBRATION_REPORT.md).48 тестов в двух средах включают пять новых проверок хронологии/пропусков/контроля интервалов. Предыдущая полная пересборка сохраняет собственный исторический отчёт с 43 тестами.


Групповое приложение: `PYTHONPATH=src python -m sberindex.forecasting.grouped_intervals`. [Отчёт](research/GROUPED_INTERVAL_REPORT.md), [настройки](../configs/grouped_intervals.json). Запускается отдельно, не увеличивает число этапов основногоrefresh. На том историческом этапе общая тестовая команда выполняла 51 тест; исторические журналы сохраняют число тестов своего запуска.


## Сезонные варианты Prophet

```bash
PYTHONPATH=src python -m sberindex.forecasting.prophet_seasonality
```

Два фиксированных варианта обучаются заново, четыре процесса. [Настройки](../configs/prophet_seasonality.json), [отчёт](research/PROPHET_SEASONALITY_REPORT.md). Это отдельное приложение, вне основногоrefresh; исходный Prophet берётся из объявленного рабочего кэша. На том историческом этапе общая тестовая команда выполняла 54 теста.

## Новые проверки внешнего отзыва (6 октября)

```bash
python run_review.py --mode verify
python run_review.py --mode recompute
python run_pipeline.py --mode full --review
```

`verify` — только целостность сохранённого снимка. `recompute` фактически повторяет сильный Prophet, 2075-МО сравнение, новости, семь детекторов, foundation, иерархию, direct и соседей, поправку роста, короткую историю детекторов и новостные сроки. `record` создаёт новый снимок лишь после завершения всех обязательных блоков; не является проверкой воспроизводимости прежних чисел. См. [первую страницу жюри](JURY_START.md).

Linux CI устанавливает pinned зависимости, запускает тесты и SHA-проверку; научное полное переобучение Linux не выполнялось. Dockerfile предоставлен, локальная сборка не проверена из-за отсутствия daemon. Новые веса foundation скачиваются отдельно по закреплённым ревизиям.


## Выборочный повтор последнего исследования

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src python -m sberindex.detection.short_history_review
PYTHONPATH=src python -m sberindex.reporting.news_alert_lead
PYTHONPATH=src python -m sberindex.reporting.final_review_figures
# Отдельно, с сетью: RSS сбор не нужен для повторения закреплённых метрик
PYTHONPATH=src python -m sberindex.external.mchs_rss --start 2023-01-01 --end 2024-12-31
# Архивы из чистого Git после фиксации результатов
python tools/build_submission.py
```

Последний фактический повтор короткой синтетики и новостных сроков в отдельно установленной среде: 14/14 численных файлов и графиков совпали побайтно. Сетевые снимки повторяемости не гарантируют. Общая тестовая команда выполняет 109 тестов; Linux результат и точный кодовый коммит — `reports/linux_ci/latest_run.json`.

### Восстановленный архив МЧС

```bash
PYTHONPATH=src python -m sberindex.external.mchs_archive
```

Обрабатывает сохранённые ответы официального поискового индекса без сети. Результаты — `reports/mchs_archive_review/`; правила допуска дат и ограничения — `docs/protocols/MCHS_ARCHIVE_REVIEW.md`. Исходный сетевой журнал RSS сохранён отдельно.

## Итоговые дополнительные проверки

```bash
PYTHONPATH=src python -m sberindex.forecasting.growth_sensitivity_review
PYTHONPATH=src python -m sberindex.detection.peer_residual_review
PYTHONPATH=src python tools/prepare_presentation_data.py
```

Первые два расчёта используют сохранённые индивидуальные прогнозы; они не требуют нового обучения foundation-моделей. Слайды: `artifacts/presentation.pdf`, редактируемый исходник `artifacts/presentation.pptx`. Для пересборки оформления нужны Node.js и предоставляемый Codex пакет `@oai/artifact-tool`; эти зависимости не нужны для запуска прогнозных моделей. Подробно — [сборка слайдов](PRESENTATION_BUILD.md).

## Расширенные проверки и локальный интерфейс

Сохранённые сравнения можно проверить в основной закреплённой среде без установки новых моделей:

```bash
PYTHONPATH=src python tools/verify_expansion_evidence.py
PYTHONPATH=src python -m sberindex.forecasting.foundation_expansion_review --report
PYTHONPATH=src python -m sberindex.forecasting.foundation_expansion_reporting
PYTHONPATH=src python -m sberindex.external.news_event_extraction_review
PYTHONPATH=src python -m sberindex.external.news_event_field_review
PYTHONPATH=src python -m sberindex.external.news_event_expanded_review
PYTHONPATH=src python -m sberindex.external.news_forecast_diagnostic_review
PYTHONPATH=src python -m sberindex.detection.event_conditioned_review
PYTHONPATH=src python -m sberindex.detection.yoy_shift_review
PYTHONPATH=src python tools/build_research_dashboard.py
```

Эти команды пересчитывают производные результаты из сохранённых прогнозов и исходных ответов Qwen. Они не повторяют нейросетевое обучение или генерацию текстовых ответов. Полное повторение новых TimesFM/Moirai/TiRex требует отдельной среды из `requirements-foundation-expansion.txt` и точных исходных версий из [протокола](protocols/FOUNDATION_EXPANSION_REVIEW.md). Для нового локального извлечения событий нужны зависимости `requirements-news-review.txt` и веса закреплённой Qwen: [протокол](protocols/NEWS_EVENT_EXTRACTION_REVIEW.md). Устанавливайте дополнительные модели в отдельную среду: Moirai требует другой версии PyTorch. Веса не распространяются в Git; ревизии и SHA256 сохранены. Дообучение Bolt использует только 2023 год, современное предобучение этих моделей остаётся ограничением ретроспективного теста.

```bash
# Повтор 2500 выборок МО и регионов; занимает заметное время
PYTHONPATH=src python -m sberindex.forecasting.forecast_uncertainty_review
# Категории: сохранённые промежуточные обучения повторно используются
PYTHONPATH=src python -m sberindex.forecasting.category_growth_review --full-pooled
# Локальный сервер для интерфейса, доступен только этому компьютеру
python -m http.server 8765 --bind 127.0.0.1
```

Откройте `http://127.0.0.1:8765/dashboard/` или непосредственно `dashboard/index.html`. Интерфейс работает без сетевых запросов за данными. Пересборка `data.js` нужна только при изменении результатов. Просмотр результатов и подсказки аналитика не являются онлайн-выпуском новых прогнозов.

Полный повтор категорий заново обучает Prophet при отсутствии совместимых промежуточных файлов. Эти локальные файлы исключены из Git и архива; конечные прогнозы, конфигурация и идентификаторы исходных данных включены. Смена входов или конфигурации отклоняет старый кэш вместо его молчаливого использования.

## Публикация интерфейса

Публичный сайт отключён по решению владельца. Используйте локальный `dashboard/index.html` или HTTP-инструкцию выше. Workflow `dashboard-pages.yml` сохранён и деактивирован; подготовка материалов возможна без публикации: `python tools/prepare_pages_site.py`.
