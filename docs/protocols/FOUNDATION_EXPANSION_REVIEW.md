# Расширение foundation benchmark

Фиксация 07.10.2026, исследовательское расширение после просмотра 2024. Ключи: reports/presentation_comparison_cases/common_pairs.csv (7532 строк), только «Все категории». Горизонты 1/3/6/12;12/10/7/1 целевых месяцев;251/251/252/252 разных МО. Проверяем полноту и отсутствие дублей каждой модели; внутреннее пересечение не используется для скрытого уменьшения покрытия.

TimesFM2.5 google/timesfm-2.5-200m-pytorch, Moirai2 Salesforce/moirai-2.0-R-small, TiRex NX-AI/TiRex. Точные model и source revisions в configs/foundation_expansion_review.json. Установки и исключения в reports/foundation_expansion_review. Отдельное окружение work/foundation_expansion_env: uni2ts требует torch<2.5. Основное окружение и lock не изменяются. CPU,2 потока, batch32, одна модель одновременно. Ограничение памяти 10GiB проверяется по process peak RSS. Checkpoints после 2024; происхождение предобучения не доказывает отсутствие SberIndex, контаминация неизвестна.

Фиксированные режимы: zero-shot, отношение к 2023 сезонному профилю, ансамбль 50/50 zero-shot+seasonal pooled. Профиль суммы по всем положительным полным 2023 МО нормирован средним; уровень МО mean(value2023/profile2023); отношение value/(уровень×профиль), вход обрезан поorigin. Прогноз отношения умножается на 2023 уровень и повторяемый календарный профиль.2024 значения послеorigin не входят в подготовку; фактическое значениеtarget используется только в оценке.

TiRex dynamic_padding=True — документированный режим короткой истории; его влияние на качество отмечается. Moirai2 context32 и фактическая история 12–23 месяцев; дополнение выполняет официальный predictor. TimesFM context32,horizon12, медиана, compile=False. Параметры не подбираются после оценки.

Chronos-Bolt: обучающие значения Jan–Sep2023, validation Oct–Dec2023;20 фиксированных шагов AdamW lr 1e-5, batch16,seed 2023; без early stopping и 2024 validation. Перед 2024 inference checkpoint фиксируется. Это короткий бюджетный пилот, не поиск гиперпараметров.

Полная панель:2075 МО по положительной полной истории 2023; число фактически оцениваемыхМО и пар меняется поorigin/target. Предыдущие Prophet/seasonal из full_cohort_review сохранены; новый fullpanel сравнивается на явно сохранённых ключах и покрытии, отдельным набором от 7532 frozen keys.

Официальные документы: https://github.com/google-research/timesfm; https://github.com/SalesforceAIResearch/uni2ts; https://github.com/NX-AI/tirex; https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training.

## Результат выполнения и воспроизведение

Три новые модели успешно установлены и выполнили все 7532 ключа в трех режимах. TiRex PyPI1.4.2 отклонил документированный dynamic_padding; установлен точный upstream Git revision из config, ошибка и повторная установка сохранены. Все девять новых режимов проиграли seasonal pooled на всехh. Это результат для короткой месячной истории, не универсальное утверждение о foundation models.

Все десять ранее закэшированных моделей расширены на 60700 ключей полного 2023 eligible-panel (добавлен direct_context как 11-й дополнительный режим). Фактическое покрытиеh1/3/6/12:2031/2031/2029/2023 МО и 24282/20233/14162/2023 пары;12/10/7/1 целевых месяцев. Chronos2 univariate полный overlap7532 воспроизводится точно; direct HGB7280 обучаемых overlap пар maxdiff<2e-10.

Первичный 20-step Bolt pilot покрывает 300 случайно выбранных МО из 2075 eligible. Для выполнения требования обучения по всей панели дополнительно зафиксирован один полный проход по 2075 МО (batch 128,17 batches, lr 1e-5, seed 2023). Это продолжение исходного требования полного 2023 обучения; оба результата сохраняются,2024 метрики не выбирают бюджет или checkpoint. Полный проход читает толькоJan–Sep2023 для обучения иOct–Dec2023 для проверки без early stopping. Validation всего 2075 МО, frozen checkpoint до 2024 inference. Этот протокол уточнен после 20-step pilot; независимого holdout нет.

Запуск изwork/submission:

```sh
PYTHONPATH=src ../foundation_expansion_env/bin/python -m sberindex.forecasting.foundation_expansion_review --model timesfm25
PYTHONPATH=src ../foundation_expansion_env/bin/python -m sberindex.forecasting.foundation_expansion_review --model moirai2
PYTHONPATH=src ../foundation_expansion_env/bin/python -m sberindex.forecasting.foundation_expansion_review --model tirex
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src ../foundation_expansion_env/bin/python -m sberindex.forecasting.foundation_expansion_direct
PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_expansion_fullpanel --stage chronos2_univariate_full
PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_expansion_fullpanel --stage chronos2_profile_full
PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_expansion_fullpanel --stage bolt_finetune
PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_expansion_fullpanel --stage bolt_finetune_epoch1
PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_expansion_review --report
PYTHONPATH=src ../venv/bin/python -m sberindex.forecasting.foundation_expansion_reporting
PYTHONPATH=src ../venv/bin/python -m unittest discover -s tests -p test_foundation_expansion_review.py -v
```

CPU2 threads для foundation моделей; direct HGB требует OMP1 из-за наблюдавшегося ожидания наOpenMP barriers при 2 threads. Установлен один checkpoint в каждый момент, peak RSS<10GiB. Версии основного и отдельного окружений зафиксированы отдельно. Исходные checkpoint downloads и trained checkpoints являются локальными весами; versionable predictions иsha256-манифесты сохраняются вreports.

Официальная документация fine-tuning Bolt: https://auto.gluon.ai/1.4.0/tutorials/timeseries/forecasting-chronos.html#fine-tuning; реализован бюджетный цикл через установленныйChronosBoltModule.forward(context,target) и его официальный quantile loss, без установкиAutoGluon в основной environment.
