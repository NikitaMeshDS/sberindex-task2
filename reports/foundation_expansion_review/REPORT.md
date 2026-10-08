# Расширение foundation benchmark

Все девять фиксированных новых вариантов уступают seasonal pooled на каждом из четырех горизонтов. Остаточное отношение уменьшает ошибку zero-shot, но этого недостаточно для превосходства сезонного профиля. Короткая месячная история 12–23 точки ограничивает выводы о переносимости моделей.

Сравнение проводится на точных замороженных ключах: h=1: 251 МО/12 целей, h=3: 251/10, h=6: 252/7, h=12: 252/1. Это ретроспективная исследовательская проверка, не независимый holdout.

Новые checkpoints выпущены после 2024: отсутствие данных SberIndex в предобучении не подтверждено. Возможная контаминация неизвестна.

Профиль и масштаб остаточного отношения оцениваются только по 2023; вход истории заканчивается origin. Seasonal ratio восстанавливается с фиксированным профилем 2023. Ансамбль фиксирован 50/50 с seasonal pooled; параметры не подбирались по 2024. TiRex использует документированный dynamic_padding=True для короткой истории; это ограничение сопоставимости. PyPI1.4.2 отклоняет этот аргумент; успешно применена установка точной upstream Git revision, ошибка сохранена. TimesFM 2.5 используется через PyTorch на CPU; современная документация native MLX относится к 3.0. Direct HGB h12 в cached reference — явно не обучаемый seasonal fallback.

| Модель | h1 MAE | h3 MAE | h6 MAE | h12 MAE |
|---|---:|---:|---:|---:|
| bolt_base_univariate | 2416.25 | 2776.71 | 4353.32 | 10586.53 |
| bolt_finetune | 1897.65 | 2514.36 | 4040.20 | 10857.27 |
| bolt_finetune_epoch1 | 1885.67 | 2366.83 | 3900.65 | 10823.62 |
| chronos2_profile | 1937.87 | 2575.89 | 3599.80 | 9125.36 |
| chronos2_univariate | 2176.66 | 2734.00 | 4063.96 | 9664.99 |
| direct_lags | 2165.56 | 1251.21 | 3188.61 | 4853.75 |
| hierarchy05 | 926.99 | 1513.07 | 1906.70 | 4853.75 |
| hierarchy05_growth_bridge | 921.32 | 1245.51 | 1338.14 | 1357.02 |
| moirai2_ensemble50 | 1380.94 | 2076.84 | 2401.08 | 8733.28 |
| moirai2_seasonal_ratio | 1286.69 | 1801.28 | 2302.07 | 7369.27 |
| moirai2_zero_shot | 2534.98 | 2972.50 | 3663.52 | 12612.81 |
| prophet_disabled | 1838.66 | 1774.12 | 2213.45 | 2930.61 |
| prophet_pooled_profile | 1616.01 | 1664.97 | 1937.35 | 3002.21 |
| prophet_yearly3 | 2402.43 | 2507.44 | 2935.85 | 7619.14 |
| seasonal_naive | 4226.89 | 4484.02 | 4482.63 | 4853.75 |
| seasonal_pooled | 979.14 | 1633.04 | 1913.64 | 4853.75 |
| timesfm25_ensemble50 | 1172.01 | 1901.10 | 2554.24 | 6993.84 |
| timesfm25_seasonal_ratio | 1146.59 | 2010.20 | 2700.59 | 4966.17 |
| timesfm25_zero_shot | 2078.02 | 2694.83 | 4147.94 | 9133.92 |
| tirex_ensemble50 | 1295.17 | 2124.51 | 3104.75 | 8613.40 |
| tirex_seasonal_ratio | 1204.47 | 2359.80 | 3518.66 | 5046.87 |
| tirex_zero_shot | 2367.54 | 3232.11 | 5321.39 | 12373.06 |

Официальные источники: [TimesFM](https://github.com/google-research/timesfm), [Moirai 2](https://github.com/SalesforceAIResearch/uni2ts), [TiRex](https://github.com/NX-AI/tirex).

Chronos-Bolt: один фиксированный epoch, все 2075 муниципалитетов 2023,17 batches × 128 (последний неполный), lr 1e-5, seed 2023. Обучение Jan–Sep2023: context Jan–Jun иtargets Jul–Sep; validation Oct–Dec2023 для всех 2075. Без early stopping и 2024 validation; checkpoint_sha256 вbolt_epoch1_training.json фиксирован до 2024 inference.20-step pilot (300 МО) сохранен отдельно; бюджет одного epoch уточнен для выполнения требования полной панели, оба результата публикуются без выбора по 2024.

[Официальная документация fine-tuning Bolt](https://auto.gluon.ai/1.4.0/tutorials/timeseries/forecasting-chronos.html#fine-tuning). Бюджетный цикл использует официальный quantile loss установленногоChronosBoltModule.forward(context,target).
