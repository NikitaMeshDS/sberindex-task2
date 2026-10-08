# Исследовательские утилиты

`detector_tradeoffs.py` строит отдельное описательное приложение из зафиксированного сопоставимого сравнения детекторов. Запускается из корня проекта с установленными зависимостями:

```bash
python tools/detector_tradeoffs.py
python tools/detector_tradeoffs.py --verify
```

Выход: `reports/detector_tradeoffs/`. Входы и код защищены SHA256 в `audit.json`. Утилита входит в единый refresh, не обучает модели и не выбирает подтверждённого победителя.

`residual_feature_experiment.py` обучает отдельные Ridge/HGB поправки только на2023, сравнивает восемь вариантов на точных исходных парах и сохраняет hashes/MAE/R²/WAPE в reports/residual_features. Настройки — configs/residual_features.json. Историческая доступность зарплатного снимка не подтверждена.

```bash
python tools/residual_feature_experiment.py
python tools/residual_feature_experiment.py --verify
python tools/plot_residual_features.py
```

Эти утилиты входят в единый refresh. Научная пара PNG/SVG приложения считается отдельно; существующие201/29 не переименовываются.

`operational_workflow.py` выполняет задержанное сравнение HGB, фиксированный прикладной выбор детектора и выпуск прогнозов. `plot_operational_workflow.py` строит научную пару PNG/SVG. [Рабочие правила и результаты](../docs/research/OPERATIONAL_WORKFLOW_REPORT.md).

```bash
python run_pipeline.py --mode refresh
python tools/operational_workflow.py --verify
python tools/operational_workflow.py --decision 2024-12 --lag 1 --output forecast.csv
```

Презентация остаётся отдельной опцией. Новые задержанные прогнозы обучаются заново; основной кэш объявлен в refresh_inputs.json.


`statistical_evidence_audit.py` — отдельный аудит сохранённых прогнозов: парность фактов, MAE, повтор исходного bootstrap и исключение любых одной/двух дат. Он не входит в основной refresh, не обучает модели и не доказывает статистическую значимость.

```bash
python tools/statistical_evidence_audit.py
```

[Сила доказательств](../docs/research/STATISTICAL_EVIDENCE_REPORT.md); результаты и SHA256 — reports/statistical_evidence/.


Дополнительная калибровка интервалов запускается отдельно: `PYTHONPATH=src python -m sberindex.forecasting.interval_calibration`. [Протокол](../docs/protocols/INTERVAL_CALIBRATION_EXPERIMENT.md), [результаты](../docs/research/INTERVAL_CALIBRATION_REPORT.md), настройки — configs/interval_calibration.json. Основной refresh и рабочий выпуск не меняются.


Групповая калибровка интервалов: `PYTHONPATH=src python -m sberindex.forecasting.grouped_intervals`. [Протокол](../docs/protocols/GROUPED_INTERVAL_EXPERIMENT.md), [отчёт](../docs/research/GROUPED_INTERVAL_REPORT.md), настройки — configs/grouped_intervals.json. Отдельное приложение, старые расчёты сохраняются.
