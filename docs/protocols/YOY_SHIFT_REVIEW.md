# Изменение роста год к году

Предварительная формула: log2024/2023 по МО минус leave-one-out median остальных МО региона (>=5), center Jan–Mar2024. Калибровка99% score всей панели Jan–Mar, без выбора отдельных случаев. CUSUM drift0.01, EWMA alpha из общего detector config. Monitoring Apr–Dec: будущая информация не влияет на центр/порог/прошлый score. Для Jan–Mar scores допускается информация всех калибровочных месяцев — это calibration, не online performance.

Кейсы ранее видели участники проекта: калибровка ретроспективная, независимый frozen holdout не заявляется. Высокая нагрузка может опровергнуть полезность короткой калибровки; порог после рассмотрения результата не изменяется. Synthetic null/spike/sustained/seasonal при том же empirical threshold — контекст, не доказанный real FAR.

Реформа: закон Москвы №13 от8мая2024 в официальном Вестнике №27, article1 об объединении Киевский -> Бекасово и Кокошкино -> Внуково. Article9 effective from official publication (некоторые исключения). Из месячного gazette точный publication day не доказан: поля event_date/available_from null; law_date отдельно. Юридическая реформа, дата выгрузки справочника, географический/статистический артефакт и реальный перелом расходов — разные метки.

Команда: `PYTHONPATH=src ../venv/bin/python -m sberindex.detection.yoy_shift_review`.


Синтетический сезонный контроль содержит сезонность с начала ряда, без введённой точки изменения. Его тревоги в месяцах 5–6 обозначены `reference_window_any_alert`, а `detected_by_second_postchange` остаётся пустым, как и для null. Для spike/sustained второе поле оценивает только два месяца после введённого изменения. Проверка: `PYTHONPATH=src python -m unittest discover -s tests -p test_yoy_context_labels.py`.
