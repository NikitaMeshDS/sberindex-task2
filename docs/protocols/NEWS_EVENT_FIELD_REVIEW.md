# Вариант абстенции по полям без повторной LLM-инференции

Replay тех же 12 native ответов сохранённой Qwen1.5B. Строгий исходный эксперимент не изменяется. Неподтверждённый event_date становится null с явным field_errors; unsupported type/quote отклоняют запись. Дата публикации/регион остаются авторитетными метаданными источника. Неизвестная confidence — null.

Native municipality_name в пилоте null. Для пригодных цитат применяется отдельный официальный словарный NER только внутри буквальной цитаты, с узкими окончаниями: городской «Орск» -> «Орске», прилагательное «Октябрьский» -> «Октябрьского». Список кандидатов строится в регионе/году; несколько кандидатов означают abstention. Переформированные ID и география не объединяются. Это гибрид LLM selection quote + dictionary NER, не native Qwen геокодирование.

Разработка варианта ретроспективная, после наблюдения отказов. Результаты не являются независимой оценкой точности. В отдельном numeric replay используется только этот автоматически разрешённый LLM-quote реестр: frozen EWMA threshold factor0.75, publication+1day availability scenario, 2-month window, release lags0/1/2, baseline и global0.75 всей панели. Ручной реестр не подмешивается; пороги не меняются для Орска. Event_date не восстанавливается, exposure не является spending changepoint truth, отсутствие новости остаётся unlabeled.

`PYTHONPATH=src ../venv/bin/python -m sberindex.external.news_event_field_review`
