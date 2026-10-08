# Дополнительная партия локальной Qwen

Исходные 12 native ответов и строгие результаты сохраняются. Дополнительная партия включает все оставшиеся сохранённые официальные тела страниц с установленными датой публикации и региональной привязкой: 12 page-open выдержек прежнего реестра, 3 исходных MЧС HTML articleBody/datePublished и 2 сохранённые проверенные страницы предупреждений. Полнота page-open выдержек не предполагается; вход ограничен теми же 5500 символами, что в пилоте.

Чистые поисковые архивные snippets исключаются. 34 тела официальных Росгидромет-страниц не имеют надёжной одной региональной привязки и не передаются в муниципальную партию. Контекст региона берётся из прежнего реестра или номера регионального поддомена МЧС; для multi-region page сохраняется первая прежняя привязка, поэтому все региональные экспозиции одной страницы не восстанавливаются. 29 суммарных тел не являются полной лентой 2023/24.

Та же Qwen2.5-1.5B revision, тот же сохранённый PROMPT и SHA, torch threads=1, float16 MPS/CPU, deterministic max_new_tokens=350. Модель и токенизатор загружаются из локального кеша с local_files_only=True; новые веса/промпты не вводятся. Партия сохраняет checkpoint native_outputs.json и при возобновлении обрабатывает только ещё не сохранённые источники. Runtime versions, metadata dates, source hashes и full input capture manifest сохранены.

Field validator и quote-only dictionary NER совпадают с отдельным вариантом проверки полей. Unsupported date -> null + field_errors; unsupported type/quote -> rejection; неоднозначная география -> abstention. Чистая уверенность модели не калибрована. Тип предупреждения/новостного контекста не является доказательством физического onset или перелома расходов. Численная абляция использует только этот автоматический реестр: EWMA frozen factor 0.75, publication+1day scenario, окно 2 месяца, лаг релиза 0/1/2, baseline и global0.75 всей панели. Ни ручной реестр, ни подбор по Орску не применяются. Отсутствие новости остаётся неразмеченным.

Полный inference/checkpoint: `HF_HUB_OFFLINE=1 PYTHONPATH=src ../venv/bin/python -m sberindex.external.news_event_expanded_review --infer`.
Replay сохранённых ответов без inference: `PYTHONPATH=src ../venv/bin/python -m sberindex.external.news_event_expanded_review`.
CSV contract: `reports/news_event_extraction_review/expanded_batch/structured_events.csv` — те же колонки, что в `field_validation_variant/structured_events.csv`; принадлежность партии в отдельном `batch_labels.csv`.
