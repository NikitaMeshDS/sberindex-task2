# Порог EWMA с доступным новостным контекстом

Заранее задан factor=0.75, window=2 месяца начиная с publication+1day месяца. Порог взят из frozen short_history_review. Политики: baseline, named available event subset factor0.75, entire-panel factor0.75; никакой настройки по Орску. Exposure registry основан на сохранённых ручных официальных источниках, не на принятии LLM. Real false-alarm rate не оценивается: отсутствие источника — unlabeled. Лаг релиза0/1/2 не разрешает пересчитать targets до публикационного месяца.

Команда: `PYTHONPATH=src ../venv/bin/python -m sberindex.detection.event_conditioned_review`.
