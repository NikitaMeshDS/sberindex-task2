# Полевая валидация тех же native ответов Qwen

```json
{
  "status": "complete",
  "new_inference": false,
  "retrospective_extractor_development": true,
  "strict_original_accepted": 2,
  "field_variant_accepted": 3,
  "municipal_exposures": 2,
  "municipality_ids": [
    1673,
    1829
  ],
  "event_date_inference_allowed": false,
  "threshold_factor": 0.75,
  "threshold_tuned_to_cases": false,
  "manual_registry_used": false,
  "municipal_method": "dictionary NER inside exact native LLM quote; native municipality_name was null",
  "absence_label": "unlabeled",
  "real_false_alarm_rate": null,
  "causality": false,
  "input_sha256": {
    "src/sberindex/external/news_event_field_review.py": "bfda87a55e023de37f80cb8771804ce2c7778a7c5a7c51b02fdfb4d18cf89cb7",
    "data/external/news_event_extraction_review/llm_outputs.json": "6f6c4db54c0ed7e94b54c4d41465c96adabff428d07100edbce1afc12400851c",
    "data/external/news_event_extraction_review/extraction_inputs.json": "9ca4afd748fb697901ad160c31e14b8e1820a2dbfaf2bed460f6d6fd4f0f7363",
    "reports/peer_residual_review/detector_panel.parquet": "9aec397868fb411a6c02460122da3d0296796a2d6566c9ea58d11894a1ab0cb1",
    "results/municipal_lookup.csv": "0164a1d6479cce33fb7affe2e2b0b05c850bf93e2a32b9e84b1fdcd815aefb91"
  },
  "municipal_ids_in_detector_panel": [
    1673
  ],
  "municipal_ids_absent_detector_panel": [
    1829
  ]
}
```

```csv
policy,release_lag_months,observed_months,event_available_months,active_alert_months,episodes,active_per100,unlabeled_active_months,real_false_alarm_rate
baseline,0,24282,2,148,145,0.6095049831150646,148,
llm_quote_event_0.75,0,24282,2,148,145,0.6095049831150646,148,
global_0.75,0,24282,2,530,515,2.1826867638580016,530,
baseline,1,24282,2,148,145,0.6095049831150646,148,
llm_quote_event_0.75,1,24282,2,148,145,0.6095049831150646,148,
global_0.75,1,24282,2,530,515,2.1826867638580016,530,
baseline,2,24282,2,148,145,0.6095049831150646,148,
llm_quote_event_0.75,2,24282,2,148,145,0.6095049831150646,148,
global_0.75,2,24282,2,530,515,2.1826867638580016,530,

```

Сохранены исходные строгие результаты 2/12. Вариант повторно разбирает ровно те же 12 сохранённых native ответов: неподтверждённая дата становится null с field_errors, а поддержанные тип и точная цитата сохраняются. Неподдержанные цитаты/типы остаются отклонёнными. ID разрешается только уникальным словарным NER внутри буквальной модельной цитаты, с узкими русскими окончаниями; это гибридная постобработка, не native геокодирование Qwen.

Развитие экстрактора ретроспективное, после наблюдения прежних отказов; независимая точность не заявляется. Порог EWMA не менялся: factor 0.75, публикация + 1 день, окно 2 месяца, отдельная LLM-quote экспозиция против baseline и global 0.75. Ручной реестр не используется в этом варианте. Неизвестный onset не восстанавливается из даты публикации. Событийная экспозиция не является границей расходов или причинным эффектом; отсутствие источника не является отрицательной меткой/FAR.

МО вне исходной detector-панели: [1829]; в нагрузку входят только наблюдаемые месяцы с доступной экспозицией. Неизвестные дата события и confidence остаются null.
