# Дополнительные сохранённые официальные страницы

```json
{
  "status": "complete",
  "new_inference_documents": 17,
  "original12_preserved": true,
  "combined_documents": 29,
  "combined_accepted": 5,
  "expanded_accepted": 2,
  "municipal_exposures": 2,
  "municipality_ids": [
    1673,
    1829
  ],
  "ids_absent_detector_panel": [
    1829
  ],
  "threshold_factor": 0.75,
  "threshold_tuned": false,
  "retrospective_extractor_development": true,
  "publication_vintage_verified": false,
  "full_corpus": false,
  "absence_label": "unlabeled",
  "event_onset_inferred": false,
  "real_false_alarm_rate": null,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "input_sha256": {
    "src/sberindex/external/news_event_expanded_review.py": "1996023b9e48e7bc56bc8d79ec6361e40b0c835ce42530a1848980d3d9c68b93",
    "src/sberindex/external/news_event_field_review.py": "bfda87a55e023de37f80cb8771804ce2c7778a7c5a7c51b02fdfb4d18cf89cb7",
    "data/external/news_event_extraction_review/expanded_batch/batch_manifest.json": "6daba6dfa92d68f4bb2ebae3af85293c6e990e16a3a9394896abd0e3d90d3de2",
    "data/external/news_event_extraction_review/expanded_batch/native_outputs.json": "e89dff84c6e1bc3add8489456f55fc870568be5d856adbcbde39107b3516c15f",
    "data/external/news_event_extraction_review/llm_outputs.json": "6f6c4db54c0ed7e94b54c4d41465c96adabff428d07100edbce1afc12400851c",
    "data/external/news_event_extraction_review/extraction_inputs.json": "9ca4afd748fb697901ad160c31e14b8e1820a2dbfaf2bed460f6d6fd4f0f7363",
    "data/external/regional_news_review/registry.csv": "3b3913470906f6bd0cf88ce410573988295f859987858c661ff41dbcb5e5bf48",
    "results/municipal_lookup.csv": "0164a1d6479cce33fb7affe2e2b0b05c850bf93e2a32b9e84b1fdcd815aefb91",
    "docs/protocols/NEWS_EVENT_EXPANDED_REVIEW.md": "0faf26ccf8c97f9acea0fd78073316c3e5be0a6647ace3c3f138c128f0454922",
    "reports/peer_residual_review/detector_panel.parquet": "9aec397868fb411a6c02460122da3d0296796a2d6566c9ea58d11894a1ab0cb1"
  },
  "coverage_reason_counts": {
    "historical_search_index_not_saved_article_body": 80,
    "official_weather_body_but_no_reliable_single_region_binding": 34,
    "official_saved_page_body_with_visible_publication_date": 14,
    "already_original12_preserved": 12,
    "not_official_mchs_host": 3,
    "official_HTML_articleBody_and_datePublished": 3
  },
  "runtime_versions": {
    "torch": "2.14.0",
    "transformers": "5.17.0",
    "huggingface_hub": "1.33.0"
  },
  "saved_capture_completeness": "raw HTML articleBody for 3 documents; other official saved page-body excerpts; input capped 5500 characters"
}
```

```csv
policy,release_lag_months,observed_months,event_available_months,active_alert_months,episodes,active_per100,unlabeled_active_months,real_false_alarm_rate
baseline,0,24282,2,148,145,0.6095049831150646,148,
expanded_llm_quote_0.75,0,24282,2,148,145,0.6095049831150646,148,
global_0.75,0,24282,2,530,515,2.1826867638580016,530,
baseline,1,24282,2,148,145,0.6095049831150646,148,
expanded_llm_quote_0.75,1,24282,2,148,145,0.6095049831150646,148,
global_0.75,1,24282,2,530,515,2.1826867638580016,530,
baseline,2,24282,2,148,145,0.6095049831150646,148,
expanded_llm_quote_0.75,2,24282,2,148,145,0.6095049831150646,148,
global_0.75,2,24282,2,530,515,2.1826867638580016,530,

```

В дополнительную партию включены все оставшиеся сохранённые официальные страницы с подтверждённой датой публикации и содержательным телом: 3 исходных HTML articleBody и 14 сохранённых выдержек из открытых страниц. Полнота выдержек не заявляется. Поисковые сниппеты и навигация исключены. Совпадают фиксированная ревизия Qwen, промпт, детерминированная генерация максимум 350 токенов и ограничение входа 5500 символов исходного пилота. Ограничение входа отмечено явно. Регион — прежняя сохранённая ручная привязка источника; для страницы с несколькими регионами взята первая сохранённая привязка, поэтому это не полный реестр всех региональных экспозиций страницы.

Исходные 12 ответов модели и строгие результаты сохранены. Объединённый вариант с проверкой отдельных полей заменяет неподтверждённые даты на null; муниципальный ID разрешается только уникальным словарём внутри точной модельной цитаты. Это ретроспективное развитие экстрактора, не независимая оценка. Отдельное сравнение EWMA всей панели с фиксированным множителем 0.75 не использует ручной реестр и не настраивается по случаям. Экспозиция не является разрывом расходов или причинным эффектом; отсутствие новости остаётся неразмеченным.
