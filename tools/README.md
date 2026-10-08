# Сборка и проверки

- `prepare_pages_site.py` — автономный сайт и отчёт HTML.
- `test_dashboard.cjs` — браузерные сценарии интерфейса.
- `build_submission.py` — архивы сдачи с контрольными суммами.
- `download_research_snapshot.py` — загрузка полного исследовательского комплекта.
- `presentation_claim_checks.py` — проверка численных утверждений по сохранённым данным полного архива.
- `build_final_presentation.mjs` — редактируемый PPTX; [инструкция](../docs/PRESENTATION_BUILD.md).

Для прогноза на новых данных используйте `python predict.py --config configs/predict.json`: [инструкция](../docs/PREDICT_NEW_DATA.md). Скрипты ранних этапов находятся в полном архиве Releases.
