# Frozen public event registry

Six source-only selected local episodes in five regions, with official publication days and explicit incident timing. Selection window: July–December 2024; selected episodes occur July, September and October. No uniform monthly coverage is implied. Selection freeze and non-holdout limitations are in selection_freeze.json.

registry.csv contains the prescribed schema. Dates have day precision; dispatch receipt is distinguished from exact onset in notes. Unknown ends remain blank. rejected_unknown.csv holds three examined sources rejected for insufficient onset evidence or forecast-only content. capture_manifest.json records URLs, UTC retrieval stamps, SHA-256, capture types and HTTP status.

*.web.txt files preserve the returned web-tool text extraction, with its provenance and line references. They are not raw HTML. *.http_response.html files preserve actual HTTP response bytes: non-200 response files are explicitly failed captures, never evidence. Existing official publication pages can contain later updates; in particular the Rostov source displays 12:40 publication while describing 16:40 liquidation. Captured content is therefore historical event evidence, not proof the complete article was available at its displayed publication time.

Possible categories are mechanism hints only and must not be represented as event labels of spending changes. Reused 2024 observations are not an independent holdout. Selection did not inspect expenses, predictions, alarms, errors, geography dictionary or outputperiphery.


Уточнение после проверки: в замороженном selection_freeze.json ошибочно написано «five regions». Фактически реестр содержит шесть различных регионов. Исходная запись фиксации не переписана; ошибка исправлена этим дополнением и в отчёте.
