# Real event registry experiment

Protocol recorded at 2026-10-05T11:44:01.500992+00:00 before source searches.

## Selection and freeze
Select 5–8 distinct local hazard, infrastructure, flood or weather episodes occurring July–December 2024, outside January–June calibration. Use official public government primary sources only. Prefer six episodes with distinct dates and municipal areas; do not split one episode into multiple cases. Selection must depend only on public event evidence, never expenses, forecasts, alarms, residuals or errors. No consumption.parquet or expense forecasts/alarms may be read. Geography dictionaries may only be consulted after episode selection, if needed; outputperiphery must not be accessed.

An included matching case requires an identifiable official publication date and an explicit event date or period plus municipal geography. Unknown or unproven event start/end remain blank. Publication timing is never silently substituted for onset. Record rejected or unknown candidates separately with reasons. Time precision distinguishes day, period, month and unknown. Category hints describe plausible mechanisms, not observed expenditure effects.

## Capture and provenance
Preserve official raw HTML/text source snapshots where obtainable and a manifest containing URL, retrieval UTC timestamp, SHA-256 and capture type. If HTTP fetching is blocked, a saved web-tool text extraction is explicitly marked as such and is not claimed to be raw HTML. Preserve source evidence notes and limitations. Freeze registry selection after collecting evidence and before any expense matching; record freeze UTC timestamp.

## Interpretation
The registry contains external events, not labels of expenditure changes. Reusing 2024 does not constitute an independent holdout. The July–December period avoids January–June calibration but does not establish causal impact, complete event coverage, predictive validity or unbiased generalization. Local events may overlap, affect areas beyond named municipalities, have ambiguous onset or be reported after occurrence. Inclusion and non-inclusion cannot establish presence or absence of spending effects.

## Outputs
Only this protocol and new files under data/external/real_event_registry/ are authored by this collection task. Registry CSV fields: event_id,event_type,title,published_at,event_start,event_end,event_time_precision,region,municipality_names,source_url,snapshot_path,possible_categories,evidence_note,limitations. Separate rejected/unknown cases CSV and capture manifest accompany source snapshots. No pipeline execution or modifications to shared code, reports or source configuration.
