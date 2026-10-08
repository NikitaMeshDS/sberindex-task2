# Fixed hierarchical seasonal profile experiment

Declared 2026-10-06 before computing metrics. Fit on all municipalities with finite positive January–December2023 facts in «Все категории». Normalize summed2023 municipal expenses by the12-month mean to form global and regional profiles. Forecast uses the origin's observed level times target/origin ratio of an arithmetic mixture of normalized profiles. Fixed regional weights0,.25,.5,.75; no tuning/selection on2024. Regions with fewer than10eligible municipalities fall back to global profile. No claim that summed per-capita values equal regional monetary turnover.

Evaluate exactly the stored common pairs from reports/prophet_seasonality/predictions.parquet (frozen256ID sample, lag0,h1/3/6/12); assert actual values equal source. Report date-balancedMAE, pooled/withinR2,YoYMAE/R2,skill against global profile. h12 one date cannot establish temporal generalization. The2024 geographic dictionary is retrospective if2023 geography is absent; state it explicitly, no point-in-time geography claim. Spatial forecasting requires a dated compatible municipal adjacency graph; otherwise do not fabricate one.

Separate new module/config/artifacts only. Preserve full cohort and Prophet implementations and results.
