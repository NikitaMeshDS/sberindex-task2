"""Point-in-time join of public news to monthly forecast origins.

The one-day delay is conservative. A post published during target month can
never enter a forecast made at the previous month-end.
"""

from pathlib import Path
import pandas as pd

from sberindex.paths import ROOT
root = ROOT
events = pd.read_csv(root / "data/external/news_events.csv", parse_dates=["published_date", "available_from"])
origins = pd.DataFrame({"origin": pd.date_range("2023-12-31", "2024-11-30", freq="ME")})
rows = []
for origin in origins.origin:
    known = events[events.available_from <= origin]
    recent = known[known.available_from > origin - pd.Timedelta(days=90)]
    rows.append({"origin": origin.strftime("%Y-%m"),
                 "last_known_rate_pct": known.rate_after_pct.iloc[-1] if len(known) else 7.5,
                 "hikes_last_90d_bps": int(recent.delta_bps.sum()),
                 "news_count_last_90d": len(recent),
                 "latest_news_date": known.published_date.iloc[-1].date().isoformat() if len(known) else ""})
out = pd.DataFrame(rows)
out.to_csv(root / "results" / "news_asof_features.csv", index=False)
assert out.loc[out.origin == "2024-06", "last_known_rate_pct"].item() == 16
assert out.loc[out.origin == "2024-07", "last_known_rate_pct"].item() == 18
assert out.loc[out.origin == "2024-08", "last_known_rate_pct"].item() == 18
assert out.loc[out.origin == "2024-09", "last_known_rate_pct"].item() == 19
print(out.to_string(index=False))
