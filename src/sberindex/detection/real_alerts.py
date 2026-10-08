"""Apply the selected detector to real 2024 observations.

Alerts are candidates, not confirmed structural changes. Road distances are
used only for ex-post interpretation, never as forecast features.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sberindex.detection.change_detection import CATEGORIES, panel_for, residuals, score


from sberindex.paths import ROOT
rows = []
for category in CATEGORIES:
    ids, values = panel_for(category)
    z = residuals(values, "level")
    statistic = score(z, "rolling_3m")
    threshold = np.quantile(statistic[:, :6].max(axis=1), 0.99)
    alarms = statistic[:, 6:] > threshold
    for i in np.flatnonzero(alarms.any(axis=1)):
        yoy = values[i, 12:] / values[i, :12] - 1
        rows.append({"category": category, "territory_id": int(ids[i]),
                     "first_alarm_month": f"2024-{int(alarms[i].argmax()) + 7:02d}",
                     "max_statistic_jul_dec": statistic[i, 6:].max(),
                     "threshold": threshold,
                     "median_relative_log_growth_jul_dec": np.median(z[i, 6:]),
                     "median_yoy_pct_jan_jun": 100 * np.median(yoy[:6]),
                     "median_yoy_pct_jul_dec": 100 * np.median(yoy[6:])})
alerts = pd.DataFrame(rows)
alerts.to_csv(ROOT / "results" / "real_alerts.csv", index=False)

# Two fixed illustrative IDs from the strongest total-spending alerts.
ids, values = panel_for("Все категории")
z = residuals(values, "level")
position = {int(id_): i for i, id_ in enumerate(ids)}
connection = pd.read_parquet(ROOT / "data" / "connection.parquet")
roads = connection.loc[connection.type == "highway"]
examples = []
for target, paired in [(2593, 2595), (2595, 2593)]:
    linked = roads.loc[(roads.territory_id_x == target) |
                       (roads.territory_id_y == target)].copy()
    linked["neighbor"] = np.where(linked.territory_id_x == target,
                                  linked.territory_id_y, linked.territory_id_x)
    close = (linked.loc[(linked.neighbor.isin(position)) &
                        (linked.neighbor != paired) & (linked.distance <= 50)]
             .sort_values("distance").drop_duplicates("neighbor").head(10))
    neighbors = [np.median(z[position[int(n)], 6:]) for n in close.neighbor]
    joint = roads.loc[(roads.territory_id_x == target) &
                      (roads.territory_id_y == paired)]
    if joint.empty:
        joint = roads.loc[(roads.territory_id_x == paired) &
                          (roads.territory_id_y == target)]
    examples.append({"territory_id": target,
                     "median_relative_log_growth_jul_dec": np.median(z[position[target], 6:]),
                     "nearest_10_neighbors_median": np.median(neighbors),
                     "neighbor_count": len(neighbors),
                     "paired_road_distance_km": joint.distance.min()})
pd.DataFrame(examples).to_csv(ROOT / "results" / "real_case_study.csv", index=False)
print(alerts.groupby("category").size().to_string())
print(pd.DataFrame(examples).to_string(index=False))
